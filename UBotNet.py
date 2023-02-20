import torch
from torch import einsum, nn

from botNet import BoTStack
import antialiased_cnns
import pytorch_lightning as pl
from torchsummary import summary


def calc_loss(noise,pred_noise):
    mse = nn.MSELoss()
    loss = mse(noise, pred_noise)
    return loss


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   groups=in_channels, bias=bias, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    
class UNet_down_block(torch.nn.Module):
    def __init__(self, input_channel, output_channel, down_size, emb_dim=256):
        super(UNet_down_block, self).__init__()
        self.conv1 = SeparableConv2d(input_channel, output_channel, 3)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = SeparableConv2d(output_channel, output_channel, 3)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.conv3 = SeparableConv2d(output_channel, output_channel, 3)
        self.bn3 = torch.nn.BatchNorm2d(output_channel)
        # self.max_pool = torch.nn.MaxPool2d(2, 2)
        antialiased_max_pool = [torch.nn.MaxPool2d(kernel_size=2,stride=1),antialiased_cnns.BlurPool(output_channel//2, stride=2)]
        self.max_pool = nn.Sequential(*antialiased_max_pool)
        self.relu = torch.nn.ReLU()
        self.down_size = down_size
        
        self.emb_layer = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    emb_dim,
                    output_channel
                ),
            )

    def forward(self, x, t):
        if self.down_size:
            x = self.max_pool(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x+emb
class UNet_up_block(torch.nn.Module):
    def __init__(self, prev_channel, input_channel, output_channel,emb_dim=256):
        super(UNet_up_block, self).__init__()
        # self.up_sampling = torch.nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        self.up_sampling = nn.ConvTranspose2d(input_channel,output_channel,kernel_size=2,stride=2)
        self.conv1 = SeparableConv2d(input_channel, output_channel, 3)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = SeparableConv2d(output_channel, output_channel, 3)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.relu = torch.nn.ReLU()
        self.emb_layer = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    emb_dim,
                    output_channel
                ),
            )
        
        
    def forward(self, prev_feature_map,x,t):
        # print(f'Before upsampling : {x.shape}')
        x = self.up_sampling(x)
        # print(f'After upsampling : {x.shape}')
        # print(f'Previous feature map : {prev_feature_map.shape}')
        x = torch.cat((x, prev_feature_map), dim=1)
        # print(f'After concat: {x.shape}')
        x = self.relu(self.bn1(self.conv1(x)))
        # print(f'After conv1 layer: {x.shape}')
        x = self.relu(self.bn2(self.conv2(x)))
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x+emb

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class UBotNetPLModel(pl.LightningModule):
    def __init__(self,base_depth=64,img_size=64) -> None:
        super().__init__()
        fmap_sz = int(img_size/16)
        # self.transform = DataAugmentation(apply_color_jitter=True)
        # self.geometric_transform = geometricTransform()
        self.down_block1 = UNet_down_block(3, base_depth, False)
        self.down_block2 = UNet_down_block(base_depth, base_depth*2, True)
        self.down_block3 = UNet_down_block(base_depth*2, base_depth*4, True)
        self.down_block4 = UNet_down_block(base_depth*4, base_depth*8, True)
        self.down_block5 = UNet_down_block(base_depth*8, base_depth*16, True)
        self.layer = BoTStack(dim=base_depth*16, fmap_size=(fmap_sz, fmap_sz),dim_out=base_depth*16, stride=1, rel_pos_emb=True)
        self.up_block1 = UNet_up_block(base_depth*8, base_depth*16, base_depth*8)
        self.up_block2 = UNet_up_block(base_depth*4, base_depth*8, base_depth*4)
        self.up_block3 = UNet_up_block(base_depth*2, base_depth*4, base_depth*2)
        self.up_block4 = UNet_up_block(base_depth, base_depth*2, base_depth)
        self.dense1 = torch.nn.Linear(base_depth,base_depth*4)
        self.dense2 = torch.nn.Linear(base_depth*4,base_depth)
        self.noise_layer = torch.nn.Linear(base_depth,3)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
    
    def forward(self,x):
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        self.x5 = self.down_block5(self.x4)
        self.x6 = self.layer(self.x5)
        self.x7 = self.up_block1(self.x4,self.x6)
        self.x7 = self.x7 + self.x4
        self.x8 = self.up_block2(self.x3,self.x7)
        self.x8 = self.x8 + self.x3
        self.x9 = self.up_block3(self.x2,self.x8)
        self.x9 = self.x9 + self.x2
        self.x10 = self.up_block4(self.x1,self.x9)
        self.x10 = self.x10 + self.x1
        self.x10 = torch.einsum('...ijk->...jki',self.x10)
        self.x11 = (self.relu(self.dense1(self.x10)))
        self.x11 = self.dropout(self.x11)
        self.x12 = (self.relu(self.dense2(self.x11)))
        self.x12 = self.dropout(self.x12)
        pred_noise = self.noise_layer(self.x12)
        # pred_noise = self.sigmoid(self.noise_layer(self.x12))
        pred_noise = torch.einsum('...ijk->...kij',pred_noise)
        return pred_noise
    
    def training_step(self, batch, batch_idx) :
        # batch = self.geometric_transform(batch)
        noise = batch
        # aug_inputs = self.transform(inputs)
        # pred_height = self(aug_inputs)
        pred_noise = self(noise)
        loss = calc_loss(noise,pred_noise)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
  
    def validation_step(self,batch,batch_idx):
        # batch = self.geometric_transform(batch)
        noise = batch
        # aug_inputs = self.transform(inputs)
        # pred_height = self(aug_inputs)
        pred_noise = self(noise)
        loss = calc_loss(noise,pred_noise)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        # inputs, gt_height,gt_normal = batch
        noise = batch
        pred_noise= self(noise)
        depth_error_metrics = calc_loss(noise,pred_noise)
        for key in depth_error_metrics:
            self.log(key,depth_error_metrics[key])
        return depth_error_metrics


    def configure_optimizers(self):
        optimizer= torch.optim.AdamW(self.parameters(), lr=1e-4,weight_decay=1e-5)
        return [optimizer]

class UBotNetModel(nn.Module):
    def __init__(self,c_in=3, c_out=3,base_depth=64,img_size=64, time_dim=256, device="cuda") -> None:
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        fmap_sz = int(img_size/16)
        self.inc = self.inc = DoubleConv(c_in, base_depth)
        self.down_block1 = UNet_down_block(base_depth, base_depth*2, True)
        self.down_block2 = UNet_down_block(base_depth*2, base_depth*4, True)
        self.down_block3 = UNet_down_block(base_depth*4, base_depth*8, True)
        self.down_block4 = UNet_down_block(base_depth*8, base_depth*16, True)
        self.bottleneck_layer = BoTStack(dim=base_depth*16, fmap_size=(fmap_sz, fmap_sz),dim_out=base_depth*16, stride=1, rel_pos_emb=True)
        self.up_block1 = UNet_up_block(base_depth*8, base_depth*16, base_depth*8)
        self.up_block2 = UNet_up_block(base_depth*4, base_depth*8, base_depth*4)
        self.up_block3 = UNet_up_block(base_depth*2, base_depth*4, base_depth*2)
        self.up_block4 = UNet_up_block(base_depth, base_depth*2, base_depth)
        self.dense1 = torch.nn.Linear(base_depth,base_depth*4)
        self.dense2 = torch.nn.Linear(base_depth*4,base_depth)
        self.noise_layer = torch.nn.Linear(base_depth,c_out)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
    def forward(self,x,t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        
        self.x1 = self.inc(x)
        self.x2 = self.down_block1(self.x1,t)
        self.x3 = self.down_block2(self.x2,t)
        self.x4 = self.down_block3(self.x3,t)
        self.x5 = self.down_block4(self.x4,t)
        self.x6 = self.bottleneck_layer(self.x5)
        self.x7 = self.up_block1(self.x4,self.x6,t)
        self.x7 = self.x7 + self.x4
        self.x8 = self.up_block2(self.x3,self.x7,t)
        self.x8 = self.x8 + self.x3
        self.x9 = self.up_block3(self.x2,self.x8,t)
        self.x9 = self.x9 + self.x2
        self.x10 = self.up_block4(self.x1,self.x9,t)
        self.x10 = self.x10 + self.x1
        self.x10 = torch.einsum('...ijk->...jki',self.x10)
        self.x11 = (self.relu(self.dense1(self.x10)))
        self.x11 = self.dropout(self.x11)
        self.x12 = (self.relu(self.dense2(self.x11)))
        self.x12 = self.dropout(self.x12)
        pred_noise = self.noise_layer(self.x12)
        # pred_noise = self.sigmoid(self.noise_layer(self.x12))
        pred_noise = torch.einsum('...ijk->...kij',pred_noise)
        return pred_noise
  
if __name__ == "__main__":
    model = UBotNetModel(base_depth=128,img_size=128).cuda()
    # inputData = (3,128,128)
    # print(summary(model,inputData))
    x = torch.ones(4, 3, 128, 128).cuda()
    y = model(x)
    print(y.shape)