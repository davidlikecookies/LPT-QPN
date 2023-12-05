import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numbers
from einops import rearrange
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils import CSI

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

        

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        out = (x-mu) / torch.sqrt(sigma+1e-5) * self.weight
        return torch.nn.functional.sigmoid(out)        
             

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = BiasFree_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


## Feed-Forward Network (FFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * nn.functional.sigmoid(x2)
        x = self.project_out(x)
        return x


##########################################################################
## Multi-Head Squared Attention (MHSA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)



    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature  
        
        attn = torch.nn.functional.sigmoid(attn)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Patch Embedding
class PatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(PatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


#Swish function
class Swish(nn.Module):
    def __init__(self,beta_init=1.0):
        super().__init__()
        self.beta=nn.Parameter(torch.tensor(beta_init,dtype=torch.float))
    
    def forward(self, x):
        
        return x*torch.sigmoid(self.beta*x)




class ConvectionDiffusionLoss(nn.Module):
    def __init__(self, alpha=1, beta=0.5, a=1,b1=1,b2=1,c=1):
        super(ConvectionDiffusionLoss,self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.a =a 
        self.b1=b1
        self.b2=b2
        self.c=c
              
    def forward(self,x,matrix):
        
        gradients_x = torch.abs(matrix[:,:,:,:-1]-matrix[:,:,:,1:])
        gradients_y = torch.abs(matrix[:,:,:-1,:]-matrix[:,:,1:,:])

        gradients_xx = torch.abs(matrix[:,:,:,:-2]+matrix[:,:,:,2:]-2*matrix[:,:,:,1:-1])
        gradients_yy = torch.abs(matrix[:,:,:-2,:]+matrix[:,:,2:,:]-2*matrix[:,:,1:-1,:])        
  
        c_d = self.a*torch.diff(matrix,dim=1)+self.b1*gradients_x+self.b2*gradients_y+self.c*gradients_xx+self.c*gradients_yy
           
            
        loss = self.alpha*F.mse(matrix,x)+self.beta*torch.abs(c_d)   

        return loss



##########################################################################
class LPTQPN(nn.Module):
    def __init__(self,
                 inp_channels=5,
                 out_channels=20,
                 dim=20,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=2,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 ):

        super(LPTQPN, self).__init__()

        self.patch_embed = PatchEmbed(inp_channels, dim)

        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.encoder_levels = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.decoder_levels = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for i in range(len(num_blocks)-1):
            self.encoder_levels.append(nn.Sequential(*[
                TransformerBlock(dim=int(dim * 2 ** i), num_heads=heads[i], ffn_expansion_factor=ffn_expansion_factor,
                                 bias=bias) for _ in range(num_blocks[i])]))

            self.downsamples.append(Downsample(int(dim * 2 ** i)))

            if i > 0:

                self.decoder_levels.append(nn.Sequential(*[
                    TransformerBlock(dim=int(dim * 2 ** i ), num_heads=heads[i],
                                    ffn_expansion_factor=ffn_expansion_factor, bias=bias) for _ in range(num_blocks[i])]))
            else:
                self.decoder_levels.append(nn.Sequential(*[
                    TransformerBlock(dim=int(dim * 2 ** (i+1) ), num_heads=heads[i],
                                    ffn_expansion_factor=ffn_expansion_factor, bias=bias) for _ in range(num_blocks[i])]))
                            
            self.upsamples.append(Upsample(int(dim * 2 ** (i+1))))

        self.central = self._create_transformer_blocks(int(dim * 2 ** 3), num_blocks[3], heads[3], ffn_expansion_factor, bias)

        self.refinement = self._create_transformer_blocks(int(dim * 2 ** 1), num_refinement_blocks, heads[0], ffn_expansion_factor, bias)

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        
        self.last = nn.Sequential(nn.Conv2d(in_channels=20,out_channels=20,kernel_size=3,padding=1),Swish())
        

        #Convection Diffusion Loss parameters
        self.params = nn.ModuleList()
        self.p = nn.ParameterList()
        for _ in range(4):
            self.params.append(nn.Sequential(nn.Conv2d(in_channels=20,out_channels=1,kernel_size=3, stride=1, padding=1),nn.ReLU()))
            self.p.append(nn.Parameter(torch.ones(1)))

    def forward(self, inp_img):
        
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_levels[0](inp_enc_level1)

        inp_enc_level2 = self.downsamples[0](out_enc_level1)
        out_enc_level2 = self.encoder_levels[1](inp_enc_level2)

        inp_enc_level3 = self.downsamples[1](out_enc_level2)
        out_enc_level3 = self.encoder_levels[2](inp_enc_level3)

        inp_enc_level4 = self.downsamples[2](out_enc_level3)
        central = self.central(inp_enc_level4)

        inp_dec_level3 = self.upsamples[-1](central)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_levels[-1](inp_dec_level3)

        inp_dec_level2 = self.upsamples[-2](out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_levels[-2](inp_dec_level2)

        inp_dec_level1 = self.upsamples[-3](out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_levels[-3](inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        

        out_dec_level1 = self.output(out_dec_level1)


        
        add_lastframe=out_dec_level1+inp_img[:,-1,].unsqueeze(1)
        
        out = self.last(add_lastframe)
        
        #This can be used for Convection Diffusion Loss
        params = []
        for i in range(4):
            params.append(torch.mean(self.params[i](out) * self.p[i]))

        return out, *params
                        
        #return out  #comment the params if you want to use MSE loss
        

    



    
    def _create_transformer_blocks(self, dim, num_blocks, num_heads, ffn_expansion_factor, bias):
        return nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in range(num_blocks)])



class LPTQPN_pl(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = LPTQPN()


    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30,eta_min=1e-9)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds,a,b1,b2,c = self.model(imgs)
        loss = ConvectionDiffusionLoss(1,0.5,a,b1,b2,c)(labels,preds)   
        # loss = F.mse_loss(preds, labels)            #MSE loss
        csi = CSI(preds,labels,threshold=74)

        self.log("%s_loss" % mode, loss)
        self.log("%s_csi" % mode, csi)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")