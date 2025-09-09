import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

class RGB_Head(nn.Module):
    def __init__(self, width=32, adaptive_size=2, block_size=2):
        super().__init__()

        self.get_q=nn.Sequential(
            LayerNorm(width),
            nn.Conv2d(width,width,3,1,1,groups=width,padding_mode='reflect'),
            nn.AdaptiveAvgPool2d(adaptive_size),
            nn.Conv2d(width,width,1)
        )

        self.compute_attn=compute_attn(width)

        self.final_conv1 = nn.Conv2d(width, width, 3, 1, 1, padding_mode='reflect')
        self.gelu=nn.GELU()
        self.final_conv2 = nn.Conv2d(width, 3*block_size**2, 1, 1, 0, padding_mode='reflect')
        self.final_up_sampling = nn.PixelShuffle(block_size)

    def forward(self, x, k ,v):
        q = self.get_q(x)
        x = self.compute_attn(x, q, k, v)
        x = self.final_conv1(x)
        x = self.gelu(x)
        x = self.final_conv2(x)
        x = self.final_up_sampling(x)
        return x


class RAW_Head(nn.Module):
    def __init__(self, width=32, adaptive_size=2, block_size=2):
        super().__init__() 

        self.get_q=nn.Sequential(
            LayerNorm(width),
            nn.Conv2d(width,width,3,1,1,groups=width,padding_mode='reflect'),
            nn.AdaptiveAvgPool2d(adaptive_size),
            nn.Conv2d(width,width,1)
        )

        self.compute_attn=compute_attn(width)

        self.final = nn.Conv2d(width, block_size**2, 3, 1, 1, padding_mode='reflect')
            
    def forward(self, x, k, v):
        q = self.get_q(x)
        x = self.compute_attn(x,q,k,v)
        x = self.final(x)
        return x
    

class compute_attn(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width=width
        self.out=nn.Conv2d(width,width,kernel_size=1)
    def forward(self,x,q,k,v):
        inp = x
        q=rearrange(q,'B C H W -> B C (H W)')
        k=rearrange(k,'B C H W -> B C (H W)')
        attn=(q@k.transpose(-2,-1)).softmax(dim=-1)
        B,C,H,W=x.shape
        v=rearrange(v,'B C H W -> B C (H W)')        
        x=rearrange(attn@v, 'B C (H W) -> B C H W',H=H,W=W)
        x=self.out(x) + inp
        return x

class SpatialGatingUnit(nn.Module):
    def __init__(self, f_number, excitation_factor=2) -> None:
        super().__init__()
        self.pconv1 = nn.Conv2d(f_number, int(excitation_factor * f_number), kernel_size=1)
        self.pconv2 = nn.Conv2d(int(excitation_factor * f_number)//2, f_number, kernel_size=1)
    def forward(self, x):
        x = self.pconv1(x)
        y1,y2=x.chunk(2,dim=1)
        x=y1*y2
        x = self.pconv2(x)# + inp
        return x
    
class RefinedDetailExtractor(nn.Module):
    def __init__(self, f_number, padding_mode='reflect') -> None:
        super().__init__()
        self.dconv = nn.Conv2d(f_number, f_number, kernel_size=7, padding=3, groups=f_number, padding_mode=padding_mode)
        self.norm = LayerNorm(f_number)
        self.SGU = SpatialGatingUnit(f_number, excitation_factor=2)

    def forward(self, x):
        inp=x
        x = self.dconv(x)
        x = self.norm(x)
        return self.SGU(x) + inp

class GlobalChromaticPerceptor(nn.Module):
    def __init__(self, inp_channel, adaptive_size=2, bias=False):
        super().__init__()
        self.inp_channel=inp_channel
        self.norm=LayerNorm(inp_channel)
        self.pconv=nn.Conv2d(inp_channel,inp_channel*3,kernel_size=1, bias=bias)
        self.dconv=nn.Conv2d(inp_channel*3,inp_channel*3,kernel_size=3,padding=1,groups=inp_channel*3, bias=bias)
        self.out=nn.Conv2d(inp_channel,inp_channel,kernel_size=1)

        self.pool1=nn.AdaptiveAvgPool2d(adaptive_size)
        self.pool2=nn.AdaptiveAvgPool2d(adaptive_size)
        
    def forward(self,x):
        inp=x
        x = self.norm(x)
        c1,c2,c3=self.dconv(self.pconv(x)).chunk(3,dim=1)
        c1=rearrange(self.pool1(c1),'B C H W -> B C (H W)')
        c2=rearrange(self.pool2(c2),'B C H W -> B C (H W)')
        attn=(c1@c2.transpose(-2,-1)).softmax(dim=-1)
        B,C,H,W=x.shape
        c3=rearrange(c3,'B C H W -> B C (H W)')
        x=rearrange(attn@c3, 'B C (H W) -> B C H W',H=H,W=W)
        x=self.out(x)
        x=x+inp
        return x
    

class CNPModule(nn.Module):
    """
    Chromaticity and Noise Perception Module
    """
    def __init__(self, inp_channel, adaptive_size=2, global_aware=True) -> None:
        super().__init__()
        self.global_aware = global_aware
        if self.global_aware:
            self.GCP=GlobalChromaticPerceptor(inp_channel,adaptive_size)
        self.RDE=RefinedDetailExtractor(inp_channel,'reflect')
    
    def forward(self,x):
        if self.global_aware:
            x=self.GCP(x)
        x=self.RDE(x)
        return x

class Backbone(nn.Module):
    def __init__(self, in_channel=4, width=32, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[],adaptive_size=2):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=in_channel,
                               out_channels=width, kernel_size=5, padding=2, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=width,
                                kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[CNPModule(chan, global_aware=False) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[CNPModule(chan, global_aware=False) for _ in range(middle_blk_num)]
            )

        for i,num in enumerate(dec_blk_nums):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[CNPModule(chan,adaptive_size) for _ in range(num)]
                )
            )

    def forward(self, inp):
        x = inp
        B, C, H, W = x.shape
        x = self.intro(x)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x+enc_skip
            x = decoder(x)

        x = self.ending(x)

        return x

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g + self.b

from utils.registry import MODEL_REGISTRY
@MODEL_REGISTRY.register()
class CANS_Plus(nn.Module):
    def __init__(self, in_channel=4, width=32, middle_blk_num=2, enc_blk_nums=[2,2,2,2], dec_blk_nums=[2,2,2,2], adaptive_size=2, block_size=2):
        super(CANS_Plus, self).__init__()
        self.backbone = Backbone(in_channel, width, middle_blk_num, enc_blk_nums, dec_blk_nums, adaptive_size)
        self.rgb_head = RGB_Head(width,adaptive_size,block_size)
        self.raw_head = RAW_Head(width,adaptive_size,block_size)

        self.get_kv=nn.Sequential(
            LayerNorm(width),
            nn.Conv2d(width, width*2, kernel_size=1),
            nn.Conv2d(width*2, width*2, 3, 1, 1, groups=width*2, padding_mode='reflect')
        )
        self.adaptive=nn.AdaptiveAvgPool2d(adaptive_size)

        self.block_size = block_size

    def forward(self, x):
        x = self._check_and_padding(x)
        fea = self.backbone(x)
        k, v = self.get_kv(fea).chunk(2,dim=1)
        k = self.adaptive(k)
        raw = self.raw_head(fea, k, v)
        rgb = self.rgb_head(fea, k, v)
        rgb, raw = self._check_and_crop(rgb, raw)
        return rgb, raw

    def _check_and_padding(self, x):
        # Calculate the required size based on the input size and required factor
        _, _, h, w = x.size()
        #stride = (2 ** (4 - 1))
        stride = (2 ** 4)

        # Calculate the number of pixels needed to reach the required size
        dh = -h % stride
        dw = -w % stride

        # Calculate the amount of padding needed for each side
        top_pad = dh // 2
        bottom_pad = dh - top_pad
        left_pad = dw // 2
        right_pad = dw - left_pad
        self.crop_indices = (left_pad, w+left_pad, top_pad, h+top_pad)

        # Pad the tensor with reflect mode
        padded_tensor = F.pad(
            x, (left_pad, right_pad, top_pad, bottom_pad), mode="reflect"
        )

        return padded_tensor
        
    def _check_and_crop(self, x, res1):
        left, right, top, bottom = self.crop_indices
        x = x[:, :, top*self.block_size:bottom*self.block_size, left*self.block_size:right*self.block_size]
        res1 = res1[:, :, top:bottom, left:right] if res1 is not None else None
        return x, res1