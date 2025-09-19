import torch, types, os, gc, math, json
import numpy as np
import torch.nn as nn
import numbers
from torch.nn import functional as F
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


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
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='BiasFree'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class MetaFuse(nn.Module):
    def __init__(self, dims, meta_dims):
        super(MetaFuse, self).__init__()
        self.meta_fc = nn.Linear(meta_dims, dims)
        self.conv_in = nn.Conv2d(dims, dims, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(dims, 2*dims, 3, padding=1, groups=dims),
            nn.GELU(),
            nn.Conv2d(2*dims, dims, 1)
        )
        
    def forward(self, x, metadata):
        b,c,h,w = x.shape
        shortcut = x
        x = self.conv_in(x)
        x = x * self.meta_fc(metadata).unsqueeze(-1).unsqueeze(-1)
        x = self.conv(x)
        x = x + shortcut
        return x

class ResBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return x + self.block(x)

#Pre Denoising Block
class PDB(nn.Module):
    def __init__(self, in_channels, dims):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, dims, 3, padding=1, padding_mode='reflect'),
            nn.GELU(),
            nn.Conv2d(dims, dims, 1)
        )

        self.down1 = Downsample(dims)
        self.block1 = ResBlock(block=LocalEnhanceBlock(dims=2*dims))
        self.down2 = Downsample(2*dims)

        self.block2 = ResBlock(block=LocalEnhanceBlock(dims=4*dims))

        self.up2 = Upsample(4*dims)
        self.block3 = ResBlock(block=LocalEnhanceBlock(dims=2*dims))
        self.up1 = Upsample(2*dims)

        self.conv_out = nn.Sequential(
            nn.Conv2d(dims, dims, 3, padding=1, padding_mode='reflect'),
            nn.GELU(),
            nn.Conv2d(dims, dims, 1)
        )

        self.conv_out_raw = nn.Sequential(
            nn.Conv2d(dims, in_channels, 3, padding=1),
            nn.Conv2d(in_channels, in_channels, 1)
        )
        
    def forward(self, x):
        #x (b,c_in,h,w)
        shortcut = x
        x = self.conv_in(x) #(b,c,h,w)
        t1 = x

        x = self.down1(x)
        x = self.block1(x)
        t2 = x

        x = self.down2(x)
        x = self.block2(x)
        l3 = x

        x = self.up2(x)
        x = x + t2
        x = self.block3(x)
        l2 = x

        x = self.up1(x)
        x = x + t1
        x = self.conv_out(x)
        l1 = x
        x = self.conv_out_raw(x) #(b, c_in, h, w)
        x = x + shortcut #(b, c_in, h, w)
        return x, l1, l2, l3

class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        num_scan_directions,
        d_state=16,
        d_conv=3,
        expand=2.0,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.num_scan_directions = num_scan_directions
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = tuple([
            nn.Linear(self.d_inner,(self.dt_rank + self.d_state * 2),bias=False,**factory_kwargs,)
            for _ in range(num_scan_directions)]
        )
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj], dim=0)
        )  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = tuple(
            [self.dt_init(self.dt_rank,self.d_inner,dt_scale,dt_init,dt_min,dt_max,dt_init_floor,**factory_kwargs)
            for _ in range(num_scan_directions)]
        )

        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs], dim=0)
        )  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in self.dt_projs], dim=0)
        )  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(
            self.d_state, self.d_inner, copies=num_scan_directions, merge=True
        )  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=num_scan_directions, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None


    @staticmethod
    def dt_init(dt_rank,d_inner,dt_scale=1.0,dt_init="random",dt_min=0.001,dt_max=0.1,dt_init_floor=1e-4,**factory_kwargs,):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, spiral_indices=None):
        B, C, H, W = x.shape
        L = H * W
        K = self.num_scan_directions

        row_indices = torch.arange(x.size(2)).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(-1) % 2
        col_indices = torch.arange(x.size(3)).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(-1) % 2

        xs = torch.stack([
            torch.where(row_indices == 0, x, torch.flip(x, dims=[3])).view(B, -1, L),
            torch.where(col_indices == 0,torch.transpose(x, dim0=2, dim1=3).contiguous(),torch.flip(torch.transpose(x, dim0=2, dim1=3).contiguous(), dims=[3])).view(B, -1, L),
        ], dim=1).view(B, K, -1, L)

        if self.num_scan_directions == 4:
            xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1) # (B, 4, C, L)

        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight
        )
        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2
        )
        dts = torch.einsum(
            "b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight
        )
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(xs,dts,As,Bs,Cs,Ds,z=None,delta_bias=dt_projs_bias,delta_softplus=True,return_last_state=False,
                                    ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        if self.num_scan_directions == 2:
            y1 = out_y[:, 0].view(B,-1,H,W)
            y2 = out_y[:, 1].view(B,-1,W,H)
            y1 = torch.where(row_indices == 0, y1, torch.flip(y1, dims=[3])).view(B, -1, L)
            y2 = torch.transpose(torch.where(col_indices == 0, y2, torch.flip(y2, dims=[3])),dim0=2,dim1=3).contiguous().view(B,-1,L)
            return y1, y2
        elif self.num_scan_directions == 4:
            y1 = out_y[:, 0].view(B,-1,H,W)
            y2 = out_y[:, 1].view(B,-1,W,H)
            y1 = torch.where(row_indices == 0, y1, torch.flip(y1, dims=[3])).view(B, -1, L)
            y2 = torch.transpose(torch.where(col_indices == 0, y2, torch.flip(y2, dims=[3])),dim0=2,dim1=3).contiguous().view(B,-1,L)

            y1_inv = torch.flip(out_y[:, 1], dims=[-1]).view(B,-1,H,W)
            y1_inv = torch.where(row_indices == 0, y1_inv, torch.flip(y1_inv, dims=[3])).view(B, -1, L)
            y2_inv = torch.flip(out_y[:, 3], dims=[-1]).view(B,-1,W,H)
            y2_inv = torch.transpose(torch.where(col_indices == 0, y2_inv, torch.flip(y2_inv, dims=[3])),dim0=2,dim1=3).contiguous().view(B,-1,L)
            return y1, y2, y1_inv, y2_inv
            
    def forward(self, x: torch.Tensor, spiral_indices=None, **kwargs):
        """
        x: tensor with shape (b,c,h,w)
        output: tensor with shape (b,c,h,w)
        """
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (b,h,w,c)
        xz = self.in_proj(x)  # (b,h,w,2*e*c)
        x, z = xz.chunk(2, dim=-1)  # both (b,h,w,e*c)
        x = x.permute(0, 3, 1, 2).contiguous()  # (b,e*c,h,w)
        x = self.act(self.conv2d(x))  # main branch that will be selective scaned by SS2D

        #y1, y2, y3, y4 = self.forward_core(x)#four directions scans
        if self.num_scan_directions == 2:
            y1, y2 = self.forward_core(x)
            assert y1.dtype == torch.float32
            y = y1 + y2
        elif self.num_scan_directions == 4:
            y1, y2, y3, y4 = self.forward_core(x)
            assert y1.dtype == torch.float32
            y = y1 + y2 + y3 + y4 
        
        y = (torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1))  # (b,h,w,e*c)
        y = self.out_norm(y)  # (b,h,w,e*c)
        y = y * F.silu(z)  # mutiplied by the other info flow after SiLU
        out = self.out_proj(y)  # (b,h,w,e*c) -> #(b,h,w,c)
        out = out.permute(0, 3, 1, 2)  # (b,c,h,w)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class MetaAttentionBlock(nn.Module):
    def __init__(self, dims, num_heads, meta_dims):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q = MetaFuse(dims, meta_dims)
        self.k = nn.Linear(dims, dims, bias=False)
        self.v = nn.Linear(dims, dims, bias=False)
        self.conv_out = nn.Conv2d(dims, dims, kernel_size=1, bias=False)

    def forward(self, x, metadata):
        #x (b,c,h,w)
        #metadata (b,n,d)
        b,c,h,w = x.shape
        q = self.q(x, metadata)
        x = rearrange(x, 'b c h w -> b (h w) c')

        q = rearrange(q, 'b c h w -> b (h w) c')
        k = self.k(x)
        v = self.v(x) # (b, hw, c)
        
        q = rearrange(q, 'b l (head c) -> b head c l', head=self.num_heads)#(b, mum_heads, c//mum_heads, hw)
        k = rearrange(k, 'b l (head c) -> b head c l', head=self.num_heads)#(b, mum_heads, c//mum_heads, hw)
        v = rearrange(v, 'b l (head c) -> b head c l', head=self.num_heads)#(b, mum_heads, c//mum_heads, hw)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature # (b, mum_heads, c//mum_heads, c//mum_heads)
        attn = attn.softmax(dim=-1)

        x = (attn @ v)# (b, mum_heads, c//mum_heads, hw)
        x = rearrange(x, 'b head c (h w) -> b (head c) h w', h=h, w=w)
        x = self.conv_out(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True) 
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

#large scale block
class LSBlock(nn.Module):
    def __init__(self, dims, num_heads, meta_dims):
        super().__init__()
        self.ln1 = LayerNorm(dim=dims)
        self.ln2 = LayerNorm(dim=dims)

        self.att = MetaAttentionBlock(dims=dims, num_heads=num_heads, meta_dims=meta_dims)
        self.ffn = LocalEnhanceBlock(dims=dims)

        self.gamma1 = nn.Parameter(torch.ones((1,dims,1,1)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones((1,dims,1,1)), requires_grad=True)

    def forward(self, x, metadata):
        if metadata is not None:
            x = x + self.gamma1 * self.att(self.ln1(x), metadata)
        else:
            x = x + self.gamma1 * self.att(self.ln1(x))
        x = x + self.gamma2 * self.ffn(self.ln2(x))
        return x

#small scale block
class SSBlock(nn.Module):
    def __init__(self, dims, num_scan_directions, expand):
        super().__init__()
        self.ln1 = LayerNorm(dim=dims)
        self.ln2 = LayerNorm(dim=dims)

        # self.att = Mamba(d_model=dims, num_scan_directions=num_scan_directions, expand=expand)
        self.att =SpatialAttention()
        self.ffn = LocalEnhanceBlock(dims=dims)

        self.gamma1 = nn.Parameter(torch.ones((1,dims,1,1)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones((1,dims,1,1)), requires_grad=True)

    def forward(self, x): 
        x = x + self.gamma1 * self.att(self.ln1(x))
        x = x + self.gamma2 * self.ffn(self.ln2(x))
        return x

class Downsample(nn.Module):
    def __init__(self, dims):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(dims, dims//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, dims):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(dims, dims*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)

class SimpleFuse(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.body = nn.Conv2d(2*dims, dims, 1)
    def forward(self, x, y):
        x = self.body(torch.cat([x,y],dim=1))
        return x

class MultiPriorFuse(nn.Module):
    def __init__(self, dims, in_channels):
        super().__init__()
        self.refine_pred_raw = nn.Conv2d(in_channels, dims, 1)
        self.refine_aligned_raw = nn.Conv2d(in_channels, dims, 1)
        self.hf_refine1 = nn.Conv2d(in_channels, dims, 3, padding=1)
        self.mix_hf = nn.Sequential(
            nn.Conv2d(3*dims, dims, 3, padding=1, groups=dims),
            nn.GELU(),
            nn.Conv2d(dims, dims, 1)
        )
        self.refine1 = nn.Conv2d(dims, dims, 1)
        self.refine2 = nn.Conv2d(dims, dims, 1)
    def forward(self, x, y, pred_raw, hf_pred_raw, aligned_raw):
        '''
        x,y (b,c,h,w)
        prior, pred_raw_high_fre (b, c_in, h, w)
        '''
        shortcut = y
        aligned_raw = self.refine_aligned_raw(aligned_raw)
        pred_raw = self.refine_pred_raw(pred_raw)
        hf_pred_raw = self.hf_refine1(hf_pred_raw)

        y = y + aligned_raw
        y_high_fre, _, _ = extract_fre(y)

        x_high_fre, x_high_fre_imag, x_low_fre = extract_fre(x)
        x_high_fre = self.mix_hf(torch.cat([x_high_fre, hf_pred_raw, y_high_fre], dim=1))
        x = inv_extract_fre(x_low_fre, x_high_fre, x_high_fre_imag)

        x = self.refine1(x) + pred_raw
        x = self.refine2(x)
        x = x + shortcut
        return x

class RemainChannelDown(nn.Module):
    def __init__(self, dims):
        super(RemainChannelDown, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dims, dims, kernel_size=2, stride=2, bias=False)
        )
    def forward(self, x):
        return self.body(x)

class LocalNorm(nn.Module):
    def __init__(self, patch_size, stride):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride

    def forward(self, x):
        B, C, H, W = x.shape
        unfold  = torch.nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=self.stride) #no overlap
        x = torch.transpose(unfold(x), -2, -1) #(b, num_patches, c*kernel_size*kernel_size)
        x = x.reshape(B, -1, self.patch_size, self.patch_size) #(b, num_patches, c, kernel_size, kernel_size)

        pool = torch.nn.AvgPool2d(kernel_size=self.patch_size,stride=self.stride) 
        mean = pool(x) #(b, num_patches*c, 1, 1)
        std = x.std(dim=[-2, -1]).unsqueeze(-1).unsqueeze(-1)
        normed_x = (x - mean) / (std + 1e-7)
        normed_x = normed_x.view(B, -1, C*self.patch_size*self.patch_size) #(b, num_patches, c*kernel_size*kernel_size)
        normed_x = torch.transpose(normed_x, -2, -1) #(b, c*kernel_size*kernel_size, num_patches)

        fold = torch.nn.Fold(output_size=(H, W), kernel_size=(self.patch_size, self.patch_size), stride=self.stride)
        normed_x = fold(normed_x)
        return normed_x

class LocalEnhanceBlock(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        self.conv1 = nn.Conv2d(dims, 2*dims, kernel_size=3, padding=1, stride=1, groups=dims, bias=True, dilation = 1)
        self.conv2 = nn.Conv2d(dims, 2*dims, kernel_size=3, padding=4, stride=1, groups=dims, bias=True, dilation = 4)
        self.conv3 = nn.Conv2d(dims, 2*dims, kernel_size=3, padding=9, stride=1, groups=dims, bias=True, dilation = 9)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dims, dims, 1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(dims, dims, 3, padding=1, groups=dims),
            nn.Conv2d(dims, dims, 1)
        )

    def forward(self, x):
        x1, t1 = self.conv1(x).chunk(2, dim=1)
        x2, t2 = self.conv2(x).chunk(2, dim=1)
        x3, t3 = self.conv3(x).chunk(2, dim=1)
        x = x1 * t1 + x2 * t2 + x3 * t3
        x = self.sca(x) * x #(b,c,h,w)
        x = self.conv4(x)
        return x

def extract_fre(x):
    h, w = x.shape[-2:]
    fft_x = torch.fft.fft2(x, dim=(-2, -1))
    fft_x_shifted = torch.fft.fftshift(fft_x, dim=(-2, -1))

    threshold = 0.01 
    center_h = int(h * threshold)
    center_w = int(w * threshold)

    mask_low = torch.zeros_like(fft_x_shifted)
    mask_low[..., h // 2 - center_h:h // 2 + center_h, w // 2 - center_w:w // 2 + center_w] = 1

    mask_high = 1 - mask_low

    fft_x_low = fft_x_shifted * mask_low
    fft_x_high = fft_x_shifted * mask_high

    fft_x_high_real = torch.real(fft_x_high)
    fft_x_high_imag = torch.imag(fft_x_high)

    return fft_x_high_real, fft_x_high_imag, fft_x_low

def inv_extract_fre(fft_x_low, fft_x_high_real, fft_x_high_imag):
    fft_x_high = torch.complex(fft_x_high_real, fft_x_high_imag)
    fft_x = fft_x_low + fft_x_high
    fft_x_shifted_back = torch.fft.ifftshift(fft_x, dim=(-2, -1))
    x = torch.fft.ifft2(fft_x_shifted_back, dim=(-2, -1)).real
    return x

def get_local_mean_and_std(x, patch_size, stride):
    """
    Args:
        x: (B, C, H, W) tensor
        patch_size: local region size
        stride: sliding step
    Returns:
        mean, std: both of shape (B, C, H, W)
    """
    B, C, H, W = x.shape
    unfold = F.unfold(x, kernel_size=(patch_size, patch_size), stride=stride)  # (B, C*ps*ps, L)
    ps_area = patch_size * patch_size
    num_patches = unfold.shape[-1]
    
    patches = unfold.view(B, C, ps_area, num_patches)  # (B, C, ps*ps, L)

    # Mean and std per patch
    mean = patches.mean(dim=2, keepdim=True)  # (B, C, 1, L)
    std = patches.std(dim=2, keepdim=True)    # (B, C, 1, L)

    # Expand back to patch layout
    mean = mean.expand(-1, -1, ps_area, -1).contiguous().view(B, C * ps_area, num_patches)
    std = std.expand(-1, -1, ps_area, -1).contiguous().view(B, C * ps_area, num_patches)

    # Fold to original shape
    mean = F.fold(mean, output_size=(H, W), kernel_size=patch_size, stride=stride)
    std = F.fold(std, output_size=(H, W), kernel_size=patch_size, stride=stride)

    return mean, std

class LocalDistributionAlign(nn.Module):
    def __init__(self, in_channels, dims, patch_size, stride):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride

        self.delta_means = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, dims, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(dims, in_channels, 1),
            )
            for _ in range(4)
        ])
        self.delta_log_stds = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, dims, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(dims, in_channels, 1),
            )
            for _ in range(4)
        ])
        self.refine = nn.Sequential(
            nn.Conv2d(4*in_channels, 4*in_channels, 3, padding=1, groups=4*in_channels),
            nn.GELU(),
            nn.Conv2d(4*in_channels, in_channels, 1)
        )

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        origin_means = []
        origin_stds = []
        for ps in [2, 4, 8, 16]:
            m, s = get_local_mean_and_std(x, ps, ps)
            origin_means.append(m)
            origin_stds.append(s + 1e-7)
        aligned_xs = []
        for m, s, refine_m, refine_s in zip(origin_means, origin_stds, self.delta_means, self.delta_log_stds):
            delta_mean = refine_m(m)
            delta_log_std = refine_s(s)
            refined_mean = m + delta_mean
            refined_std = s * torch.exp(delta_log_std)
            aligned_x = (x - m) / s * refined_std + refined_mean
            aligned_xs.append(aligned_x)
        
        x = self.refine(torch.cat(aligned_xs, dim=1))
        return x

from utils.registry import MODEL_REGISTRY
@MODEL_REGISTRY.register()
class HiMA_Spatial(nn.Module):
    def __init__(self, in_channels=3, dims=32, layers=4, num_blocks_per_layer=[1,1,1,1]):
        super().__init__()
        self.layers = layers

        self.sqrt_inchannels = int(math.sqrt(in_channels))

        self.pdb = PDB(in_channels=in_channels, dims=dims)
        
        self.meta_dims = 2*dims
        self.learnable_metadata = nn.Parameter(torch.ones(self.meta_dims))

        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, dims, 3, padding=1, padding_mode='reflect'),
            nn.GELU(),
            nn.Conv2d(dims, dims, 1)
        )

        self.pred_raw_downs = nn.ModuleList([
            RemainChannelDown(dims=in_channels) for _ in range(layers-1)
        ])
        self.hf_pred_raw_downs = nn.ModuleList([
            RemainChannelDown(dims=in_channels) for _ in range(layers-1)
        ])
        self.aligned_x_downs = nn.ModuleList([
            RemainChannelDown(dims=in_channels) for _ in range(layers-1)
        ])

        #==================================encoders==================================
        self.encoder1 = nn.ModuleList([
            LSBlock(dims=dims*(2**0), num_heads=4, meta_dims=self.meta_dims) for i in range(num_blocks_per_layer[0])
        ])
        self.encoder2 = nn.ModuleList([
            LSBlock(dims=dims*(2**1), num_heads=8, meta_dims=self.meta_dims) for i in range(num_blocks_per_layer[1])
        ])
        self.encoder3 = nn.ModuleList([
            SSBlock(dims=dims*(2**2), num_scan_directions=4, expand=1.33) for i in range(num_blocks_per_layer[2])
        ])
        #==================================encoders==================================

        #==================================bottleneck==================================
        self.bottleneck = nn.ModuleList([
            SSBlock(dims=dims*(2**3), num_scan_directions=4, expand=1.11) for i in range(num_blocks_per_layer[3]) 
        ])
        #==================================bottleneck==================================

        #==================================decoders==================================
        self.decoder3 = nn.ModuleList([
            SSBlock(dims=dims*(2**2), num_scan_directions=4, expand=1.33) for i in range(num_blocks_per_layer[2])
        ])
        self.decoder2 = nn.ModuleList([
            LSBlock(dims=dims*(2**1), num_heads=8, meta_dims=self.meta_dims) for i in range(num_blocks_per_layer[1])
        ])
        self.decoder1 = nn.ModuleList([
            LSBlock(dims=dims*(2**0), num_heads=4, meta_dims=self.meta_dims) for i in range(num_blocks_per_layer[0])
        ])
        #==================================decoders==================================

        self.downs = nn.ModuleList([
            Downsample(dims=dims*2**(i)) for i in range(layers-1)
        ])

        self.ups = nn.ModuleList([
            Upsample(dims=dims*2**(i)) for i in range(layers-1, 0, -1)
        ])

        self.fuses = nn.ModuleList([
            MultiPriorFuse(dims=dims*2**(i), in_channels=in_channels) for i in range(layers-2, -1, -1)
        ])

        self.refine = nn.Sequential(
            nn.Conv2d(dims, dims, 3, padding=1),
            nn.Conv2d(dims, 3*in_channels, 1),
            nn.PixelShuffle(self.sqrt_inchannels)
        )
        
        self.input_local_distribution_align = LocalDistributionAlign(in_channels=in_channels, dims=dims, patch_size=16, stride=16)

    def forward(self, x):
        x = self._check_and_padding(x)
        aligned_raw = self.input_local_distribution_align(x)
        pred_raw, l1, l2, l3 = self.pdb(aligned_raw) #(b,c_in,h,w) (b, 4c, h//4, w//4) (b, c, h, w)
        raw_out = pred_raw
        hf_pred_raw, _, _ = extract_fre(pred_raw)

        pred_raws = []
        hf_pred_raws = []
        aligned_raws = []
        x = self.conv_in(x)

        encoder_features = []

        for encoder in self.encoder1:
            x = encoder(x, self.learnable_metadata)
        encoder_features.append(x)
        x = self.downs[0](x)
        pred_raws.append(pred_raw)
        pred_raw = self.pred_raw_downs[0](pred_raw)
        hf_pred_raws.append(hf_pred_raw)
        hf_pred_raw = self.hf_pred_raw_downs[0](hf_pred_raw)
        aligned_raws.append(aligned_raw)
        aligned_raw = self.aligned_x_downs[0](aligned_raw)

        for encoder in self.encoder2:
            x = encoder(x, self.learnable_metadata)
        encoder_features.append(x)
        x = self.downs[1](x)
        pred_raws.append(pred_raw)
        pred_raw = self.pred_raw_downs[1](pred_raw)
        hf_pred_raws.append(hf_pred_raw)
        hf_pred_raw = self.hf_pred_raw_downs[1](hf_pred_raw)
        aligned_raws.append(aligned_raw)
        aligned_x = self.aligned_x_downs[1](aligned_raw)

        for encoder in self.encoder3:
            x = encoder(x)
        encoder_features.append(x)
        x = self.downs[2](x)
        pred_raws.append(pred_raw)
        pred_raw = self.pred_raw_downs[2](pred_raw)
        hf_pred_raws.append(hf_pred_raw)
        hf_pred_raw = self.hf_pred_raw_downs[2](hf_pred_raw)
        aligned_raws.append(aligned_x)
        aligned_x = self.aligned_x_downs[2](aligned_raw)

        for block in self.bottleneck:
            x = block(x)

        encoder_features.reverse() #(4,2,1)
        pred_raws.reverse()
        hf_pred_raws.reverse()
        aligned_raws.reverse()

        x = self.ups[0](x)
        x = self.fuses[0](x, encoder_features[0], pred_raws[0], hf_pred_raws[0], aligned_raws[0])
        for decoder in self.decoder3:
            x = decoder(x)

        x = self.ups[1](x)
        x = self.fuses[1](x, encoder_features[1], pred_raws[1], hf_pred_raws[1], aligned_raws[1])
        for decoder in self.decoder2:
            x = decoder(x, self.learnable_metadata)

        x = self.ups[2](x)
        x = self.fuses[2](x, encoder_features[2], pred_raws[2], hf_pred_raws[2], aligned_raws[2])
        for decoder in self.decoder1:
            x = decoder(x, self.learnable_metadata)

        x = self.refine(x)
        x, raw_out = self._check_and_crop(x, raw_out)
        return x, raw_out

    def _check_and_padding(self, x):
        _, _, h, w = x.size()
        stride = 2**3
        dh = -h % stride
        dw = -w % stride
        top_pad = dh // 2
        bottom_pad = dh - top_pad
        left_pad = dw // 2
        right_pad = dw - left_pad
        self.crop_indices = (left_pad, w + left_pad, top_pad, h + top_pad)

        # Pad the tensor with reflect mode
        padded_tensor = F.pad(
            x, (left_pad, right_pad, top_pad, bottom_pad), mode="reflect"
        )

        return padded_tensor

    def _check_and_crop(self, x, raw=None):
        left, right, top, bottom = self.crop_indices
        x = x[:, :, top * self.sqrt_inchannels : bottom * self.sqrt_inchannels, left * self.sqrt_inchannels : right * self.sqrt_inchannels]
        raw = raw[:, :, top:bottom, left:right] if raw is not None else None
        return x, raw

def cal_model_complexity(model, x):
    import thop
    flops, params = thop.profile(model, inputs=(x,), verbose=False)
    print(f"FLOPs: {flops / 1e9} G")
    print(f"Params: {params / 1e6} M")

if __name__ == '__main__':
    from tqdm import tqdm
    model = HiMA_Spatial(in_channels=4, dims=32, layers=4, num_blocks_per_layer=[2,2,2,2]).to(device='cuda') 
    x = torch.randn((1,4,512,512)).to(device='cuda')
    with torch.no_grad():
        cal_model_complexity(model, x)
        rgb, raw = model(x)
        print(rgb.shape, raw.shape)
    exit(0)

    iterations = 10
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    times = torch.zeros(iterations)

    with torch.no_grad():
        for _ in range(10):
            _ = model(x)

    with torch.no_grad():
        for iter in tqdm(range(iterations), desc="Measuring Inference Time", unit="iteration"):
            starter.record()
            _ = model(x)
            ender.record()

            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            times[iter] = curr_time

    mean_time = times.mean().item()
    fps = 1000 / mean_time

    print(f"Inference time: {mean_time:.6f} ms, FPS: {fps:.2f}")