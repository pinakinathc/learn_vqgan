"""
Self-contained Minimal implementation of VQ-GAN model.
Paper: https://arxiv.org/abs/2012.09841
Reference: https://compvis.github.io/taming-transformers
Copyright: Do whatever you want. Don't ask me.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, channel=128, in_channel=3, channel_mult=(1,1,2,2,4),
        num_res_blocks=2, attn_resolution=[16], dropout=0.0, 
        resamp_with_conv=True, resolution=256, z_channel=256
    ):
        super().__init__()
        self.temb_channel = 0
        self.num_resolution = len(channel_mult)
        self.num_res_blocks = num_res_blocks

        # downsampling
        self.conv_in = nn.Conv2d(
            in_channel, channel, kernel_size=3, stride=1, padding=1
        )
        current_resolution = resolution
        in_channel_mult = (1,)+tuple(channel_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolution):
            block     = nn.ModuleList()
            attn      = nn.ModuleList()
            block_in  = channel * in_channel_mult[i_level]
            block_out = channel * channel_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(
                    in_channel=block_in, out_channel=block_out,
                    temb_channel=self.temb_channel, dropout=dropout
                ))
                block_in = block_out
                if current_resolution in attn_resolution:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn  = attn
            if i_level != self.num_resolution-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                current_resolution = current_resolution // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channel=block_in, out_channel=block_in,
            temb_channel=self.temb_channel, dropout=dropout
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channel=block_in, out_channel=block_in,
            temb_channel=self.temb_channel, dropout=dropout
        )

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(
            block_in, z_channel,
            kernel_size=3, stride=1, padding=1
        )
    
    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolution):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolution-1:
                h = self.down[i_level].downsample(h)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = h + torch.sigmoid(h) # swish non-linearity
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, channel=128, out_channel=3, channel_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2, attn_resolution=[16], dropout=0.0,
        resamp_with_conv=True, resolution=256, z_channel=256
    ):
        super().__init__()
        self.temb_channel = 0
        self.num_resolution = len(channel_mult)
        self.num_res_blocks = num_res_blocks

        current_resolution = resolution // 2**(self.num_resolution-1) # Encoder uses (current_resolution = current_resolution // 2)
        in_channel_mult = (1,)+tuple(channel_mult)
        block_in = channel * channel_mult[-1]

        # z to block_in
        self.conv_in = nn.Conv2d(
            z_channel, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channel=block_in, out_channel=block_in,
            temb_channel=self.temb_channel, dropout=dropout
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channel=block_in, out_channel=block_in,
            temb_channel=self.temb_channel, dropout=dropout
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolution)):
            block     = nn.ModuleList()
            attn      = nn.ModuleList()
            block_out = channel * channel_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(
                    in_channel=block_in, out_channel=block_out,
                    temb_channel=self.temb_channel, dropout=dropout
                ))
                block_in = block_out
                if current_resolution in attn_resolution:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                current_resolution = current_resolution * 2
            self.up.append(up)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(
            block_in, out_channel, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z):
        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in range(self.num_resolution):
            for i_block in range(self.num_res_blocks):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != self.num_resolution-1:
                h = self.up[i_level].upsample(h)
        
        # end
        h = self.norm_out(h)
        h = h + torch.sigmoid(h) # swish non-linearity
        h = self.conv_out(h)
        return h


class ResnetBlock(nn.Module):
    def __init__(self, in_channel, out_channel=None, conv_shortcut=False, dropout=0.0, temb_channel=512):
        super().__init__()
        self.in_channel = in_channel
        out_channel = in_channel if out_channel is None else out_channel
        self.out_channel = out_channel
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channel, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(
            in_channel, out_channel,
            kernel_size=3, stride=1, padding=1
        )
        if temb_channel > 0:
            self.temb_proj = nn.Linear(temb_channel, out_channel)
        
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channel, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            out_channel, out_channel,
            kernel_size=3, stride=1, padding=1
        )
        if self.in_channel != self.out_channel:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channel, out_channel,
                    kernel_size=3, stride=1, padding=1
                )
            else:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channel, out_channel,
                    kernel_size=1, stride=1, padding=0
                )
    
    def forward(self, x, temb):
        h = x
        h = self.norm1(x)
        h = x * torch.sigmoid(x) # swish non-linearity
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(temb + torch.sigmoid(temb))[:, :, None, None]
        
        h = self.norm2(h)
        h = h + torch.sigmoid(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channel != self.out_channel:
            x = self.conv_shortcut(x)
        
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channel, eps=1e-6, affine=True)
        self.q = nn.Conv2d(
            in_channel, in_channel, kernel_size=1, stride=1, padding=0
        )
        self.k = nn.Conv2d(
            in_channel, in_channel, kernel_size=1, stride=1, padding=0
        )
        self.v = nn.Conv2d(
            in_channel, in_channel, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = nn.Conv2d(
            in_channel, in_channel, kernel_size=1, stride=1, padding=0
        )
    
    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        k = k.reshape(b, c, h*w)
        w_ = torch.bmm(q.permute(0, 2, 1), k) # (b, hw, c) x (b, c, hw) --> (b, hw, hw)
        w_ = w_ / (int(c)**0.5) # (q.T x k) / √c
        w_ = F.softmax(w_, 2)

        # attend to values
        v = v.reshape(b, c, h*w)
        h_ = torch.bmm(v, w_.permute(0, 2, 1))
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)
        return x + h_


class Downsample(nn.Module):
    def __init__(self, in_channel, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(
                in_channel, in_channel, kernel_size=3, stride=2, padding=0
            )
    
    def forward(self, x):
        if self.with_conv:
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0) # (left, right, top, bottom)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        
        return x


class Upsample(nn.Module):
    def __init__(self, in_channel, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(
                in_channel, in_channel, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class VectorQuantizer(nn.Module):
    def __init__(self, num_emb=1024, emb_dim=256, beta=0.25):
        super().__init__()
        self.num_emb = num_emb
        self.emb_dim = emb_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.num_emb, self.emb_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_emb, 1.0 / self.num_emb)
        self.re_emb = num_emb

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous() # (b, c, h, w) --> (b, h, w, c)
        z_flatten = z.view(-1, self.emb_dim) # (b, h, w, c) --> (bhw, c)

        # distance from z to embeddings e should be calculated as: |z-e|^2 = |z|^2 + |e|^2 - 2 z.e
        # This has better computational speed, easier backprop, and better precision.
        # (z-e) might cancel very similar values i.e., rounding-off errors.
        emb_distance = torch.sum(z_flatten ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) + \
            -2 * torch.einsum("bd,dn->bn", z_flatten, self.embedding.weight.permute(1, 0))
        
        min_emb_indices = torch.argmin(emb_distance, dim=1) # non-differentiable
        z_quantise = self.embedding(min_emb_indices).view(z.shape) # (bhw, c) --> (b, h, w, c)

        # compute loss for embedding and preserve gradients
        loss = self.beta * torch.mean((z_quantise.detach() - z)**2) + torch.mean((z_quantise - z.detach())**2)

        # preserve gradients using straight-through gradient-estimation
        z_quantise = z + (z_quantise - z).detach()
        # Use contiguous as it will improve performance by storing tensor in a new memory contiguously
        z_quantise = z_quantise.permute(0, 3, 1, 2).contiguous() # (b, h, w, c) --> (b, c, h, w)

        return z_quantise, loss


class VQGAN(nn.Module):
    def __init__(self, resolution=256, z_channel=256, emb_dim=256):
        super().__init__()
        self.encoder = Encoder(in_channel=3, resolution=resolution, z_channel=z_channel)
        self.pre_quant_conv = nn.Conv2d(
            z_channel, emb_dim, kernel_size=1, stride=1, padding=0
        )
        self.quantise = VectorQuantizer(num_emb=1024, emb_dim=emb_dim)
        self.post_quant_conv = nn.Conv2d(
            emb_dim, z_channel, kernel_size=1, stride=1, padding=0
        )
        self.decoder = Decoder(out_channel=3, resolution=resolution, z_channel=z_channel)

    def forward(self, x, train=False):
        h = self.encoder(x)
        h = self.pre_quant_conv(h)
        z_quantise, loss_quantise = self.quantise(h)
        z_quantise = self.post_quant_conv(z_quantise)
        x_rec = self.decoder(z_quantise)

        return (x_rec, loss_quantise) if train else x_rec


if __name__ == "__main__":
    """
    Testing each module.
    Run: python -m pdb model.py
    break 391
    and inspect using: print (zq.requires_grad, out.requires_grad)

    """
    encoder = Encoder()
    quantiser = VectorQuantizer()
    decoder = Decoder()

    inp = torch.randn(16, 3, 256, 256).requires_grad_(True) # B x C x H x W
    z = encoder(inp) # B x 256 x 16 x 16
    zq, loss_q = quantiser(z)
    out = decoder(zq) # B x 3 x 256 x 256
    print (f"Shape of latent code: {z.shape} | quantised latent code: {zq.shape} | Output: {out.shape}")