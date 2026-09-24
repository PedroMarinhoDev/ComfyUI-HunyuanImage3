"""HunyuanImage-3.0 image VAE (hunyuan-image-vae-v1): 3D autoencoder, 32 latent channels, /16.

Ported from the reference `autoencoder_kl_3d.py`. Decoder first — decode is all text-to-image needs —
with the encoder included because the gate for this step is an encode/decode round trip and the
image-conditioning phase will want it.

Deliberate differences from the reference:

* built from `operations` (Conv3d/GroupNorm) like every other module in this port, so the
  ModelPatcher can see and offload it;
* the reference's temporally chunked `Conv3d` (a workaround for >2 GB activations) is not reproduced:
  that is memory management, which belongs to ComfyUI, not to the model;
* no tiling/slicing state on the model — ComfyUI's VAE wrapper owns that;
* no einops: reshape/permute, as the repo requires.

Attribute names follow the checkpoint keys exactly (`encoder.down.N.block.N.conv1`, `decoder.up.N.
upsample.conv`, `mid.attn_1.q`, ...) so the state dict maps without a key table.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x):
    return x * torch.sigmoid(x)


class HunyuanImage3VAEResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dtype=None, device=None, operations=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = operations.GroupNorm(32, in_channels, eps=1e-6, affine=True, dtype=dtype, device=device)
        self.conv1 = operations.Conv3d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        self.norm2 = operations.GroupNorm(32, self.out_channels, eps=1e-6, affine=True, dtype=dtype, device=device)
        self.conv2 = operations.Conv3d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = operations.Conv3d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device)

    def forward(self, x):
        h = self.conv1(swish(self.norm1(x)))
        h = self.conv2(swish(self.norm2(h)))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class HunyuanImage3VAEAttnBlock(nn.Module):
    def __init__(self, in_channels, dtype=None, device=None, operations=None):
        super().__init__()
        self.in_channels = in_channels
        self.norm = operations.GroupNorm(32, in_channels, eps=1e-6, affine=True, dtype=dtype, device=device)
        self.q = operations.Conv3d(in_channels, in_channels, kernel_size=1, dtype=dtype, device=device)
        self.k = operations.Conv3d(in_channels, in_channels, kernel_size=1, dtype=dtype, device=device)
        self.v = operations.Conv3d(in_channels, in_channels, kernel_size=1, dtype=dtype, device=device)
        self.proj_out = operations.Conv3d(in_channels, in_channels, kernel_size=1, dtype=dtype, device=device)

    def attention(self, h_):
        h_ = self.norm(h_)
        b, c, f, h, w = h_.shape
        q = self.q(h_).reshape(b, 1, f * h * w, c)
        k = self.k(h_).reshape(b, 1, f * h * w, c)
        v = self.v(h_).reshape(b, 1, f * h * w, c)
        out = F.scaled_dot_product_attention(q, k, v)
        return out.reshape(b, f, h, w, c).permute(0, 4, 1, 2, 3)

    def forward(self, x):
        return x + self.proj_out(self.attention(x))


class HunyuanImage3VAEDownsample(nn.Module):
    """DownsampleDCAE: conv to out_channels/factor, then the channel-fold, plus a mean-pooled shortcut."""

    def __init__(self, in_channels, out_channels, add_temporal_downsample=True, dtype=None, device=None, operations=None):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_downsample else 1 * 2 * 2
        assert out_channels % factor == 0
        self.conv = operations.Conv3d(in_channels, out_channels // factor, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        self.add_temporal_downsample = add_temporal_downsample
        self.group_size = factor * in_channels // out_channels

    def _fold(self, x, channels):
        r1 = 2 if self.add_temporal_downsample else 1
        b, _, f_in, h_in, w_in = x.shape
        # equivalent to rearrange(x, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w"): the composed
        # axes split with the *first* named dim outermost, so f = f_in / r1 and h = h_in / 2
        x = x.reshape(b, channels, f_in // r1, r1, h_in // 2, 2, w_in // 2, 2)
        x = x.permute(0, 3, 5, 7, 1, 2, 4, 6)
        return x.reshape(b, channels * r1 * 4, f_in // r1, h_in // 2, w_in // 2)

    def forward(self, x):
        r1 = 2 if self.add_temporal_downsample else 1
        h = self._fold(self.conv(x), self.conv.out_channels)
        shortcut = self._fold(x, self.conv.in_channels)
        b, _, t, hh, ww = shortcut.shape
        shortcut = shortcut.view(b, h.shape[1], self.group_size, t, hh, ww).mean(dim=2)
        return h + shortcut


class HunyuanImage3VAEUpsample(nn.Module):
    """UpsampleDCAE: conv to out_channels*factor, channel-unfold, plus a repeat_interleave shortcut."""

    def __init__(self, in_channels, out_channels, add_temporal_upsample=True, dtype=None, device=None, operations=None):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_upsample else 1 * 2 * 2
        self.conv = operations.Conv3d(in_channels, out_channels * factor, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        self.add_temporal_upsample = add_temporal_upsample
        self.repeats = factor * out_channels // in_channels

    def _unfold(self, x, channels):
        r1 = 2 if self.add_temporal_upsample else 1
        b, _, f, h, w = x.shape
        # equivalent to rearrange(x, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)")
        x = x.reshape(b, r1, 2, 2, channels, f, h, w)
        x = x.permute(0, 4, 5, 1, 6, 2, 7, 3)
        return x.reshape(b, channels, f * r1, h * 2, w * 2)

    def forward(self, x):
        out_channels = self.conv.out_channels // (2 * 2 * 2 if self.add_temporal_upsample else 4)
        h = self._unfold(self.conv(x), out_channels)
        # the repeat_interleave is sized so the shortcut carries out_channels * factor too, so it
        # unfolds with the same grouping as the conv output
        shortcut = self._unfold(x.repeat_interleave(repeats=self.repeats, dim=1), out_channels)
        return h + shortcut


class HunyuanImage3VAEEncoder(nn.Module):
    def __init__(self, in_channels, z_channels, block_out_channels, num_res_blocks, ffactor_spatial, ffactor_temporal,
                 downsample_match_channel=True, dtype=None, device=None, operations=None):
        super().__init__()
        assert block_out_channels[-1] % (2 * z_channels) == 0

        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks

        self.conv_in = operations.Conv3d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)

        self.down = nn.ModuleList()
        block_in = block_out_channels[0]
        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch
            for _ in range(self.num_res_blocks):
                block.append(HunyuanImage3VAEResnetBlock(block_in, block_out, dtype=dtype, device=device, operations=operations))
                block_in = block_out
            down = nn.Module()
            down.block = block

            add_spatial_downsample = bool(i_level < math.log2(ffactor_spatial))
            add_temporal_downsample = add_spatial_downsample and bool(i_level >= math.log2(ffactor_spatial // ffactor_temporal))
            if add_spatial_downsample or add_temporal_downsample:
                assert i_level < len(block_out_channels) - 1
                block_out = block_out_channels[i_level + 1] if downsample_match_channel else block_in
                down.downsample = HunyuanImage3VAEDownsample(block_in, block_out, add_temporal_downsample, dtype=dtype, device=device, operations=operations)
                block_in = block_out
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = HunyuanImage3VAEResnetBlock(block_in, block_in, dtype=dtype, device=device, operations=operations)
        self.mid.attn_1 = HunyuanImage3VAEAttnBlock(block_in, dtype=dtype, device=device, operations=operations)
        self.mid.block_2 = HunyuanImage3VAEResnetBlock(block_in, block_in, dtype=dtype, device=device, operations=operations)

        self.norm_out = operations.GroupNorm(32, block_in, eps=1e-6, affine=True, dtype=dtype, device=device)
        self.conv_out = operations.Conv3d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)

    def forward(self, x):
        h = self.conv_in(x)
        for i_level in range(len(self.block_out_channels)):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
            if hasattr(self.down[i_level], "downsample"):
                h = self.down[i_level].downsample(h)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        group_size = self.block_out_channels[-1] // (2 * self.z_channels)
        shortcut = h.view(h.shape[0], 2 * self.z_channels, group_size, *h.shape[2:]).mean(dim=2)
        h = self.conv_out(swish(self.norm_out(h)))
        return h + shortcut


class HunyuanImage3VAEDecoder(nn.Module):
    def __init__(self, z_channels, out_channels, block_out_channels, num_res_blocks, ffactor_spatial, ffactor_temporal,
                 upsample_match_channel=True, dtype=None, device=None, operations=None):
        super().__init__()
        assert block_out_channels[0] % z_channels == 0

        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks

        block_in = block_out_channels[0]
        self.conv_in = operations.Conv3d(z_channels, block_in, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)

        self.mid = nn.Module()
        self.mid.block_1 = HunyuanImage3VAEResnetBlock(block_in, block_in, dtype=dtype, device=device, operations=operations)
        self.mid.attn_1 = HunyuanImage3VAEAttnBlock(block_in, dtype=dtype, device=device, operations=operations)
        self.mid.block_2 = HunyuanImage3VAEResnetBlock(block_in, block_in, dtype=dtype, device=device, operations=operations)

        self.up = nn.ModuleList()
        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch
            for _ in range(self.num_res_blocks + 1):
                block.append(HunyuanImage3VAEResnetBlock(block_in, block_out, dtype=dtype, device=device, operations=operations))
                block_in = block_out
            up = nn.Module()
            up.block = block

            add_spatial_upsample = bool(i_level < math.log2(ffactor_spatial))
            add_temporal_upsample = bool(i_level < math.log2(ffactor_temporal))
            if add_spatial_upsample or add_temporal_upsample:
                assert i_level < len(block_out_channels) - 1
                block_out = block_out_channels[i_level + 1] if upsample_match_channel else block_in
                up.upsample = HunyuanImage3VAEUpsample(block_in, block_out, add_temporal_upsample, dtype=dtype, device=device, operations=operations)
                block_in = block_out
            self.up.append(up)

        self.norm_out = operations.GroupNorm(32, block_in, eps=1e-6, affine=True, dtype=dtype, device=device)
        self.conv_out = operations.Conv3d(block_in, out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)

    def forward(self, z):
        repeats = self.block_out_channels[0] // self.z_channels
        h = self.conv_in(z) + z.repeat_interleave(repeats=repeats, dim=1)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        for i_level in range(len(self.block_out_channels)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
            if hasattr(self.up[i_level], "upsample"):
                h = self.up[i_level].upsample(h)

        return self.conv_out(swish(self.norm_out(h)))


class HunyuanImage3VAE(nn.Module):
    """The 3D autoencoder. `scaling_factor` is the latent scale the transformer's latents are in.

    Text-to-image decode is: `latents / scaling_factor`, unsqueeze to (b, 32, 1, h, w), decode, keep
    the last temporal frame, squeeze back, then `/ 2 + 0.5`.
    """

    def __init__(self, in_channels=3, out_channels=3, latent_channels=32, block_out_channels=(128, 256, 512, 1024, 1024),
                 layers_per_block=2, ffactor_spatial=16, ffactor_temporal=4, scaling_factor=0.562679178327931,
                 downsample_match_channel=True, upsample_match_channel=True, dtype=None, device=None, operations=None):
        super().__init__()
        self.ffactor_spatial = ffactor_spatial
        self.ffactor_temporal = ffactor_temporal
        self.scaling_factor = scaling_factor

        self.encoder = HunyuanImage3VAEEncoder(
            in_channels=in_channels, z_channels=latent_channels, block_out_channels=list(block_out_channels),
            num_res_blocks=layers_per_block, ffactor_spatial=ffactor_spatial, ffactor_temporal=ffactor_temporal,
            downsample_match_channel=downsample_match_channel, dtype=dtype, device=device, operations=operations,
        )
        self.decoder = HunyuanImage3VAEDecoder(
            z_channels=latent_channels, out_channels=out_channels, block_out_channels=list(reversed(block_out_channels)),
            num_res_blocks=layers_per_block, ffactor_spatial=ffactor_spatial, ffactor_temporal=ffactor_temporal,
            upsample_match_channel=upsample_match_channel, dtype=dtype, device=device, operations=operations,
        )

    def encode(self, x):
        """(b, 3, T, H, W) -> the posterior mean (b, 32, T, h, w). A single frame is expanded to ffactor_temporal."""
        if x.dim() == 4:
            x = x.unsqueeze(2)
        if x.shape[2] == 1:
            x = x.expand(-1, -1, self.ffactor_temporal, -1, -1)
        else:
            assert x.shape[2] % self.ffactor_temporal == 0
        h = self.encoder(x)
        return h[:, : h.shape[1] // 2]              # the mean half; the other half is logvar

    def decode(self, z):
        """(b, 32, 1, h, w) -> (b, 3, 1, H, W), the temporal frame the reference keeps for T=1."""
        decoded = self.decoder(z)
        if z.shape[2] == 1:
            decoded = decoded[:, :, -1:]
        return decoded
