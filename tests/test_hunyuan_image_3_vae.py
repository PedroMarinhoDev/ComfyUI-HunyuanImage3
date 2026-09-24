"""Step 5 tests for the HunyuanImage-3.0 VAE port.

The DCAE blocks are the part of the port where a reshape/permute error would be silent: the
reference expresses them with einops, and the channel/axis ordering is not obvious from the shapes.
These tests pin the port's reshape/permute form to the reference's einops expressions exactly, and
(checkpoint-gated) to the converted checkpoint's key set.
"""
import os

import pytest
import torch
from einops import rearrange

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ops
from comfy.ldm.hunyuan_image_3.vae import HunyuanImage3VAEDownsample, HunyuanImage3VAEUpsample, HunyuanImage3VAE

VAE_PATH = os.environ.get("HUNYUAN_IMAGE_3_VAE", "")
needs_vae_file = pytest.mark.skipif(not os.path.exists(VAE_PATH), reason="set HUNYUAN_IMAGE_3_VAE to the converted VAE file")

VAE_KWARGS = dict(
    in_channels=3, out_channels=3, latent_channels=32, block_out_channels=(128, 256, 512, 1024, 1024),
    layers_per_block=2, ffactor_spatial=16, ffactor_temporal=4, scaling_factor=0.562679178327931,
    downsample_match_channel=True, upsample_match_channel=True,
)


def fold_only(add_temporal_downsample):
    module = HunyuanImage3VAEDownsample.__new__(HunyuanImage3VAEDownsample)
    module.add_temporal_downsample = add_temporal_downsample
    return module


def unfold_only(add_temporal_upsample):
    module = HunyuanImage3VAEUpsample.__new__(HunyuanImage3VAEUpsample)
    module.add_temporal_upsample = add_temporal_upsample
    return module


@pytest.mark.parametrize("add_temporal", [True, False])
def test_dcae_fold_matches_the_reference_einops_pattern(add_temporal):
    torch.manual_seed(0)
    r1 = 2 if add_temporal else 1
    x = torch.randn(2, 128, 4, 32, 48)
    reference = rearrange(x, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
    got = fold_only(add_temporal)._fold(x, 128)
    assert got.shape == reference.shape
    assert torch.equal(got, reference)


@pytest.mark.parametrize("add_temporal", [True, False])
def test_dcae_unfold_matches_the_reference_einops_pattern(add_temporal):
    torch.manual_seed(0)
    r1 = 2 if add_temporal else 1
    x = torch.randn(2, 128 * r1 * 4, 3, 16, 24)
    reference = rearrange(x, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
    got = unfold_only(add_temporal)._unfold(x, 128)
    assert got.shape == reference.shape
    assert torch.equal(got, reference)


def test_dcae_block_shapes_follow_the_configured_factors():
    # spatial /2 per level for the first log2(16) = 4 levels, temporal /2 for the first log2(4) = 2
    x = torch.randn(1, 128, 4, 64, 64)
    module = HunyuanImage3VAEDownsample(128, 256, add_temporal_downsample=False, operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        out = module(x)
    assert out.shape == (1, 256, 4, 32, 32)

    module = HunyuanImage3VAEDownsample(128, 256, add_temporal_downsample=True, operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        out = module(x)
    assert out.shape == (1, 256, 2, 32, 32)

    # and the upsampler is its inverse on shapes
    up = HunyuanImage3VAEUpsample(256, 128, add_temporal_upsample=True, operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        back = up(out)
    assert back.shape == (1, 128, 4, 64, 64)


@needs_vae_file
def test_vae_key_map_covers_the_checkpoint_exactly():
    from safetensors import safe_open
    # Read the raw keys, with no prefix filter: the harnesses that stripped `vae.` on every call site are
    # what hid a file whose keys carried the prefix, and a prefixed file matches no branch in the loader
    # (stock VAELoader looks for `decoder.conv_in.weight`). Reading it the way its consumer does is the
    # point of this test.
    with safe_open(VAE_PATH, framework="pt", device="cpu") as handle:
        checkpoint_keys = set(handle.keys())

    with torch.device("meta"):
        model = HunyuanImage3VAE(dtype=torch.float16, device="meta", operations=comfy.ops.disable_weight_init, **VAE_KWARGS)
    model_keys = set(model.state_dict())

    assert not any(key.startswith("vae.") for key in checkpoint_keys), "the file must carry unprefixed keys"
    assert len(checkpoint_keys) == 280
    assert model_keys - checkpoint_keys == set()      # nothing the checkpoint does not have
    assert checkpoint_keys - model_keys == set()      # nothing it has that we do not build
