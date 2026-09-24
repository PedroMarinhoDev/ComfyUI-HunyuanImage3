"""Tests for the conv embedding path: `patch_embed` (UNetDown) and `final_layer` (UNetUp).

These forwards were owed from Step 2 and are what turn latent patches into the model's image tokens
and back. Small geometry: the real one is 4096 channels over a 64x64 grid, which is unnecessary to
exercise here.
"""
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ops
import pytest

from comfy.ldm.hunyuan_image_3.model import ResBlock, TimestepEmbedder, UNetDown, UNetUp

from op_module_init import init_op_modules

OPS = comfy.ops.disable_weight_init
# GroupNorm in this model is always 32 groups (reference `normalization()`), so every channel count
# here has to be a multiple of 32 — the real geometry is 4096 channels over a 64x64 grid, which is
# unnecessary to exercise the code path
IN_CHANNELS, EMB, HIDDEN, OUT = 32, 16, 32, 64
GRID = 8


def build(module_class, *args):
    return init_op_modules(module_class(*args, operations=OPS), dtype=torch.float32)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_patch_embed_flattens_the_latent_grid(dtype):
    model = build(UNetDown, IN_CHANNELS, EMB, HIDDEN, OUT).to(dtype)
    x = torch.randn(2, IN_CHANNELS, GRID, GRID, dtype=dtype)
    emb = torch.randn(2, EMB, dtype=dtype)
    with torch.no_grad():
        tokens, token_h, token_w = model(x, emb)
    assert tokens.shape == (2, GRID * GRID, OUT)
    assert (token_h, token_w) == (GRID, GRID)
    assert torch.isfinite(tokens.float()).all()
    # the flatten is (b, c, h, w) -> (b, h*w, c) in row-major patch order: recomputing the two stages
    # for the same batch must give exactly the returned tokens
    with torch.no_grad():
        conv = model.model[0](x)
        blocks = model.model[1](conv, emb)
    assert torch.equal(tokens, blocks.flatten(2).transpose(1, 2))
    assert torch.equal(tokens[0, 0], blocks[0, :, 0, 0])


def test_final_layer_inverts_the_token_grid():
    down = build(UNetDown, IN_CHANNELS, EMB, HIDDEN, OUT)
    up = build(UNetUp, OUT, EMB, HIDDEN, IN_CHANNELS)
    x = torch.randn(1, IN_CHANNELS, GRID, GRID)
    emb = torch.randn(1, EMB)
    with torch.no_grad():
        tokens, token_h, token_w = down(x, emb)
        out = up(tokens, emb, token_h, token_w)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_resblock_uses_the_timestep_embedding():
    model = build(ResBlock, HIDDEN, EMB, HIDDEN)
    x = torch.randn(1, HIDDEN, GRID, GRID)
    with torch.no_grad():
        first = model(x, torch.zeros(1, EMB))
        second = model(x, torch.ones(1, EMB) * 3)
    assert first.shape == x.shape
    # the adaptive group norm means the embedding has to change the result
    assert not torch.allclose(first, second)
    # zero embedding is not a no-op: emb_layers has a bias, exactly as in the reference
    assert not torch.allclose(first, x)


def test_resblock_changes_channels_through_the_skip_connection():
    model = build(ResBlock, HIDDEN, EMB, OUT)
    x = torch.randn(1, HIDDEN, GRID, GRID)
    with torch.no_grad():
        out = model(x, torch.randn(1, EMB))
    assert out.shape == (1, OUT, GRID, GRID)
    assert isinstance(model.skip_connection, comfy.ops.disable_weight_init.Conv2d)


def test_timestep_embedder_matches_the_reference_frequencies():
    embedder = build(TimestepEmbedder, EMB)
    with torch.no_grad():
        embedding = embedder(torch.tensor([0.0, 1.0]))
    assert embedding.shape == (2, EMB)
    assert torch.isfinite(embedding).all()
    # t = 0 gives cos(0) = 1 for every frequency, so the first half of the sinusoid block is all ones;
    # the MLP then mixes it, so this is checked on the sinusoid helper, not the output
    from comfy.ldm.hunyuan_image_3.model import timestep_embedding
    sinusoids = timestep_embedding(torch.tensor([0.0]), 256)
    assert sinusoids.shape == (1, 256)
    assert torch.allclose(sinusoids[0, :128], torch.ones(128))
    assert torch.allclose(sinusoids[0, 128:], torch.zeros(128))
