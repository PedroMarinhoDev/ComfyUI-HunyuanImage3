"""Step 6a tests: the input sequence builder.

A wrong sequence layout surfaces only as a bad image with no diagnostic, so the layout is pinned
structurally here and, where the tokenizer file is available, against the ids the reference tokenizer
produces.

The ratio-token ids are pinned specifically because they are *not* a contiguous block:
`<img_ratio_0..32>` sit at 128044..128076 and `<img_ratio_33..36>` at 130103..130106 — the four
appended resolutions got their own id block. Anything computing `128044 + idx` is therefore correct
up to 32 and silently wrong for 33..36, which are 1024x768, 1280x720, 768x1024 and 720x1280, i.e. the
most commonly requested non-square ratios including 16:9. This port looks ids up by name; this test
exists so a later "simplification" to arithmetic cannot break exactly those sizes.
"""
import functools
import os

import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy.ldm.hunyuan_image_3.system_prompt import UNIFIED_SYSTEM_PROMPT_EN
from comfy.ldm.hunyuan_image_3.tokenizer import (ImageGeometry, ResolutionGroup, build_sequence as _build_sequence,
                                                 build_text_sequence)

# The reference captures these tests compare against were taken from the Instruct-Distil checkpoint,
# whose sequence carries `<guidance>` and `<timestep_r>`. The pack makes that contract a required
# keyword rather than a default — the flags have to travel with the weights, since rendering one
# checkpoint under another's contract is silent — so the tests bind it once here instead of repeating
# it at every call.
_DISTIL = {"cfg_distilled": True, "use_meanflow": True}
build_sequence = functools.partial(_build_sequence, **_DISTIL)

TOKENIZER_PATH = os.environ.get(
    "HUNYUAN_IMAGE_3_TOKENIZER", "")
needs_tokenizer = pytest.mark.skipif(
    not os.path.exists(TOKENIZER_PATH), reason="set HUNYUAN_IMAGE_3_TOKENIZER to the repo's tokenizer.json")

PROMPT = "A young woman in a green dress reads by a window"

# Verified against the reference tokenizer itself (step6a_port_vs_reference.py):
# size -> (image_height, image_width, token_height, token_width, ratio_index, total_ids)
REFERENCE_SNAP = """
What the reference's `build_gen_image_info` does, captured in Step 6a as
(height, width, token_height, token_width, ratio_index, total):

    1024x1024 -> 1024x1024   64x64   ratio 16   (row 16 of 37)
    512x512   -> 1024x1024   64x64   ratio 16   <- snapped up
    2048x2048 -> 1024x1024   64x64   ratio 16   <- snapped down
    1344x768  -> 1280x720    80x45   ratio 34
    1024x256  -> 2048x512    128x32  ratio 32

This port used to mirror that snap. Addendum 29 established that RoPE comes from the token grid and
`<img_ratio_N>` is only a semantic hint, so the grid now follows the request and only the hint snaps —
which is why the assertions below are split in two rather than checked against a single capture.
"""

GRID_SIZES = ["1024x1024", "512x512", "2048x2048", "1344x768", "1024x256", "768x1024", "1152x896", "1200x880"]
RATIO_NAMES = ["<img_ratio_0>", "<img_ratio_3>", "<img_ratio_16>", "<img_ratio_36>"]


@pytest.fixture(scope="module")
def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(TOKENIZER_PATH)


def test_resolution_group_layout():
    group = ResolutionGroup(1024)
    assert len(group) == 37
    # the staircase is sorted by ratio; the four extra resolutions are appended afterwards
    staircase = group.data[:33]
    assert staircase == sorted(staircase, key=lambda resolution: resolution.ratio)
    assert [(resolution.height, resolution.width) for resolution in group.data[33:]] == [
        (1024, 768), (1280, 720), (768, 1024), (720, 1280)]
    # 1024x1024 is row 16 of 37, not the middle, precisely because the extras are not merged in
    assert (group[16].height, group[16].width) == (1024, 1024)


@needs_tokenizer
def test_ratio_token_ids_are_not_contiguous(tokenizer):
    ids = [tokenizer.token_to_id(f"<img_ratio_{index}>") for index in range(37)]
    assert ids[:33] == list(range(128044, 128077))
    assert ids[33:] == [130103, 130104, 130105, 130106]
    # the discontinuous block is exactly the four appended resolutions, in table order
    rows = [(resolution.height, resolution.width) for resolution in ResolutionGroup(1024).data[33:]]
    assert rows == [(1024, 768), (1280, 720), (768, 1024), (720, 1280)]
    # and every row the port can emit has a token: <img_ratio_34> is 1280x720 (16:9-ish), a common
    # request, and 128044 + 34 would be the wrong id for it
    assert tokenizer.token_to_id("<img_ratio_34>") == 130104


CAPTURE_PATH = os.environ.get(
    "HUNYUAN_IMAGE_3_CAPTURE", "")
needs_capture = pytest.mark.skipif(
    not os.path.exists(CAPTURE_PATH), reason="set HUNYUAN_IMAGE_3_CAPTURE to the reference capture")


@needs_tokenizer
@needs_capture
def test_sequences_match_the_reference_capture(tokenizer):
    """All four layouts, id for id, against the reference's own tokenizer on the deployed path.

    The capture comes from `step11d_instruct_capture.py`, which reads `sequence_template` out of the
    checkpoint's `generation_config.json` the way `prepare_model_inputs` does (`"instruct"`, not the
    template's own `"pretrain"` default) and applies the `system_prompt.strip()` `generate` performs.
    Both of those were wrong in the first two captures, and each wrong version produced a layout that
    runs, renders a plausible image and says nothing — so this test pins the ids, not the shapes
    (spec §48–§50).
    """
    capture = torch.load(CAPTURE_PATH, weights_only=False)
    sequences = capture["sequences"]
    system_prompt, prompt = capture["system_prompt"], capture["prompt"]
    assert capture["sequence_template"] == "instruct"

    assert build_sequence(tokenizer, prompt, "1024x1024", system_prompt)["ids"].tolist() == \
        sequences["t2i_plain"]
    assert build_sequence(tokenizer, prompt, "1024x1024", system_prompt,
                          cot_text=capture["cot_text"])["ids"].tolist() == sequences["t2i_cot"]
    assert build_text_sequence(tokenizer, prompt, system_prompt, "think")["ids"].tolist() == \
        sequences["text_stage"]
    assert build_sequence(tokenizer, prompt, "1024x1024", system_prompt,
                          cond_images=[(capture["cond_size"], (32, 32))])["ids"].tolist() == sequences["i2i"]


def test_the_reasoning_stage_defaults_come_from_the_checkpoint():
    """The reference's reasoning stage samples: the checkpoint says so itself.

    `generation_config.json` sets `do_sample: true` with `temperature 0.6, top_p 0.95, top_k 1024`, and
    HF's `GenerationConfig` defaults (greedy) are what a harness that constructs the model directly
    sees instead. Getting this backwards makes the stage run, produce fluent text and simply not be the
    reference's behaviour.
    """
    from comfy_extras.nodes_hunyuan_image_3 import HunyuanImage3PromptRewriting

    schema = HunyuanImage3PromptRewriting.define_schema()
    defaults = {entry.id: getattr(entry, "default", None) for entry in schema.inputs}
    assert defaults["do_sample"] is True
    assert defaults["temperature"] == 0.6
    assert defaults["top_p"] == 0.95
    assert defaults["top_k"] == 1024
    # the checkpoint's own budget, not a bound of mine: a shorter one stops the model mid-analysis and
    # the image conditions on a partial trace. Measured 0.92 s/token with the per-expert cast, so it is
    # ~half an hour on a 24 GB card; the stage ends early on its own closing tag when it can.
    assert defaults["max_new_tokens"] == 2048
    assert defaults["enabled"] is True


@needs_tokenizer
@pytest.mark.parametrize("size", GRID_SIZES)
def test_grid_follows_the_requested_size(size, tokenizer):
    """A request keeps its size: the token grid is the request over the VAE's spatial factor.

    This is the half that used to snap, and the deviation is deliberate — see REFERENCE_SNAP.
    """
    width, height = (int(part) for part in size.split("x"))
    sequence = build_sequence(tokenizer, PROMPT, size, UNIFIED_SYSTEM_PROMPT_EN)
    assert (sequence["image_width"], sequence["image_height"]) == (width, height)
    assert (sequence["token_width"], sequence["token_height"]) == (width // 16, height // 16)
    assert sequence["image_slice"].stop - sequence["image_slice"].start == (width // 16) * (height // 16)
    assert set(sequence["ids"][sequence["image_slice"]].tolist()) == {tokenizer.token_to_id("<img>")}


@needs_tokenizer
@pytest.mark.parametrize("size", GRID_SIZES)
def test_only_the_ratio_hint_snaps(size, tokenizer):
    """`<img_ratio_N>` is the nearest table row by aspect ratio: a hint, not the geometry."""
    width, height = (int(part) for part in size.split("x"))
    group = ResolutionGroup(1024)
    sequence = build_sequence(tokenizer, PROMPT, size, UNIFIED_SYSTEM_PROMPT_EN)
    assert sequence["ratio_index"] == group.nearest_index(width, height)


@needs_tokenizer
@pytest.mark.parametrize("name", RATIO_NAMES)
def test_ratio_named_requests_take_the_whole_row(name, tokenizer):
    """`<img_ratio_N>` does name geometry: there is no size to honour, so the row is the answer."""
    index = int(name.split("_")[-1].rstrip(">"))
    group = ResolutionGroup(1024)
    sequence = build_sequence(tokenizer, PROMPT, name, UNIFIED_SYSTEM_PROMPT_EN)
    assert sequence["ratio_index"] == index
    assert (sequence["image_width"], sequence["image_height"]) == (group[index].width, group[index].height)
    assert (sequence["token_width"], sequence["token_height"]) == (group[index].width // 16,
                                                                   group[index].height // 16)


@needs_tokenizer
def test_sequence_layout(tokenizer):
    sequence = build_sequence(tokenizer, PROMPT, "1024x1024", UNIFIED_SYSTEM_PROMPT_EN)
    ids = sequence["ids"].tolist()
    image_ids = ids[sequence["image_slice"]]

    assert ids[0] == tokenizer.token_to_id("<|startoftext|>")
    assert len(image_ids) == sequence["token_height"] * sequence["token_width"]
    assert set(image_ids) == {tokenizer.token_to_id("<img>")}
    assert ids[sequence["image_slice"].stop] == tokenizer.token_to_id("<eoi>")
    assert ids[sequence["image_slice"].start - 1] == tokenizer.token_to_id("<timestep_r>")
    assert ids[sequence["timestep_position"]] == tokenizer.token_to_id("<timestep>")
    assert ids[sequence["guidance_position"]] == tokenizer.token_to_id("<guidance>")
    assert ids[sequence["timestep_r_position"]] == tokenizer.token_to_id("<timestep_r>")
    # <boi>, <img_size_base>, <img_ratio_i> immediately before the meta tokens
    assert ids[sequence["image_slice"].start - 6] == tokenizer.token_to_id("<boi>")
    assert ids[sequence["image_slice"].start - 5] == tokenizer.token_to_id("<img_size_1024>")
    assert ids[sequence["image_slice"].start - 4] == tokenizer.token_to_id("<img_ratio_16>")
    # and the reference's t2i path appends no eos
    assert ids[-1] != tokenizer.token_to_id("<|endoftext|>")


@needs_tokenizer
def test_size_token_carries_the_base_size_not_the_image_size(tokenizer):
    # the regression pin for a bug the reference diff caught: <img_size_N> is the resolution table's
    # base size, while the absolute size comes from the ratio row
    sequence = build_sequence(tokenizer, PROMPT, "768x1024", UNIFIED_SYSTEM_PROMPT_EN)
    ids = sequence["ids"].tolist()
    assert (sequence["image_width"], sequence["image_height"]) == (768, 1024)
    assert ids[sequence["image_slice"].start - 5] == tokenizer.token_to_id("<img_size_1024>")


@needs_tokenizer
def test_prompt_text_is_encoded_without_special_tokens(tokenizer):
    prompt = "a red teapot"
    prefix = build_sequence(tokenizer, prompt, "1024x1024", UNIFIED_SYSTEM_PROMPT_EN)["ids"].tolist()
    suffix = build_sequence(tokenizer, prompt + "!!", "1024x1024", UNIFIED_SYSTEM_PROMPT_EN)["ids"].tolist()
    # the prompt tokens appear verbatim (no bos/eos injected around each text section): appending to
    # the prompt extends the sequence by exactly the extra tokens
    assert len(suffix) - len(prefix) == len(tokenizer.encode("!!", add_special_tokens=False).ids)


@needs_tokenizer
def test_the_rewritten_prompt_is_extracted_by_token_id(tokenizer):
    """The `<recaption>` body comes back out of the generated ids, and a missing one is distinguishable.

    The delimiters are special tokens, and `decode([<special>])` returns `""` by default, so an
    extraction that splits *decoded text* on them finds nothing — which looks identical to a run whose
    budget ran out mid-analysis. Slice by id instead, and return `""` for both cases so the caller can
    warn about the truncation rather than presenting it as an empty rewrite.
    """
    from comfy.ldm.hunyuan_image_3.rewrite import recaption_from_ids as _recaption_from_ids

    recaption = tokenizer.token_to_id("<recaption>")
    end_of_recaption = tokenizer.token_to_id("</recaption>")
    body = "a red fox asleep in tall grass"
    body_ids = list(tokenizer.encode(body).ids)

    complete = [1, 2, 3, recaption] + body_ids + [end_of_recaption, 4]
    assert _recaption_from_ids(tokenizer, complete, recaption, end_of_recaption) == body

    # a budget that ran out before the rewrite began: the opening tag is never reached
    assert _recaption_from_ids(tokenizer, [1, 2, 3], recaption, end_of_recaption) == ""
    # reached the opening tag but not its closing one
    assert _recaption_from_ids(tokenizer, [recaption] + body_ids, recaption, end_of_recaption) == ""
    # the model emitted an empty rewrite
    assert _recaption_from_ids(tokenizer, [recaption, end_of_recaption], recaption, end_of_recaption) == ""


def test_the_conditioning_strengths_and_cot_head_are_declared():
    """The two conditioning strengths default to the reference's behaviour, and the head defaults to auto.

    Both strengths are diagnostic knobs added on request: at 1.0 the node is exactly the reference. The
    head lives on the rewriting node, where `auto` picks the file matching the loaded model.
    """
    from comfy_extras.nodes_hunyuan_image_3 import HunyuanImage3ImageEncode, HunyuanImage3PromptRewriting

    encode = {entry.id: getattr(entry, "default", None)
              for entry in HunyuanImage3ImageEncode.define_schema().inputs}
    assert encode["vit_strength"] == 1.0
    assert encode["latent_strength"] == 1.0

    rewriting = {entry.id: getattr(entry, "default", None)
                 for entry in HunyuanImage3PromptRewriting.define_schema().inputs}
    assert rewriting["cot_head"] == "auto"


def test_the_size_token_is_derived_from_the_requested_area():
    """`<img_size_N>` follows the requested area rather than always claiming 1024.

    Every table's rows span roughly 0.79N^2 to 1.05N^2, so a request above the 1024 band belongs to a
    larger table the vocabulary already has. Reporting 1536x1536 as `<img_size_1024>` described a
    9216-token grid as living in the 1024 position range, which is why sizes beyond the band generated
    scrambled images. Sizes inside the band must not move, which is what keeps everything already
    working: the off-table 1200x880 case and the preset rows still resolve to 1024.
    """
    from comfy.ldm.hunyuan_image_3.tokenizer import ImageGeometry

    for size, base, tokens in (("1024x1024", 1024, 4096), ("768x1024", 1024, 3072),
                               ("1200x880", 1024, 4125), ("512x512", 1024, 1024),
                               ("1280x1280", 1536, 6400), ("1536x1536", 1536, 9216),
                               ("2048x2048", 2048, 16384), ("<img_ratio_16>", 1024, 4096)):
        geometry = ImageGeometry(size)
        assert geometry.base_size == base, f"{size} chose base {geometry.base_size}, expected {base}"
        assert geometry.image_token_length == tokens, \
            f"{size} has {geometry.image_token_length} image tokens, expected {tokens}"


COND_GEOMETRY = ((1216, 832), (39, 26))  # the reviewer's test image: 1014 tower patches
PLAIN_FLAGS = {"cfg_distilled": False, "use_meanflow": False}


def _cond_sequence(tokenizer, **overrides):
    flags = dict(PLAIN_FLAGS)
    flags.update(overrides)
    return _build_sequence(tokenizer, PROMPT, "1024x1024", UNIFIED_SYSTEM_PROMPT_EN,
                           cond_images=[COND_GEOMETRY], **flags)


@needs_tokenizer
def test_compact_tower_runs_are_still_the_default(tokenizer):
    """`reference_vit_padding` off (the default) keeps the true-grid tower run."""
    blocks = _cond_sequence(tokenizer)["cond_blocks"]
    assert len(blocks) == 1
    vit = blocks[0]["vit_slice"]
    assert vit.stop - vit.start == 39 * 26


@needs_tokenizer
def test_reference_vit_padding_pads_the_tower_run(tokenizer):
    """On: the tower run is 1024 `<img>` — the default ids with the pads spliced out."""
    default = _cond_sequence(tokenizer)
    padded = _cond_sequence(tokenizer, reference_vit_padding=True)
    dflt, pad = default["cond_blocks"][0], padded["cond_blocks"][0]
    assert pad["vit_slice"].stop - pad["vit_slice"].start == 1024
    assert (pad["patch_height"], pad["patch_width"]) == (39, 26)
    assert pad["joint_slice"].stop - pad["joint_slice"].start == \
        (dflt["joint_slice"].stop - dflt["joint_slice"].start) + (1024 - 39 * 26)
    img = tokenizer.token_to_id("<img>")
    ids_d, ids_p = default["ids"].tolist(), padded["ids"].tolist()
    assert len(ids_p) - len(ids_d) == 1024 - 39 * 26
    vit_start = pad["vit_slice"].start
    assert ids_p[vit_start + 39 * 26:vit_start + 1024] == [img] * (1024 - 39 * 26)
    assert ids_p[:vit_start + 39 * 26] + ids_p[vit_start + 1024:] == ids_d


def test_rope_grid_positions_are_truncated_to_integers():
    """The reference's `.long()`: beta_y = 5000 + (1014 - 39) / 2 = 5487.5 becomes 5487."""
    import math
    from comfy.ldm.hunyuan_image_3.model import build_rope_freqs
    info = [(slice(5000, 5000 + 1014), (39, 26))]
    freqs = build_rope_freqs(6252, 128, info, 10000.0)
    # pair 0 uses the y position at theta_0 = 1
    assert abs(freqs[0, 0, 5000, 0, 0, 0].item() - math.cos(5487)) < 1e-4


@needs_tokenizer
def test_padded_sequence_feeds_mask_and_rope(tokenizer):
    """The padded sequence survives `_sequence_from_ids`, the mask and the RoPE builders."""
    import types
    from comfy.ldm.hunyuan_image_3.model import _sequence_from_ids, build_rope_freqs
    from comfy.ldm.hunyuan_image_3.pipeline import build_attention_mask
    seq = _cond_sequence(tokenizer, reference_vit_padding=True)
    img = tokenizer.token_to_id("<img>")
    config = types.SimpleNamespace(image_token_id=img, cfg_distilled=False, use_meanflow=False)
    gen_latent = torch.zeros(1, 16, 64, 64)
    cond_latent = [torch.zeros(1, 16, 76, 52)]
    step = _sequence_from_ids(seq["ids"], gen_latent, config, cond_latent=cond_latent,
                              cond_patch_grid=[torch.tensor([39, 26])])
    assert step["cond_blocks"][0]["vit_slice"].stop - step["cond_blocks"][0]["vit_slice"].start == 1024
    seq_len = seq["ids"].shape[0]
    mask = build_attention_mask(step, seq_len, torch.bfloat16, torch.device("cpu"))
    assert tuple(mask.shape[-2:]) == (seq_len, seq_len)
    freqs = build_rope_freqs(seq_len, 128, step["rope_image_info"], 10000.0)
    assert freqs.shape[2] == seq_len


ROPE_CAPTURE_PATH = os.environ.get(
    "HUNYUAN_IMAGE_3_ROPE_CAPTURE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dev",
                 "reference_rope_padded_39x26.pt"))
needs_rope_capture = pytest.mark.skipif(
    not os.path.exists(ROPE_CAPTURE_PATH),
    reason="set HUNYUAN_IMAGE_3_ROPE_CAPTURE to dev/p1a_ref_rope.py's capture (reference_rope_padded_39x26.pt)")


@needs_rope_capture
def test_truncated_rope_matches_the_reference_table():
    """`build_rope_freqs` == the reference `build_2d_rope` cos/sin.

    The capture is a real img1 case (padded 39x26 tower run, base 10000, head_dim 128).
    The reference stores each row's 64 angles twice (`repeat(1, 2)` tiles the row, so the
    first half holds the unique angles) in the same y-even/x-odd order the pack builds,
    so the first halves compare directly, to fp32 rounding.
    """
    from comfy.ldm.hunyuan_image_3.model import build_rope_freqs
    capture = torch.load(ROPE_CAPTURE_PATH, weights_only=False, map_location="cpu")
    seq_len = capture["seq_len"]
    info = [(slice(*capture["vae"]), (76, 52)), (slice(*capture["vit"]), (39, 26))]
    freqs = build_rope_freqs(seq_len, 128, info, 10000.0)
    pack_cos = freqs[0, 0, :, :, 0, 0]
    pack_sin = freqs[0, 0, :, :, 1, 0]
    ref_cos, ref_sin = capture["cos"][:, :64], capture["sin"][:, :64]
    assert pack_cos.shape == ref_cos.shape == (seq_len, 64)
    assert torch.allclose(pack_cos, ref_cos, atol=1e-5)
    assert torch.allclose(pack_sin, ref_sin, atol=1e-5)
