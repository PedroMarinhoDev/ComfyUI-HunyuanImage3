"""The node pack's structure: model detection, the separate text head, prompt rewriting, and the
multi-image and negative sequences the encoders build."""
import os
from types import SimpleNamespace

import pytest
import torch

from comfy.ldm.hunyuan_image_3.loader import FINGERPRINT_KEY, MODEL_FINGERPRINTS, detect_model_type, weight_fingerprint
from comfy.ldm.hunyuan_image_3.system_prompt import SYSTEM_PROMPTS, UNIFIED_SYSTEM_PROMPT_EN
from comfy.ldm.hunyuan_image_3.tokenizer import build_sequence, build_text_sequence

TOKENIZER_PATH = os.environ.get("HUNYUAN_IMAGE_3_TOKENIZER", "")
MULTI_CAPTURE_PATH = os.environ.get("HUNYUAN_IMAGE_3_MULTI_CAPTURE", "")
MODELS_DIR = os.environ.get("HUNYUAN_IMAGE_3_MODELS", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(os.path.dirname(os.path.abspath(__file__)))))),
    "models", "diffusion_models"))
needs_tokenizer = pytest.mark.skipif(
    not os.path.exists(TOKENIZER_PATH), reason="set HUNYUAN_IMAGE_3_TOKENIZER to the repo's tokenizer.json")
needs_multi_capture = pytest.mark.skipif(
    not os.path.exists(MULTI_CAPTURE_PATH),
    reason="set HUNYUAN_IMAGE_3_MULTI_CAPTURE to dev/step12_multi_image_sequence.py's capture")

PLAIN = {"cfg_distilled": False, "use_meanflow": False}
DISTIL = {"cfg_distilled": True, "use_meanflow": True}


@pytest.fixture(scope="module")
def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(TOKENIZER_PATH)


# ------------------------------------------------------------------------------------ model detection

def test_undistilled_weights_fall_back_to_the_file_name_then_to_instruct():
    unknown = {FINGERPRINT_KEY: torch.ones(8, dtype=torch.bfloat16)}
    assert detect_model_type(unknown, "/x/my_base_finetune.safetensors") == "base"
    assert detect_model_type(unknown, "/x/hunyuan_image_3_instruct_w4a8.safetensors") == "instruct"
    assert detect_model_type(unknown, "/x/something.safetensors") == "instruct"


def test_the_distilled_embedders_decide_the_instruct_distil_whatever_the_name():
    # running the distilled contract without its embedders, or the plain one with them, is a broken
    # render rather than a choice, so the structure outranks the file name
    distilled = {FINGERPRINT_KEY: torch.ones(8, dtype=torch.bfloat16), "guidance_emb.mlp.0.weight": torch.zeros(1)}
    assert detect_model_type(distilled, "/x/hunyuan_image_3_base_w4a8.safetensors") == "instruct_distil"


def test_a_known_fingerprint_outranks_a_misleading_name():
    tensor = torch.arange(16, dtype=torch.bfloat16)
    fingerprint = weight_fingerprint(tensor)
    MODEL_FINGERPRINTS[fingerprint] = "base"
    try:
        assert detect_model_type({FINGERPRINT_KEY: tensor}, "/x/instruct.safetensors") == "base"
    finally:
        del MODEL_FINGERPRINTS[fingerprint]


@pytest.mark.parametrize("model_type,stem", [("instruct_distil", "instruct_distil"), ("instruct", "instruct"), ("base", "base")])
def test_the_published_checkpoints_are_recognised_by_their_weights(model_type, stem):
    path = os.path.join(MODELS_DIR, f"hunyuan_image_3_{stem}_w4a8.safetensors")
    if not os.path.exists(path):
        pytest.skip(f"{os.path.basename(path)} not present")
    from safetensors import safe_open
    with safe_open(path, "pt") as handle:
        state_dict = {FINGERPRINT_KEY: handle.get_tensor(FINGERPRINT_KEY)}
        if "guidance_emb.mlp.0.weight" in handle.keys():
            state_dict["guidance_emb.mlp.0.weight"] = torch.zeros(1)
    # a misleading name, so only the weights can be what answers
    assert detect_model_type(state_dict, "/x/renamed.safetensors") == model_type


def test_each_checkpoint_gets_its_own_system_prompt():
    assert SYSTEM_PROMPTS["instruct_distil"] == SYSTEM_PROMPTS["instruct"] == UNIFIED_SYSTEM_PROMPT_EN
    # the base's generation_config.json selects `use_system_prompt: "None"`
    assert SYSTEM_PROMPTS["base"] == ""


# ------------------------------------------------------------------------------------ text head

def test_the_head_loads_as_its_own_module(tmp_path):
    from safetensors.torch import save_file

    from comfy.ldm.hunyuan_image_3.rewrite import load_head

    config = SimpleNamespace(hidden_size=8, vocab_size=32, rms_norm_eps=1e-6)
    weights = {"lm_head.weight": torch.randn(32, 8).to(torch.bfloat16),
               "model.ln_f.weight": torch.rand(8).to(torch.bfloat16)}
    path = str(tmp_path / "head.safetensors")
    save_file(weights, path)
    head = load_head(path, "instruct", config).model

    hidden = torch.randn(2, 8).to(torch.bfloat16)
    normed = hidden.float() * torch.rsqrt(hidden.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    expected = (normed * weights["model.ln_f.weight"].float()) @ weights["lm_head.weight"].float().t()
    torch.testing.assert_close(head(hidden).float(), expected, atol=0.1, rtol=0.05)

    bad = str(tmp_path / "bad.safetensors")
    save_file({"model.layers.0.weight": torch.zeros(1)}, bad)
    with pytest.raises(ValueError, match="not a text head"):
        load_head(bad, "instruct", config)


def test_auto_picks_the_head_matching_the_model(monkeypatch):
    import folder_paths

    from comfy.ldm.hunyuan_image_3 import rewrite

    files = ["hunyuan_image_3_instruct_cot_head.safetensors", "hunyuan_image_3_base_cot_head.safetensors"]
    monkeypatch.setattr(folder_paths, "get_filename_list", lambda folder: files)
    monkeypatch.setattr(folder_paths, "get_full_path_or_raise", lambda folder, name: f"/models/{name}")
    assert rewrite.resolve_head("auto", "base").endswith("base_cot_head.safetensors")
    assert rewrite.resolve_head("auto", "instruct").endswith("instruct_cot_head.safetensors")
    # the Instruct-Distil's head is byte-identical to the Instruct's, so that file serves it too
    assert rewrite.resolve_head("auto", "instruct_distil").endswith("instruct_cot_head.safetensors")
    assert rewrite.resolve_head(files[1], "instruct").endswith(files[1])
    monkeypatch.setattr(folder_paths, "get_filename_list", lambda folder: [])
    with pytest.raises(FileNotFoundError, match="hunyuan_image_3_base_cot_head"):
        rewrite.resolve_head("auto", "base")


# ------------------------------------------------------------------------------------ text generation

class _ScriptedTransformer:
    """Records which tokens were fed after the prefill; the scripted head decides what comes next."""

    def __init__(self):
        self.fed = []

    def wte(self, ids):
        if ids.shape[-1] == 1:
            self.fed.append(int(ids.flatten()[0]))
        return torch.zeros(1, ids.shape[-1], 4)

    def __call__(self, embeds, freqs, mask=None, cache=None):
        return embeds


def test_a_transition_feeds_the_stop_token_before_the_forced_tokens():
    """The reference's `_StageTransitionLogitsProcessor` lets `</think>` into the context and then forces
    `<recaption>`; the rewrite is conditioned on both, so dropping the first changes it."""
    from comfy.ldm.hunyuan_image_3.rewrite import generate_text

    end_of_think, recaption, end_of_recaption = 7, 6, 8
    script = iter([5, end_of_think, 0, 9, end_of_recaption])

    def head(hidden):
        logits = torch.full((1, 16), -1e4)
        logits[0, next(script)] = 0
        return logits

    transformer = _ScriptedTransformer()
    model = SimpleNamespace(model=transformer, dtype=torch.float32,
                            config=SimpleNamespace(num_hidden_layers=1, attention_head_dim=8, rope_theta=10000.0))
    result = generate_text(model, head, {"ids": torch.tensor([1, 2, 3])}, stop_tokens=[end_of_recaption],
                           transitions={end_of_think: [recaption]}, max_new_tokens=16, device="cpu",
                           dtype=torch.float32)
    assert result["tokens"] == [5, end_of_think, recaption, 9, end_of_recaption]
    assert transformer.fed == [5, end_of_think, recaption, 9]


# ------------------------------------------------------------------------------------ sequences

@needs_tokenizer
@needs_multi_capture
def test_multi_image_and_negative_sequences_match_the_reference(tokenizer):
    """1-3 images, the rewriting stage over images, and CFG's negative, id for id against the reference's
    own image processor and tokenizer (`dev/step12_multi_image_sequence.py`)."""
    capture = torch.load(MULTI_CAPTURE_PATH, weights_only=False)
    prompt = capture["prompt"]
    cot = {"image_2_cot": "<recaption>a cat on a sofa</recaption>",
           "t2i_cfg_instruct_cot": "<think>plan it</think><recaption>a cat on a sofa</recaption>",
           "image_1_cfg_instruct_cot": "<recaption>a cat on a sofa</recaption>"}
    checked = 0
    for name, case in capture.items():
        if not isinstance(case, dict):
            continue
        # the reference sizes each image to its table row; the tower always gives 1024 patches here
        cond_images = [(tuple(size), (32, 32)) for size in case["cond_vae_sizes"]]
        template = "pretrain" if "base" in name else "instruct"
        contract = PLAIN if "cfg" in name else DISTIL
        for row in range(case["tokens"].shape[0]):
            if name.startswith("text_"):
                ids = build_text_sequence(tokenizer, prompt, case["system_prompt"], name.split("_")[1],
                                          cond_images=cond_images, sequence_template=template)["ids"]
            else:
                ids = build_sequence(tokenizer, prompt, "1024x1024", case["system_prompt"], cot_text=cot.get(name),
                                     cond_images=cond_images, sequence_template=template, uncond=row == 1,
                                     **contract)["ids"]
            assert ids.tolist() == case["tokens"][row].tolist(), f"{name}[{row}]"
            checked += 1
    assert checked == 20


@needs_tokenizer
def test_the_negative_blanks_only_the_text(tokenizer):
    kwargs = dict(cond_images=[((832, 1216), (32, 32))], cot_text="<recaption>a cat</recaption>", **PLAIN)
    positive = build_sequence(tokenizer, "a cat on a sofa", "1024x1024", UNIFIED_SYSTEM_PROMPT_EN, **kwargs)
    negative = build_sequence(tokenizer, "a cat on a sofa", "1024x1024", UNIFIED_SYSTEM_PROMPT_EN, uncond=True, **kwargs)
    cfg = tokenizer.token_to_id("<cfg>")
    changed = (positive["ids"] != negative["ids"]).nonzero().flatten()
    assert positive["ids"].shape == negative["ids"].shape
    assert (negative["ids"][changed] == cfg).all()
    prompt_tokens = len(tokenizer.encode("a cat on a sofa", add_special_tokens=False).ids)
    assert len(changed) == prompt_tokens + len(tokenizer.encode("a cat", add_special_tokens=False).ids)
    assert positive["cond_blocks"] == negative["cond_blocks"]


@needs_tokenizer
def test_the_model_derives_every_image_block_from_the_ids(tokenizer):
    """Two conditioning images plus the generated block: the geometry the model reads off the ids is the
    one the builder wrote, so each embedder lands where the sequence says it is."""
    from comfy.ldm.hunyuan_image_3.model import _sequence_from_ids

    config = SimpleNamespace(image_token_id=tokenizer.token_to_id("<img>"), **DISTIL)
    built = build_sequence(tokenizer, "combine them", "768x1024", UNIFIED_SYSTEM_PROMPT_EN,
                           cond_images=[((832, 1216), (26, 38)), ((1024, 1024), (32, 32))], **DISTIL)
    latents = torch.zeros(1, 32, built["token_height"], built["token_width"])
    cond_latents = [torch.zeros(1, 32, block["token_height"], block["token_width"]) for block in built["cond_blocks"]]
    grids = [torch.tensor([block["patch_height"], block["patch_width"]]) for block in built["cond_blocks"]]
    derived = _sequence_from_ids(built["ids"], latents, config, cond_latent=cond_latents, cond_patch_grid=grids)

    for key in ("image_slice", "timestep_position", "guidance_position", "timestep_r_position"):
        assert derived[key] == built[key], key
    for derived_block, built_block in zip(derived["cond_blocks"], built["cond_blocks"]):
        for key in ("vae_slice", "vit_slice", "timestep_position"):
            assert derived_block[key] == built_block[key], key
    assert derived["full_attention_slices"] == [block["joint_slice"] for block in built["cond_blocks"]] + [built["image_slice"]]
    with pytest.raises(ValueError, match="did not arrive"):
        _sequence_from_ids(built["ids"], latents, config, cond_latent=cond_latents[:1], cond_patch_grid=grids[:1])


@needs_tokenizer
def test_the_base_uses_its_own_ratio_table(tokenizer):
    """The base's table and vocabulary stop at `<img_ratio_32>`; the four extra rows (1024x768 and kin)
    arrived with the Instruct. The base's tokenizer does not have `<img_ratio_33>` at all."""
    instruct = build_sequence(tokenizer, "a cat", "768x1024", "", **PLAIN)
    base = build_sequence(tokenizer, "a cat", "768x1024", "", sequence_template="pretrain", extra_rows=False, **PLAIN)
    assert instruct["ratio_index"] == 33          # the extra 1024-high, 768-wide row
    assert base["ratio_index"] <= 32
    assert (base["token_height"], base["token_width"]) == (instruct["token_height"], instruct["token_width"])
    square = build_sequence(tokenizer, "a cat", "1024x1024", "", extra_rows=False, **PLAIN)
    assert square["ratio_index"] == 16


def test_the_tencent_files_ship_with_the_pack():
    from comfy.ldm.hunyuan_image_3.loader import CONFIG_PATH, TOKENIZER_PATH
    assert os.path.getsize(TOKENIZER_PATH) == 25028750
    assert os.path.exists(CONFIG_PATH)


# ------------------------------------------------------------------------------------ nodes

def _inputs(node):
    return {entry.id: entry for entry in node.define_schema().inputs}


def test_the_node_layout():
    from comfy_extras.nodes_hunyuan_image_3 import (HunyuanImage3ImageEncode, HunyuanImage3ModelLoader,
                                                    HunyuanImage3PromptRewriting, HunyuanImage3TextEncode)

    assert list(_inputs(HunyuanImage3ModelLoader)) == ["model"]
    text = _inputs(HunyuanImage3TextEncode)
    assert not {"image", "vae", "clip_vision", "system_prompt", "rewrite_prompt"} & set(text)
    for node in (HunyuanImage3TextEncode, HunyuanImage3ImageEncode):
        inputs = _inputs(node)
        assert inputs["prompt_rewriting"].optional and inputs["custom_system_prompt"].optional
        assert [output.display_name for output in node.define_schema().outputs] == \
            ["positive", "negative", "rewritten_prompt"]
    images = _inputs(HunyuanImage3ImageEncode)["images"]
    assert images.template.names == ["image_1", "image_2", "image_3"] and images.template.min == 1
    assert "cot_head" in _inputs(HunyuanImage3PromptRewriting)


def test_conditioning_images_are_sized_like_the_reference():
    from comfy_extras.nodes_hunyuan_image_3 import _resize_and_crop

    # 1200x800 -> the 1216x832 row (the reference's own processor, captured); 640x960 is scaled up
    assert tuple(_resize_and_crop(torch.rand(1, 800, 1200, 3), 1216, 832).shape) == (1, 832, 1216, 3)
    assert tuple(_resize_and_crop(torch.rand(1, 960, 640, 3), 832, 1216).shape) == (1, 1216, 832, 3)
