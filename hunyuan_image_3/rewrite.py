"""The prompt-rewriting stage: the model writes a better prompt (optionally reasoning first) before the image.

Everything the stage needs beyond the diffusion model lives here: the text head (`lm_head` + `ln_f`),
which only predicts text and so travels in its own ~1 GiB file rather than in every checkpoint; the
choice of that file; and the autoregressive decode. The encoders call `run_rewrite` with the request
they are about to build, conditioning images included, and put the result in the image sequence —
the image is conditioned on the model's whole CoT text, not only on the rewritten prompt, which is why
the stage runs inside the encoder rather than as a node that returns a string.

Ported from the reference's `generate_image` (modeling:3280): bot_task `recaption` asks for the
rewritten prompt directly, `think_recaption` writes the analysis first and hands over to `<recaption>`
at `</think>` (the reference's `stage_transitions`).
"""
import logging
import os
import time
from dataclasses import dataclass

import torch

import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.utils
import folder_paths

from .loader import weight_fingerprint
from .model import build_rope_freqs
from .pipeline import build_attention_mask, build_input_embeddings
from .tokenizer import build_text_sequence

REWRITE_ONLY = "rewrite only"
THINK_AND_REWRITE = "think + rewrite"
AUTO = "auto"

HEAD_KEYS = ("lm_head.weight", "model.ln_f.weight")

# `ln_f`'s fingerprint (`weight_fingerprint`) in each checkpoint's head. The Instruct's and the
# Instruct-Distil's heads are byte-identical files, so either serves both; the base's is its own.
HEAD_FINGERPRINTS = {
    "ff87973199632c7f": ("instruct_distil", "instruct"),
    "d6935d2dc429a161": ("base",),
}

# where `auto` looks, in order, for each model: the file named after the checkpoint first
HEAD_FILES = {
    "instruct_distil": ("hunyuan_image_3_instruct_distil_cot_head.safetensors",
                        "hunyuan_image_3_instruct_cot_head.safetensors"),
    "instruct": ("hunyuan_image_3_instruct_cot_head.safetensors",
                 "hunyuan_image_3_instruct_distil_cot_head.safetensors"),
    "base": ("hunyuan_image_3_base_cot_head.safetensors",),
}


@dataclass(frozen=True)
class RewriteSettings:
    """What the Prompt Rewriting node hands the encoders."""
    mode: str = REWRITE_ONLY
    cot_head: str = AUTO
    max_new_tokens: int = 2048
    do_sample: bool = True
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 1024
    repetition_penalty: float = 1.0
    seed: int = 0


class CotHead(torch.nn.Module):
    """`ln_f` then `lm_head`: hidden state -> next-token logits."""

    def __init__(self, hidden_size, vocab_size, eps, dtype=torch.bfloat16):
        super().__init__()
        operations = comfy.ops.manual_cast
        self.ln_f = operations.RMSNorm(hidden_size, eps=eps, dtype=dtype, device="meta")
        self.lm_head = operations.Linear(hidden_size, vocab_size, bias=False, dtype=dtype, device="meta")

    def forward(self, hidden):
        return self.lm_head(self.ln_f(hidden))


# path -> ModelPatcher around a CotHead: the head is read once per file, and ComfyUI moves it on and off
# the GPU like any other model
_HEADS = {}


def head_candidates():
    """Head files offered by the node: anything in diffusion_models that looks like one."""
    return [name for name in folder_paths.get_filename_list("diffusion_models") if "cot_head" in name]


def resolve_head(choice, model_type):
    """The head file to use: the named one, or for `auto` the one matching `model_type`."""
    if choice not in (None, "", AUTO):
        return folder_paths.get_full_path_or_raise("diffusion_models", choice)
    available = folder_paths.get_filename_list("diffusion_models")
    for name in HEAD_FILES[model_type]:
        if name in available:
            return folder_paths.get_full_path_or_raise("diffusion_models", name)
    raise FileNotFoundError(
        f"prompt rewriting needs the {model_type} text head: put {HEAD_FILES[model_type][0]} in "
        f"models/diffusion_models (it is in the same HuggingFace repo as the model), or pick a head "
        f"file on the Prompt Rewriting node.")


def load_head(path, model_type, config):
    """The head at `path` as a ModelPatcher, loaded once and reused. Warns when its weights say it
    belongs to a different checkpoint than the one it is paired with."""
    if path in _HEADS:
        return _HEADS[path]
    state_dict = comfy.utils.load_torch_file(path)
    absent = [key for key in HEAD_KEYS if key not in state_dict]
    extra = [key for key in state_dict if key not in HEAD_KEYS]
    if absent or extra:
        raise ValueError(f"{os.path.basename(path)} is not a text head: missing {absent}, unexpected {extra[:4]}")
    owners = HEAD_FINGERPRINTS.get(weight_fingerprint(state_dict["model.ln_f.weight"]))
    if owners is not None and model_type not in owners:
        logging.warning("HunyuanImage3: %s is the %s head but the model is %s; the rewrite will be "
                        "nonsense", os.path.basename(path), owners[0], model_type)
    head = CotHead(config.hidden_size, config.vocab_size, config.rms_norm_eps)
    head.ln_f.load_state_dict({"weight": state_dict["model.ln_f.weight"]}, assign=True)
    head.lm_head.load_state_dict({"weight": state_dict["lm_head.weight"]}, assign=True)
    patcher = comfy.model_patcher.ModelPatcher(
        head, load_device=comfy.model_management.get_torch_device(),
        offload_device=comfy.model_management.unet_offload_device())
    _HEADS[path] = patcher
    return patcher


def _next_token(logits, history, do_sample, temperature, top_p, top_k, repetition_penalty, generator=None):
    """One token from one step's logits, the way HF's generation does it.

    The repetition penalty is applied whether or not sampling is on, which is HF's behaviour and why it
    can change a greedy run. Everything after that is skipped when `do_sample` is false.
    """
    scores = logits.float()
    if repetition_penalty != 1.0 and history:
        seen = torch.tensor(sorted(set(history)), device=scores.device, dtype=torch.long)
        values = scores[seen]
        scores[seen] = torch.where(values > 0, values / repetition_penalty, values * repetition_penalty)
    if not do_sample:
        return int(scores.argmax())
    scores = scores / max(temperature, 1e-5)
    if top_k:
        cutoff = torch.topk(scores, min(top_k, scores.shape[-1]))[0][-1]
        scores = scores.masked_fill(scores < cutoff, float("-inf"))
    if top_p < 1.0:
        ordered, order = torch.sort(scores, descending=True)
        probs = torch.softmax(ordered, dim=-1)
        # keep the token that crosses the threshold, as HF's shifted cumulative does
        remove = probs.cumsum(dim=-1) - probs > top_p
        ordered = ordered.masked_fill(remove, float("-inf"))
        scores = torch.empty_like(scores).scatter_(0, order, ordered)
    return int(torch.multinomial(torch.softmax(scores, dim=-1), 1, generator=generator))


def generate_text(model, head, sequence, *, stop_tokens, transitions=None, max_new_tokens=2048,
                  temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0, do_sample=False,
                  seed=None, cond_latent=None, cond_vit=None, device=None, dtype=None, verbose=False):
    """Autoregressive decode from `sequence` (a `build_text_sequence` result), with a per-call KV cache.

    `model` is the `HunyuanImage3` nn.Module and `head` the `CotHead`. The prefill embeds the
    conditioning images exactly as the image stage does (clean `t = 0`, bidirectional inside each
    image), so the text is written looking at them. One `[keys, values]` pair per layer is created
    here and dropped when this returns — the repo's rule for temporary caches.

    `transitions` mirrors the reference's `stage_transitions` (distil_modeling:3274): a dict from a stop
    token to the tokens forced in after it, after which generation continues. That is how `</think>`
    hands over to `<recaption>`.

    Returns `{"tokens": [...]}`, the generated tokens with transition tokens included.
    """
    config = model.config
    device = device if device is not None else sequence["ids"].device
    dtype = dtype if dtype is not None else model.dtype
    ids = sequence["ids"].to(device)
    prompt_length = ids.shape[0]
    blocks = sequence.get("cond_blocks", [])
    rope_image_info = []
    for block in blocks:
        rope_image_info += [(block["vae_slice"], (block["token_height"], block["token_width"])),
                            (block["vit_slice"], (block["patch_height"], block["patch_width"]))]
    cache = [[None, None] for _ in range(config.num_hidden_layers)]
    freqs = build_rope_freqs(prompt_length + max_new_tokens, config.attention_head_dim, rope_image_info,
                             config.rope_theta, device=device)
    generator = None
    if do_sample and seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    embeds = build_input_embeddings(model, dict(sequence, ids=ids), None, None, None, None,
                                    cond_latent=cond_latent, cond_vit=cond_vit)
    mask = build_attention_mask({"full_attention_slices": [block["joint_slice"] for block in blocks]},
                                prompt_length, dtype, device)
    hidden = model.model(embeds, freqs[:, :, :prompt_length], mask, cache)
    logits = head(hidden[:, -1])[0]

    def feed(token, position):
        step = model.model.wte(torch.tensor([[token]], device=device))
        return head(model.model(step, freqs[:, :, position:position + 1], None, cache)[:, -1])[0]

    pending = dict(transitions or {})
    generated = []
    position = prompt_length
    start = time.time()
    while len(generated) < max_new_tokens:
        # a long autoregressive stage has to honour ComfyUI's cancel: without this the queue's
        # `interrupt` is only noticed by the sampler, which runs later
        comfy.model_management.throw_exception_if_processing_interrupted()
        token = _next_token(logits, generated, do_sample, temperature, top_p, top_k, repetition_penalty, generator)
        generated.append(token)
        if token in stop_tokens and token not in pending:
            break
        logits = feed(token, position)
        position += 1
        # the reference's `_StageTransitionLogitsProcessor`: the stop token is fed like any other, and
        # the transition's tokens are then forced in after it, once per transition
        for extra in pending.pop(token, ()):
            generated.append(extra)
            logits = feed(extra, position)
            position += 1
        if verbose and len(generated) % 8 == 0:
            elapsed = time.time() - start
            print(f"    rewriting: {len(generated)} tokens, {elapsed:.0f}s, "
                  f"{elapsed / len(generated):.2f}s/token", flush=True)
    if verbose:
        elapsed = time.time() - start
        print(f"  rewriting: {len(generated)} tokens in {elapsed:.0f}s "
              f"({elapsed / max(len(generated), 1):.2f}s/token)", flush=True)
    return {"tokens": generated}


def recaption_from_ids(tokenizer, generated, recaption, end_of_recaption):
    """The rewritten prompt out of the generated ids, sliced by token id.

    The delimiters have to be ids: `decode([<special>])` returns `""` by default because the tokenizer
    skips special tokens, so splitting the decoded text on them finds nothing and looks like the model
    produced no rewrite. Returns `""` when the stage never reached `</recaption>` — a truncated budget,
    which the caller warns about rather than passing off as an empty rewrite.
    """
    if recaption not in generated or end_of_recaption not in generated:
        return ""
    opening = generated.index(recaption) + 1
    if end_of_recaption not in generated[opening:]:
        return ""
    return tokenizer.decode(generated[opening:generated.index(end_of_recaption, opening)],
                            skip_special_tokens=False).strip()


def run_rewrite(patcher, settings, prompt, system_prompt, cond_images=(), cond_latent=None, cond_vit=None):
    """Run the stage for one request. Returns `(cot_text, rewritten_prompt)`.

    `cot_text` is what the image sequence carries — the opening tag, the analysis if any, and the
    rewrite — and `rewritten_prompt` is the rewrite alone, for display.
    """
    diffusion_model = patcher.model.diffusion_model
    config = diffusion_model.config
    tokenizer = patcher.model.tokenizer
    think_first = settings.mode == THINK_AND_REWRITE
    head = load_head(resolve_head(settings.cot_head, config.model_type), config.model_type, config)

    think = tokenizer.token_to_id("<think>")
    recaption = tokenizer.token_to_id("<recaption>")
    end_of_think = tokenizer.token_to_id("</think>")
    end_of_recaption = tokenizer.token_to_id("</recaption>")
    sequence = build_text_sequence(tokenizer, prompt, system_prompt, "think" if think_first else "recaption",
                                   base_size=config.image_base_size,
                                   max_position_embeddings=config.max_position_embeddings,
                                   sequence_template=config.sequence_template, cond_images=cond_images)

    # The stage drives the model directly, outside the sampler, so it does what the sampler would: arm
    # the patcher for streaming (`samplers.py:258`), load it with the head beside it so the memory
    # manager accounts for both, and make the model's GPU the current device — otherwise a model moved
    # to another GPU runs its W4A8 kernels against the default one and fails with "Can't export tensors
    # on a different CUDA device index".
    patcher.prepare_state(torch.tensor([0.0]), {})
    comfy.model_management.load_models_gpu([patcher, head])
    device = patcher.load_device
    with comfy.model_management.cuda_device_context(device):
        result = generate_text(
            diffusion_model, head.model, sequence,
            # the reference's final stop for both tasks; `</think>` is a transition, not a stop
            stop_tokens=[end_of_recaption],
            transitions={end_of_think: [recaption]} if think_first else None,
            max_new_tokens=settings.max_new_tokens, do_sample=settings.do_sample,
            temperature=settings.temperature, top_p=settings.top_p, top_k=settings.top_k,
            repetition_penalty=settings.repetition_penalty, seed=settings.seed,
            cond_latent=cond_latent, cond_vit=cond_vit,
            device=device, dtype=diffusion_model.dtype, verbose=True)

    opening = think if think_first else recaption
    cot_text = tokenizer.decode([opening] + result["tokens"], skip_special_tokens=False)
    # the whole stage's text, reasoning included, which the encoder's output does not carry
    logging.info("HunyuanImage3: prompt rewriting wrote %d tokens:\n%s", len(result["tokens"]), cot_text)
    # rewrite-only starts inside `<recaption>`, so its opening tag is part of the prompt, not the output
    generated = result["tokens"] if think_first else [recaption] + result["tokens"]
    if recaption not in generated:
        # the image then gets the truncated analysis instead of the rewrite, which changes the render
        logging.warning("HunyuanImage3: the rewriting budget (%d tokens) ran out before </recaption>: the "
                        "image is conditioned on a partial analysis. Raise max_new_tokens, or use "
                        "'rewrite only', which skips the analysis.", settings.max_new_tokens)
    elif end_of_recaption not in generated:
        logging.warning("HunyuanImage3: the rewriting budget (%d tokens) ran out inside the rewrite, so "
                        "no rewritten prompt was produced. Raise max_new_tokens.", settings.max_new_tokens)
    return cot_text, recaption_from_ids(tokenizer, generated, recaption, end_of_recaption)
