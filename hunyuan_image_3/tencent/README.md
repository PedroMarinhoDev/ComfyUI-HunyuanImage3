# Tencent's files

`config.json` and `tokenizer.json` exactly as Tencent publishes them, shipped here so the model files are
all a user downloads. They are Tencent's, under the Tencent Hunyuan Community License
([`LICENSE-TENCENT-HUNYUAN`](../../LICENSE-TENCENT-HUNYUAN), [`NOTICE`](../../NOTICE)), not the GPL.

- `config.json` — from `tencent/HunyuanImage-3.0-Instruct-Distil`. The three checkpoints' configs differ only
  in `cfg_distilled`/`use_meanflow` (and two metadata fields); the loader sets those from the checkpoint it
  detects, so this one serves all three.
- `tokenizer.json` — from `tencent/HunyuanImage-3.0-Instruct` (byte-identical in the Instruct-Distil's repo). The base's
  own tokenizer is the same vocabulary minus 25 later tokens (`<timestep_r>`, `<img_ratio_33..36>`,
  `<relation_*>`), with every shared token at the same id; the base's sequences use none of those 25.
