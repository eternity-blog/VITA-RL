# VITA-1.5 Architecture and Code Walkthrough

A guide to understanding this codebase: what the model is, how the code is
organized, and what actually happens at each step of inference and training.

Everything here describes **upstream VITA-1.5** unless a section is marked as
specific to this fork. Line references are to this repository at the time of
writing; they may drift as the code changes.

> Language: **English** | [中文](./ARCHITECTURE_zh-CN.md)
>
> Companion documents: [REPRODUCE.md](./REPRODUCE.md) is the operational log
> (how to install, run, and what broke). This file is the conceptual map.

## Contents

- [1. What the model is](#1-what-the-model-is)
- [2. Repository layout](#2-repository-layout)
- [3. The central idea: negative-index placeholder tokens](#3-the-central-idea-negative-index-placeholder-tokens)
- [4. The three encoders](#4-the-three-encoders)
- [5. `prepare_inputs_labels_for_multimodal`: the heart of the model](#5-prepare_inputs_labels_for_multimodal-the-heart-of-the-model)
- [6. The state tokens (rejection mechanism)](#6-the-state-tokens-rejection-mechanism)
- [7. Inference walkthrough](#7-inference-walkthrough)
- [8. Training walkthrough](#8-training-walkthrough)
- [9. The data pipeline](#9-the-data-pipeline)
- [10. The real-time duplex demo](#10-the-real-time-duplex-demo)
- [11. Model variants](#11-model-variants)
- [12. Known defects and rough edges](#12-known-defects-and-rough-edges)
- [13. Where RL would attach](#13-where-rl-would-attach)
- [14. The RL stack (DPO and GRPO)](#14-the-rl-stack-dpo-and-grpo)

---

## 1. What the model is

VITA-1.5 is an **omni-modal LLM**: one language model that takes images, video,
audio and text, and emits text — plus, through a separate decoder, speech.

The design is the LLaVA lineage, extended to audio:

```
image/video ──> InternViT-300M ──> MLP projector ──┐
                                                    ├──> Qwen2.5-7B ──> text
audio ────────> whale encoder ──> CNN adapter ─────┘         │
                                                             └──> TTS decoder ──> speech
```

The load-bearing idea is that **every modality is converted into vectors that
live in the LLM's embedding space**, then spliced into the token embedding
sequence. The LLM itself is unmodified Qwen2.5 — it never learns a new input
type, it just receives embeddings that happen to come from an image or a
waveform rather than from a token lookup.

What distinguishes VITA-1.5 from a vanilla VLM:

| | |
|---|---|
| **End-to-end speech out** | No external TTS. The speech decoder consumes the LLM's hidden states directly, which is where the latency win comes from (~4 s → ~1.5 s). |
| **Negative-sample training** | The model is explicitly trained to *decline* to answer noisy or non-speech audio, which is what makes always-on microphone interaction viable. |
| **Progressive training** | Vision and audio are aligned in separate stages so that adding speech barely degrades vision (71.3 → 70.8 average). |

## 2. Repository layout

```
vita/
├── constants.py                 # the placeholder token indices — read this first
├── conversation.py              # prompt templates (9 conv modes)
├── model/
│   ├── vita_arch.py             # ★ VITAMetaModel / VITAMetaForCausalLM: modality fusion
│   ├── builder.py               # load_pretrained_model() for inference
│   ├── language_model/
│   │   ├── vita_qwen2.py        # ★ the main model (VITA-1.5)
│   │   ├── vita_fo_qwen2.py     # full-duplex variant, + state predictor head
│   │   ├── vita_mixtral.py      # VITA-1.0 legacy
│   │   └── vita_nemo.py         # Mistral/Nemo variant
│   ├── multimodal_encoder/
│   │   ├── builder.py           # dispatches vision tower by name substring
│   │   ├── internvit/           # the vision tower VITA-1.5 actually uses
│   │   ├── clip/ eva_clip/ siglip/   # alternates
│   │   └── whale/               # ★ the audio encoder (self-developed)
│   ├── multimodal_projector/
│   │   └── builder.py           # mlp2x_gelu and friends
│   └── vita_tts/                # end-to-end speech synthesis
├── train/
│   ├── train.py                 # ★ single training entry point (SFT; also
│   │                            #   hosts the shared model construction that
│   │                            #   the RL entry points reuse)
│   ├── vita_trainer.py          # HF Trainer subclass
│   ├── dpo_{loss,data,trainer}.py    # offline DPO — added by this fork
│   ├── train_dpo.py             # DPO entry point
│   ├── grpo_{loss,data,trainer}.py   # GRPO — added by this fork
│   ├── train_grpo.py            # GRPO entry point
│   └── rewards.py               # pluggable reward registry (GRPO)
├── util/
│   ├── data_utils_video_audio_neg_patch.py   # ★ the active data pipeline
│   ├── data_utils_*.py          # six more variants, selected by editing imports
│   └── mm_utils.py              # tokenizer helpers
└── config/
    ├── dataset_config.py        # dataset paths (ships empty)
    └── __init__.py              # DataConfig registry

script/train/*.sh                # 18 upstream launch scripts, plus this fork's
                                 # smoke tests (SFT-LoRA, DPO, GRPO)
web_demo/                        # Flask + SocketIO real-time demo, vLLM-accelerated
videomme/                        # Video-MME benchmark
VLMEvalKit/                      # vendored copy of OpenCompass's eval kit
tools/                           # added by this fork: data generators, five
                                 # CPU test suites, config localisation
```

The four files marked ★ are where the real logic lives. `vita_arch.py` alone
accounts for most of what makes this model non-obvious.

Upstream ends at `vita_trainer.py`. Everything from `dpo_loss.py` down is
this fork's RL work, walked through in [§14](#14-the-rl-stack-dpo-and-grpo).

## 3. The central idea: negative-index placeholder tokens

From `vita/constants.py`:

```python
IGNORE_INDEX      = -100
IMAGE_TOKEN_INDEX = -200
AUDIO_TOKEN_INDEX = -500
```

These are **not real token IDs**. A tokenizer only ever produces non-negative
IDs, so negative values are unambiguous markers that survive inside an
`input_ids` tensor without colliding with anything.

The flow:

1. Text contains the literal strings `<image>` / `<audio>`.
2. `tokenizer_image_audio_token` (`vita/util/mm_utils.py:73`) splits on those
   strings and substitutes `-200` / `-500` in place of normal token IDs:

   ```python
   for chunk in re.split(r"(<audio>|<image>)", prompt):
       if chunk == "<audio>":   prompt_chunks.append([audio_token_index])
       elif chunk == "<image>": prompt_chunks.append([image_token_index])
       else:                    prompt_chunks.append(tokenizer(chunk).input_ids)
   ```

3. `prepare_inputs_labels_for_multimodal` finds those positions and **replaces
   each marker with the actual encoder output vectors**.

The consequence, and the thing to internalize: **the model is fed
`inputs_embeds`, not `input_ids`.** By the time the LLM runs, the placeholder
positions have become dense vectors and the original integer sequence no longer
exists. This is why `generate()` takes `inputs_embeds`, and it is the single
biggest complication for anything that needs to recompute log-probabilities
later — see [§13](#13-where-rl-would-attach).

## 4. The three encoders

### 4.1 Vision: InternViT-300M-448px

`vita/model/multimodal_encoder/internvit/internvit_encoder.py`

An image is first cut into tiles by `dynamic_preprocess`
(`data_utils_video_audio_neg_patch.py:1499`) — the InternVL scheme: pick the
aspect ratio from a candidate set that best matches the original, resize, and
slice into 448×448 tiles (up to `max_dynamic_patch`, default 12), optionally
appending a downscaled thumbnail of the whole image.

Each tile then goes through the ViT and a **pixel shuffle** that trades spatial
resolution for channels:

```
448×448 tile ─ViT/14─> 32×32 = 1024 patches × 1024 dim
             ─pixel_shuffle(0.5)─> 16×16 = 256 tokens × 4096 dim
```

Measured on this checkpoint: one tile → `(1, 256, 4096)`.

So **one tile costs 256 LLM tokens**, and a 12-tile image costs ~3072. This is
why `model_max_length` in the training scripts is 6200 rather than something
small, and why the real-time demo drops `max_dynamic_patch` to 1 — 256 tokens
instead of 3328 is the difference between interactive and sluggish.

`scale_pix_shuffle = 0.5` and `select_layer = -1` (last hidden layer, with the
CLS token dropped: `image_features[:, 1:]`).

### 4.2 Audio: whale

`vita/model/multimodal_encoder/whale/`

A self-developed encoder (~341M params), not Whisper. Structure:

```
waveform ─kaldi.fbank─> 80-dim mel frames @100fps
         ─GlobalCMVN─> normalized
         ─whaleEncoder─> encoder states      (Transformer + FSMN + DTC blocks)
         ─CNNAdapter─> LLM-dimension vectors
```

The subsampling is visible in `audioEncoderProcessor.process`
(`whale/init_model.py:35`):

```python
attn_mask = torch.ones(mat.shape[0])
attn_mask = attn_mask[2::2][2::2][0::2]   # three stride-2 steps => /8
```

Measured: a 3.54 s wav → 352 fbank frames → **44 LLM tokens**, i.e. about
**12.5 tokens/second** of audio. Cheap compared to images.

Note the config is not in this repo — it is loaded at runtime from the audio
encoder directory (`train.yaml`, `global_cmvn`, `final.pt`), fetched with
`get_file_from_repo`, so it resolves either a local path or a HuggingFace repo
ID (`multimodal_encoder/builder.py:46-49`).

The adapter (`adpter`, note upstream's spelling — `train.py:433` and
`vita_trainer.py:321` both string-match on it) is the only audio component
trained in stage 1.

### 4.3 Projector

`vita/model/multimodal_projector/builder.py:154`. VITA-1.5 uses
`mlp2x_gelu` = `Linear(mm_hidden, hidden) → GELU → Linear(hidden, hidden)`.
Several alternates exist (`spp`, `ldp`, `minigpt`, `vanilla`, `identity`) but
are unused by the shipped configs.

## 5. `prepare_inputs_labels_for_multimodal`: the heart of the model

`vita/model/vita_arch.py:333`. Everything else is plumbing; this function is
where the modalities actually merge. It is ~290 lines. The logic:

**Step 1 — early exit for cached decoding** (line 312). If `input_ids.shape[1] == 1`
we are generating token-by-token with a KV cache; there is nothing to splice, so
it just extends the attention mask and returns.

**Step 2 — encode** (line 334). All image tiles across the batch are concatenated
into one tensor for a single ViT forward, then split back per sample. Audio goes
through the whale encoder in one batched call.

**Step 3 — split each sequence at the markers** (line 418):

```python
image_audio_token_indices = [-1] + torch.where(
    (cur_input_ids == IMAGE_TOKEN_INDEX) | (cur_input_ids == AUDIO_TOKEN_INDEX)
)[0].tolist() + [cur_input_ids.shape[0]]
```

The sequence is cut into text runs between markers. Text runs are embedded
normally via `embed_tokens`; at each marker the corresponding encoder output is
inserted instead.

**Step 4 — labels are extended in lockstep** (line 452). Every inserted vision or
audio vector gets a label of `IGNORE_INDEX`:

```python
cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, ...))
```

This matters: **the model is never trained to predict image or audio tokens**,
only the text that follows them. One marker in `input_ids` becomes hundreds of
positions in `inputs_embeds`, and the label tensor has to grow identically or
the loss silently misaligns.

**Step 5 — pad and stack** (line 522). Sequences are padded to the batch max,
truncated to `tokenizer_model_max_length`, and `position_ids` / `attention_mask`
are rebuilt for the new lengths.

**Return** (line 602):

```python
return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
#      ^^^^ input_ids is deliberately None
```

Two subtleties worth knowing:

- **`position_ids` is usually `None` on return** (line 599). Unless
  `shared_v_pid_stride` is set, the caller lets the LLM derive positions itself.
- **`shared_v_pid_stride`** (line 628, `make_shared_position_ids`) is an optional
  scheme that gives several vision tokens the *same* position id, compressing
  the positional footprint of long visual spans. Off by default.

## 6. The state tokens (rejection mechanism)

This is VITA-1.5's mechanism for always-on microphone interaction, and it is
easy to miss because it is implemented as three Unicode characters.

In `preprocess_multimodal` (`data_utils_video_audio_neg_patch.py:128`), every
assistant reply gets a one-character prefix during training:

```python
if i == inserted_id:                          # ☞ ☟ ☜
    sentence["value"] = "☟" + sentence["value"]
elif sentence["from"] == "gpt":
    if "<audio>" in source[i - 1]["value"]:
        sentence["value"] = "☞" + sentence["value"]
    else:
        sentence["value"] = "☜" + sentence["value"]
```

| Token | Meaning |
|---|---|
| `☜` | reply to a **text** query |
| `☞` | reply to a **valid speech** query |
| `☟` | reply in the **negative** condition — noisy audio, background speech, anything not addressed to the assistant |

`inserted_id` in the JSON marks which assistant turn is the negative one.

The payoff is at inference. In the real-time server (`web_demo/server.py:312`):

```python
def judge_negative(text):
    is_negative = text.startswith('☟')
    return is_negative
```

The first generated character tells the server whether to speak the response or
stay silent — a rejection decision available after **one token**, without a
separate classifier. That is what makes an always-listening microphone
practical.

Directly observable: `asset/q1.wav` (clean speech) produces a reply prefixed
`☞`; `asset/q2.wav` (noisy) produces `☟`.

Just above this code sits the abandoned first attempt, commented out — the same
scheme using `<1>` / `<2>` / `<3>`. The switch to rare Unicode glyphs is
measurable in this checkpoint's tokenizer: `☜` / `☞` / `☟` are **one token
each** (ids 145789 / 144766 / 146164), while `<1>` / `<2>` / `<3>` are **three
tokens each** (`<`, digit, `>`). A single token is what makes the reject/accept
decision available after exactly one decoding step, and it cannot be produced
accidentally by ordinary text containing `<1>`.

## 7. Inference walkthrough

Following `video_audio_demo.py` end to end:

```
1. load_pretrained_model()                      builder.py:14
     ├── VITAQwen2ForCausalLM.from_pretrained()  loads the 7B + projector
     ├── vision_tower.load_model()               InternViT, cast to fp16
     └── audio_encoder                           whale, from train.yaml/final.pt

2. build the prompt
     ├── conv_templates["qwen2p5_instruct"]      conversation.py:326
     ├── system prompt chosen by modality        image / video / plain
     └── "<image>\n<audio>\n" placed in the user turn

3. tokenize
     └── tokenizer_image_audio_token()           mm_utils.py:73
         "…<image>…" -> [… , -200, …, -500, …]

4. preprocess inputs
     ├── dynamic_preprocess()                    image -> N tiles of 448²
     └── audio_processor.process()               wav -> fbank frames

5. model.generate(inputs, images, audios)        vita_qwen2.py:198
     └── prepare_inputs_labels_for_multimodal()  markers -> embeddings
         └── super().generate(inputs_embeds=…)   standard HF generation

6. decode
     └── tokenizer.batch_decode()                first char is ☜/☞/☟
```

Measured on one H800: text query 7.4 s, audio query 2.3 s (the audio prompt is
much shorter in tokens than a long text instruction).

## 8. Training walkthrough

### 8.1 Progressive three-stage recipe

The 18 scripts in `script/train/` are the cross product of {stage} × {backbone}
× {single-node, multi-node}. The output directory names give away the staging:

| Stage | Script prefix | Output dir | LR | What is unfrozen |
|---|---|---|---|---|
| 1a | `pretrain_mlp_*` | `llava-s1-pretrain_mlp_video` | 5e-4 | `mm_projector` only |
| 1b | `pretrain_audio_mlp_*` | `llava-s1-pretrain_audio_mlp` | 5e-4 | `audio_encoder.adpter` only |
| 2 | `finetune_*` | `llava-s2-pretrain_video` | 2e-5 | LLM + projector (+ vision tower on qwen) |
| 3 | `finetuneTask_*` | `llava-s3-finetune_task` | 2e-5 | LLM, audio fully frozen |
| 3-neg | `finetuneTaskNeg_*` | `llava-s3-finetune_task_neg` | 1e-5 | LLM + **audio adapter** |
| 3-fo | `finetuneTaskNeg_qwen_fo_*` | same | 1e-4 | audio prompt embeddings only |

The README's continual-training entry point is
`finetuneTaskNeg_qwen_nodes.sh` (4 nodes × 8 GPUs).

Common to all: DeepSpeed **ZeRO-3**, bf16, tf32, gradient checkpointing,
1 epoch, cosine schedule, `warmup_ratio=0.03`, `save_total_limit=1`.

### 8.2 The freeze switches

`train.py:377-412` is the mechanism behind the staging. Eight flags:

| Flag | Unfreezes |
|---|---|
| `--tune_mm_mlp_adapter` | `mm_projector` |
| `--tune_audio_mlp_adapter` | `audio_encoder.adpter` |
| `--audio_prompt_finetune` | `audio_encoder.prompt_embeddings` |
| `--audio_state_predictor_tuning` | `predictor_head` |
| `--freeze_audio_encoder` | (default True — the whale trunk is never trained here) |
| `--freeze_audio_encoder_adapter` | toggled per stage |
| `--unfreeze_vision_tower` | the ViT |
| `--freeze_mm_mlp_adapter` | inverse of the first |

**The first four each begin with a blanket `model.requires_grad_(False)`**, then
re-enable one submodule. So they are largely mutually exclusive — enabling two
means the second wipes the first's unfreezing. `audio_prompt_finetune` and
`audio_state_predictor_tuning` are the one pair designed to coexist.

### 8.3 What `VITATrainer` customizes

`vita/train/vita_trainer.py`, four overrides:

1. **`_get_train_sampler`** — optional `LengthGroupedSampler` that batches
   samples of similar length, using sign to separate modalities (positive length
   = multimodal, negative = text-only). Off in every shipped script.
2. **`create_optimizer`** — supports a separate `--mm_projector_lr`. Note
   line 190: the `mm_projector` clause is commented out, so despite the name the
   flag now only affects `vision_tower`.
3. **`_save_checkpoint` / `_save`** — when doing adapter-only training, saves
   just `mm_projector.bin` or `audio_adpter.bin` instead of a 16 GB full model.
   This is what makes stage-1 checkpoints cheap.
4. **`training_step`** — currently a pass-through; the debug branch is commented out.

### 8.4 Memory

Full-parameter 7B does not fit on one 80 GB GPU. AdamW holds two fp32 moments
per parameter (~56 GB) plus fp32 master weights (~28 GB); ZeRO-3 shards these
across ranks. A single-GPU run completes forward and backward and then OOMs
allocating optimizer state. **8 GPUs is the practical floor** for full-parameter
training. See [REPRODUCE.md](./REPRODUCE.md#memory-note).

## 9. The data pipeline

### 9.1 Sample format

```json
{
  "set": "sharegpt4",
  "id": "000000000164",
  "conversations": [
    {"from": "human", "value": "<image>\n<audio>\n"},
    {"from": "gpt",   "value": "This is a well-organized kitchen…"}
  ],
  "image": "coco/images/train2017/000000000164.jpg",
  "audio": ["output_wavs/f61cf238b7872b4903e1fc15dcb5a50c.wav"],
  "inserted_id": 3
}
```

- `set` keys into `FolderDict` to resolve the image directory.
- Audio resolves as `os.path.join(AudioFolder, "audio", file)` — note the
  hard-coded `audio` segment, so `AudioFolder` is the *parent* of `audio/`.
- `inserted_id` (optional) marks which assistant turn is the negative sample.
- `"gpt"` is a LLaVA-inherited label meaning "ground truth", not a model name.

### 9.2 Seven variants, chosen in source

`train.py:18-22` imports one of seven `data_utils_*` modules; the others are
commented out. **Switching the data pipeline means editing source, not passing a
flag.**

| Variant | Distinguishing feature |
|---|---|
| `neg_patch` | **active default** — dynamic tiling + negative samples |
| `neg_patch_fo` | adds `state_labels` (`-101`/`-102`) for the duplex predictor |
| `patch` | as `neg_patch`, minus `preprocess_mixtral_zh` |
| `patch_sf` | slow-fast video frame sampling |
| `neg_frameCat` | concatenates 5 frames channel-wise instead of tiling |
| `video_patch_audio`, `video_audio` | earlier iterations |

They are near-duplicates (~1400 lines each, ~9500 total). This is research
iteration debris rather than intentional design — check which one `train.py`
imports before editing anything.

### 9.3 Label masking

`preprocess_qwen2p5_instruct` (`data_utils_video_audio_neg_patch.py:526`) splits
the rendered conversation on the role separators and masks every instruction
span with `IGNORE_INDEX`, leaving only assistant replies supervised.

It ends with a consistency check worth knowing about:

```python
if cur_len != total_len:
    target[:] = IGNORE_INDEX
    print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored)")
```

On any mismatch the sample is **silently dropped from the loss** — the whole
label row becomes `IGNORE_INDEX`. It prints a warning, but with `logging_steps`
noise it is easy to miss a run where most samples contribute nothing. If loss
looks implausibly flat, grep the log for `tokenization mismatch`.

### 9.4 Collator

`DataCollatorForSupervisedDataset` (line 1390):

- Pads `input_ids` and `labels`, builds the attention mask.
- When `pad_token_id == eos_token_id`, temporarily rewrites eos to `-300` so
  padding does not mask real eos tokens, then restores it.
- Flattens per-sample lists of tiles/clips into one flat batch dimension.
- Packs audio into `batch["audios"] = {audios, lengths, lengths_for_llm}`.

`lengths` and `lengths_for_llm` differ: the first is fbank frames for the
encoder, the second is post-subsampling token count for the LLM.

## 10. The real-time duplex demo

`web_demo/server.py` (~1060 lines), Flask + SocketIO, vLLM-accelerated.

```
mic ──> WakeupAndVAD (silero) ──> audio chunks
                                     │
                         ┌───────────┴─── model worker (vLLM) ─── first char?
                         │                                          ├─ ☟ -> discard, stay silent
                         │                                          └─ else -> stream text
                         └─── tts_worker ──> TiCodec vocoder ──> PCM ──> browser
```

Requirements that are easy to trip over:

- **vLLM must be patched.** VITA is not upstreamed into vLLM, so
  `web_demo/vllm_tools/vllm_file/*` has to be copied into vLLM's
  `model_executor/models/`, and `qwen2p5_model_weight_file/*` over the
  checkpoint.
- **silero VAD is not vendored** — download `silero_vad.onnx` / `.jit` into
  `web_demo/wakeup_and_vad/resource/`.
- **Set `max_dynamic_patch: 1`** in the demo config. At the default 12 an image
  costs 3328 tokens and interactivity dies.

Note the demo redefines its own token indices (`IMAGE_TOKEN_INDEX = 51000`,
`server.py:61`) rather than importing from `vita/constants.py` — the vLLM path
uses real vocabulary slots instead of negative markers.

## 11. Model variants

Four `VITA*ForCausalLM` classes, dispatched by `--model_type`:

| `model_type` | Class | Backbone | Status |
|---|---|---|---|
| `qwen2p5_instruct` | `VITAQwen2ForCausalLM` | Qwen2.5-7B-Instruct | **VITA-1.5, the main path** |
| `qwen2p5_fo_instruct` | `VITAFOQwen2ForCausalLM` | Qwen2.5-7B | full-duplex experiment |
| `mixtral-8x7b` | `VITAMixtralForCausalLM` | Mixtral-8×7B | VITA-1.0 legacy |
| `nemo` | `VITAMistralForCausalLM` | Mistral-Nemo | variant |

### The monkey patch

`vita_qwen2.py:125`:

```python
Qwen2ForCausalLM.forward = custom_forward
```

The class's `forward` is replaced **globally at import time**, for every
`Qwen2ForCausalLM` in the process — not just VITA's subclass. The patched
version adds `output_hidden_states` handling that the duplex variant needs.
Two consequences: it tightly couples the code to a `transformers` version (see
[§12](#12-known-defects-and-rough-edges)), and any library in the same process
that instantiates a plain Qwen2 also gets the patched behaviour.

### The full-duplex variant

`vita_fo_qwen2.py` adds a `predictor_head` — a linear classifier over the last
hidden state predicting user state (speaking / not / …). Its loss is added to
the LM loss:

```python
state_logits = self.predictor_head(outputs[2][-1]).view(-1, self.predict_usr_state+1)
state_loss = loss_fct(state_logits, s_labels)
loss = loss + state_loss
outputs['loss'] = loss
```

This is the template to copy for adding any auxiliary loss — including RL.

## 12. Known defects and rough edges

Things that will cost time if unknown. Fixed items are specific to this fork.

| Issue | Status |
|---|---|
| `cache_position` breaks generation on the pinned `transformers==4.41.1` — upstream added the code and the pin in the same commit, and they were never consistent for Qwen2 | **fixed in this fork** ([REPRODUCE.md](./REPRODUCE.md#code-fixes-required)) |
| `prepare_inputs_labels_for_multimodal` sets `audio_features = None` when `audios is None`, then dereferences it unconditionally six lines later — so the `None` branch was unreachable and every caller passed a dummy `torch.zeros(400, 80)`, running the 341M-parameter audio encoder on text-only and image-only batches for nothing | **fixed in this fork** (`tools/test_audio_optional.py`) |
| `DataConfig` missing `Pretrain_video0` / `Pretrain_audio`, which several scripts pass → `KeyError` | **fixed in this fork** |
| `vita_nemo.py:78,178` has the same `cache_position` bug | **not fixed** — untestable without Nemo weights |
| `requirements.txt` does not install cleanly (unpinned `xformers` demands `torch>=2.10`; unpinned `pillow` needs a modern gcc); `six`/`timm`/`einops`/`PyYAML`/`opencv`/`librosa` are imported but unlisted | worked around ([REPRODUCE.md](./REPRODUCE.md#deviations-from-upstream-requirementstxt)) |
| Hard-coded cluster paths throughout `script/train/` (`/mnt/cfs/lhj/…`), plus `MASTER_ADDR`/`INDEX` pinned to the authors' network | must be edited locally |
| `GLOBAL_WEIGHTS_PATH` in `constants.py` is still the literal `/path/to/model_weights` | only reached on the LoRA branch |
| `mm_projector_lr` no longer affects `mm_projector` (`vita_trainer.py:190`) | upstream, unfixed |
| Tokenization mismatch silently voids a sample's labels | upstream behaviour, [§9.3](#93-label-masking) |
| `command.sh` is the authors' scratch history, referencing deleted files — not an entry point | — |
| `Conversation.get_prompt()` is **not idempotent**: it does `self.system = self.system[0]`, replacing the 3-element list with a string, so a second call on the same object indexes into that string and the whole system prompt collapses to one character. Harmless today (the data pipeline copies the template per sample and calls it once) but a live hazard for multi-turn RL rollouts that reuse a conversation object | upstream, unfixed ([PRIMER.md §6.2](./PRIMER.md#62-get_prompt-不幂等未记录的缺陷)) |
| LoRA was unusable: `find_all_linear_names` does not exclude `audio_encoder`, and whale has two `nn.Linear`s whose leaf name is the digit `"0"` (`encoder.enc.0.core.out.0`, `encoder.enc.1.embed.0`). peft matches target modules by suffix, so `"0"` matched `layers.0` — a whole `Qwen2DecoderLayer` — and peft rejected it. `--lora_enable True` failed outright regardless of memory | **fixed in this fork** (`script/train/smoke_test_lora.sh`, 23.3 GB peak on one GPU) |
| 4-bit path exists in `train.py` but no shipped script uses it | unverified |
| README audio examples reference `asset/vita_newlog.png`; the file is `.jpg` | — |
| **No training data is provided.** The paper's ~20M QA come from ~20 third-party datasets, plus ~5.7M unreleased synthetic samples and 110,000 h of *internal* ASR data | see [§13](#13-where-rl-would-attach) |

## 13. Where RL would attach

Upstream contains **only supervised fine-tuning** — no reward model, no
preference optimization, no rollout loop (`grep -rE 'reward|ppo|dpo|grpo|rlhf'`
over the upstream tree returns nothing). Adding RL was the goal of this fork,
and it is done: both lines have been trained on real data to a measured
result.

> **Status (2026-08-20): both RL lines are complete.** DPO
> (`vita/train/dpo_*.py`): first-step loss lands on the exact `-log(0.5)`,
> and SFT-then-DPO cuts POPE hallucination 10.97% → 8.82% — full record in
> [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md). GRPO (`vita/train/grpo_*.py`,
> multimodal): trained on CLEVR counting with a verifiable reward, held-out
> accuracy 44.6% → 77.4% in 400 steps — full record in
> [GRPO_DEEP_DIVE.md](./GRPO_DEEP_DIVE.md). What follows is the original
> analysis of the obstacles, kept because it explains the design; the
> resolved ones are annotated. See [HANDBOOK.md §8](./HANDBOOK.md#8-dpo离线偏好优化)
> for the operational side.

**What helps:**

- `VITAQwen2ForCausalLM` is a standard `Qwen2ForCausalLM` subclass returning
  `CausalLMOutputWithPast`, so logits are directly available for log-probs.
- `generate()` already handles multimodal input.
- `vita_fo_qwen2.py` demonstrates adding a custom loss term to the base loss.
- `VITATrainer.training_step` is an obvious injection point.
- The freeze switches make "train the LLM, freeze the encoders" a one-flag change.

**What gets in the way:**

1. **`generate()` is decorated `@torch.no_grad()`** (`vita_qwen2.py:197`). Online
   RL needs gradients on sampled sequences — the standard workaround is to
   recompute log-probs with a separate `forward`, which leads to the next point.

2. **The prompt's token sequence does not survive** ([§3](#3-the-central-idea-negative-index-placeholder-tokens),
   [§5](#5-prepare_inputs_labels_for_multimodal-the-heart-of-the-model)).
   `generate()` consumes `inputs_embeds`; sampled output gives back only the new
   text tokens. Recomputing log-probs means re-running the vision and audio
   encoders. With PPO that is ~3 encoder passes per step (policy, ref, critic).
   Mitigation: cache `inputs_embeds` from rollout and reuse it — `forward`
   already accepts `inputs_embeds` (`vita_qwen2.py:160`).
   **Resolved for DPO**: `dpo_trainer.py` calls
   `prepare_inputs_labels_for_multimodal` once per step and feeds the result
   to both the policy and the reference, so the encoders run once rather than
   four times.

3. **The monkey patch** ([§11](#11-model-variants)) makes wrappers like TRL's
   `AutoModelForCausalLMWithValueHead` risky. Writing the loss directly is
   likely safer than adding TRL on top of `transformers==4.41.1`.

4. **State tokens need a policy decision** ([§6](#6-the-state-tokens-rejection-mechanism)). Is the
   leading `☜`/`☞`/`☟` part of the action? If the reward model never sees it,
   train/inference distributions diverge. Simplest: strip it after rollout,
   before scoring.

5. **Memory.** PPO would need policy + ref + reward + critic, each carrying
   InternViT and whale. Not realistic on 8 GPUs without LoRA-sharing. DPO
   (policy + ref) and GRPO (no critic) are far more tractable.
   Partly eased: since `audios` became optional ([§12](#12-known-defects-and-rough-edges)),
   a text-and-image-only rollout no longer runs the 341M audio encoder in
   every model copy on every step. LoRA is now a working option too — one
   adapted 7B fits in 23.3 GB on a single card (161.5M trainable, 2.12%),
   so a policy/reference pair sharing frozen base weights is realistic.
   **Resolved for DPO**: the reference is the same weights under
   `disable_adapter()`, costing no extra memory at all.

6. **`mm_projector` escapes the adapter.** `train.py` applies LoRA at line 388
   but calls `initialize_vision_modules` at 395, and that method force-enables
   `mm_projector`'s gradients (`vita_arch.py:59-61`, comment: "In case it is
   frozen by LoRA"). `disable_adapter()` does not revert it, so any
   reference-model scheme built on adapter toggling silently stops matching
   the base policy once `mm_projector` updates — while the loss keeps looking
   reasonable. `train_dpo.py` freezes all non-adapter parameters explicitly
   (27.5M) and prints what it froze; the saved `non_lora_trainables.bin`
   should contain zero parameters.

**Suggested order (this is the order the fork actually followed):** offline
DPO first — no rollout, so obstacles 1 and 2 vanish and only obstacles 4 and 5
remain, both manageable. The reference model is a disabled LoRA adapter,
avoiding a second 7B in memory. Once log-prob computation was verified there,
GRPO reused it and added the rollout loop.

Note that RL does **not** require the missing SFT dataset: the released VITA-1.5
checkpoint is already trained, and preference data has to be constructed
regardless. If you do want to run the SFT stage on real data first, see
[DATASETS.md](./DATASETS.md) — about a third of the paper's 22M samples is
unreleased, but the public remainder is enough, and the survey gives three
plans sized to available disk.

## 14. The RL stack (DPO and GRPO)

Section 13 was written before any of this existed and reads as a survey of
obstacles. This section is the walkthrough of what was actually built, and
which of those obstacles turned out to be real.

Everything here is this fork's; upstream has no RL code at all.

### 14.1 What is shared

Both objectives reuse the same three pieces rather than duplicating them:

| Piece | Where | Why it is shared |
|---|---|---|
| Model construction | `train.py:232` | ~230 lines of loading, freeze switches and LoRA setup. A second copy would drift. |
| Reference policy | `peft`'s `disable_adapter()` | Same weights with the adapter off. Costs no extra memory. |
| Non-adapter freeze | `train_dpo.py`, `train_grpo.py` | See §14.5 — without it the reference silently stops being the base model. |

`train()` grew three optional parameters for this:

```python
def train(extra_arg_classes=(), data_module_factory=None, trainer_factory=None):
```

Called with none, it behaves exactly as it did before — the SFT path is
untouched, and the smoke test was re-run after the change to confirm.
`train_dpo.py` and `train_grpo.py` each supply a dataclass of extra
arguments, a data module and a trainer, and inherit everything else.

### 14.2 DPO

```
records with a `rejected` field
  → DPODataset            encodes each record twice, chosen and rejected
  → DPODataCollator       stacks them as [chosen…, rejected…], one batch of 2B
  → VITADPOTrainer.compute_loss
        fuse once  →  policy forward  →  reference forward (adapter off)
        → batch_sequence_logps (fp32)  →  dpo_loss
```

**`dpo_data.py`** wraps `LazySupervisedDataset` instead of reimplementing it.
`_encode` temporarily swaps the record in `list_data_dict`, calls the
untouched pipeline, and restores it — so chosen and rejected go through
exactly the same image tiling, prompt assembly and label masking, differing
only in the final assistant turn. That mutate-and-restore is safe under
DataLoader workers (each has its own copy) but is not thread-safe in
general.

**`dpo_loss.py`** holds two functions so the maths is testable without a
checkpoint. `batch_sequence_logps` casts to fp32 before the log-softmax
(§14.6) and sums, rather than averages, over supervised tokens — the length
bias that introduces is a property of DPO, not an oversight.

**`dpo_trainer.py`** overrides `compute_loss` only. Putting the objective in
the trainer rather than the model — which is what `vita_fo_qwen2.py:102`
does for its state head — keeps `VITAQwen2ForCausalLM` free of RL concerns,
which is why GRPO could be added later without touching the model at all.

The `-log(0.5)` identity is the load-bearing check: an untrained LoRA
adapter makes the policy identical to the reference, so the DPO logit is
zero and the first step's loss must be exactly 0.6931. Measured: 0.6931 on
the synthetic set, and 0.6931 again on 3000 real RLAIF-V pairs — the second
one matters more, because it exercises real images through the fusion path
rather than the same synthetic tile 24 times.

**Real data behaves differently from the smoke set, and the difference is
the point.** On synthetic pairs `rewards/accuracy` is 1.0 from the first
step: the preferred response answers the question and the dispreferred one
is generic filler, so the model separates them immediately. On RLAIF-V both
responses are fluent, on-topic, and differ mainly in whether a detail is
hallucinated — accuracy starts at chance and climbs slowly. A run that looks
like the smoke test on real preference data has almost certainly got a
length or format artefact to exploit, which is why
`tools/make_rlaif_v_data.py` reports mean response lengths (299 chosen
against 298 rejected here) rather than leaving that to be discovered later.

### 14.3 GRPO

```
prompt-only records
  → GRPOPromptDataset     tokenizes up to "<|im_start|>assistant\n"
  → GRPOPromptCollator    LEFT-pads (generation appends on the right)
  → VITAGRPOTrainer.compute_loss
        rollout: G samples per prompt, adapter dropout off
        → RewardCombiner scores each completion
        → group_advantages: (r - mean) / std within each group
        → policy log-probs (cached prompt embeds + sampled token embeds)
        → reference log-probs (adapter off)
        → grpo_loss
```

Two details of the rollout are specific to this codebase. Both live in
`grpo_trainer.py`.

**Sampling routes around `generate()`.** `vita_qwen2.py:209` raises
`NotImplementedError` on an `inputs_embeds` kwarg. Calling
`Qwen2ForCausalLM.generate(unwrapped, inputs_embeds=…)` directly accepts it,
which is what lets the prompt be encoded once and shared by the whole group.

**The model is put in `eval()` for sampling.** LoRA dropout would otherwise
make the sampling policy a different function from the one whose log-probs
are scored. The smoke script additionally sets `lora_dropout 0`.

**Log-probs are recomputed, not read off generation.** The gradient-carrying
pass has to happen anyway, so `old_logps` is taken from it — which pins the
ratio at exactly 1 on the first inner step instead of introducing bf16 drift
between two passes. With `--grpo_num_iterations 1` (the default) this is the
single-step regime: clipping never engages. Setting it above 1 turns on
PPO-style sample reuse: a `_ChunkRepeatSampler` replays the same prompt chunk
for μ consecutive optimizer steps, iteration 0 pays for generation, reward
scoring and the reference forward, and iterations 1..μ-1 only recompute the
policy log-probs under the updated weights (`_reuse_loss`) — that is where
the ratio departs from 1 and `grpo/clip_frac` stops being decorative.

`grpo_data.py` does **not** reuse `LazySupervisedDataset`. That class exists
to build supervised targets and every path in it assumes a final assistant
turn to compute labels for; here there is nothing to supervise. The collator
left-pads, unlike the SFT one, because batched generation appends to the
right and every prompt must end flush against the generation boundary.

### 14.4 Rewards

`rewards.py` is a registry so the score can come from a rule now and a
learned model later without the trainer changing:

```python
@register_reward("keyword")
def keyword_reward(prompt, response, meta) -> float: ...
```

Six rules ship: `keyword`, `length`, `no_repeat`, `state_token`, plus the
verifiable pair added for CLEVR — `answer` (binary exact match against
`reward_meta["answer"]`, with `<answer>` tag extraction and a last-number
fallback) and `format` (graded 1.0/0.5/0.0 for the
`<think>…</think><answer>…</answer>` structure). All return `[0, 1]`, which
is what makes the weights in `--reward_fns answer:1.0,format:0.3` comparable.

`JudgeReward` optionally loads a small instruct model and reads the
probability it assigns to each of the tokens `"1"`–`"5"`, returning their
weighted mean. Reading the distribution rather than parsing generated text
gives a continuous signal — which is what a group needs in order to be
ranked — and cannot fail to parse. When the sample carries a gold answer
(`reward_meta["gold"]`), the judge grades agreement with the reference
instead of free-floating quality.

**The property that matters:** a reward must separate the rollouts of one
prompt *by true quality*. A binary rule is fine when it is verifiable — the
group's pass/fail mix carries the signal, and `groups/degenerate_frac` tracks
the all-same groups that carry none. What is not fine is a graded score whose
within-group differences are stylistic luck: that is what the RLAIF-V proxy
rewards turned out to be, and why the project moved to CLEVR
(see [GRPO_DEEP_DIVE.md](./GRPO_DEEP_DIVE.md)).

### 14.5 Two traps that produce plausible-looking wrong runs

**`mm_projector` escapes the adapter.** `train.py` applies LoRA at line 388,
then `initialize_vision_modules` at 395 force-enables `mm_projector`'s
gradients — `vita_arch.py:59-61`, comment: *"In case it is frozen by LoRA"*.
`disable_adapter()` cannot undo that. Left alone, the "reference" keeps
training, so after step 1 it is no longer the base policy, the KL term
measures drift against a moving target, and nothing in the logs looks wrong.
Both entry points freeze every non-adapter parameter explicitly and print
the total (27.5M); the saved `non_lora_trainables.bin` should be empty.

**A degenerate group divides by zero.** GRPO's advantage is
`(r - mean) / std`. When every rollout in a group scores the same, that is
`0/0` → NaN, which flows into the gradients. With rule-based rewards this is
the common case early on, not an edge case. `group_advantages` zeroes those
groups and counts them; the trainer logs `groups/degenerate_frac`. A value
near 1.0 means the reward cannot tell the samples apart and almost nothing
is training — a failure that otherwise presents as "the model won't learn".

Note also that `group_advantages` uses the **population** standard
deviation. The unbiased one is NaN at `group_size == 1`, which would defeat
the guard.

### 14.6 Why log-probs are computed in fp32

`vita_qwen2.py:96` has `logits = logits.float()` commented out, so
`custom_forward` returns bf16 logits. Summing bf16 log-probs across a
152k-entry vocabulary loses a lot; both objectives then take a difference of
differences and scale it by beta ≈ 0.1.

Measured on a 200-token sequence:

| | error vs fp32 |
|---|---|
| bf16 throughout | 6.29 nats |
| bf16 input, fp32 inside | 0.0061 nats |

Three orders of magnitude. `batch_sequence_logps` and
`VITAGRPOTrainer._sequence_logps` both upcast before the log-softmax; do not
bypass them.

### 14.7 Shared media is encoded once

A DPO pair looks at one image, and a GRPO group looks at one image, but each
sequence carries its own copy because `vita_arch.py:429-432` asserts one
image feature per `<image>` token and has no way to express sharing.

`encode_images_deduped(images, group_size)` encodes the first repetition and
tiles the features. The vision tower is deterministic, so this is exact —
`tools/test_image_dedup.py` asserts it with `torch.equal`, not `allclose`,
because a small drift here would feed the LLM and read as training noise.

Opt-in via `image_group_size` on `prepare_inputs_labels_for_multimodal`,
defaulting to `None`, so SFT is unaffected. Saves 44–46% of the vision
forward for a pair; the gain scales as `(N-1)/N`, which is why it takes an
arbitrary repeat count — a GRPO group of 8 would save 87.5%.

### 14.8 Verification

Five CPU suites, no checkpoint needed, seconds to run:

| Suite | Covers |
|---|---|
| `test_dpo_loss.py` | 19 checks: the `-log(0.5)` identity, gradient routing, fp32 vs bf16 |
| `test_grpo_loss.py` | 39 checks: degenerate groups, `group_size=1`, KL non-negativity, clipping |
| `test_rewards.py` | 44 checks: each rule's edges, `[0, 1]` bounds, group separation |
| `test_image_dedup.py` | 11 checks: bitwise equality at ×2/×3/×4, order preservation |
| `test_audio_optional.py` | the `audios=None` path and its regressions |

Two real bugs were caught by these before any GPU time was spent, both the
same shape: a constant term left attached to the graph — `ref_delta` in DPO,
`old_logps` in GRPO. Harmless in the trainers, which produce both under
`no_grad`, but a loss function should not depend on its caller for
correctness.

End-to-end signals, measured on one H100:

| | Signal | Measured |
|---|---|---|
| DPO | first-step loss = `-log(0.5)` | 0.6931 |
| DPO | reward margin separates | +0.0020 → +0.0105 |
| GRPO | first-step KL = 0 | 0.0 |
| GRPO | reward mean rises | 0.7891 → 0.8622 |
| GRPO | completions shorten as the length reward rises | 75.1 → 67.3 tokens |

The last row is the most informative: two independent numbers moving in
agreement is much harder to get by accident than a loss curve going down.

### 14.9 What was later implemented, and what still is not

Items originally listed here as missing and since **implemented** (2026-08-20):

- **Multimodal GRPO.** `_fuse` splices vision features into the prompt
  embeddings once per batch (before the G-fold expansion, so the vision tower
  runs once per distinct image), the collator left-pads, and batched
  `generate` runs on the fused embeddings. Trained for real on CLEVR counting
  — held-out accuracy 44.6% → 77.4% in 400 steps
  ([GRPO_DEEP_DIVE.md](./GRPO_DEEP_DIVE.md)).
- **Multi-step rollout reuse.** `--grpo_num_iterations μ` replays each
  rollout batch for μ consecutive optimizer steps via `_ChunkRepeatSampler`
  + `_reuse_loss`; only the policy log-probs are recomputed on reuse steps,
  and that is where the ratio moves and clipping engages. μ=1 keeps the
  original on-policy path byte-identical.
- **Real data.** DPO trained on RLAIF-V (POPE 10.97% → 8.82% via SFT→DPO,
  [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md)); GRPO trained on CLEVR-70k with a
  verifiable reward, after two RLAIF-V proxy-reward rounds demonstrated that
  graded proxy scores rank within-group noise on open-ended description.

Still **not** implemented:

- **Full-parameter DPO/GRPO.** Both require LoRA and refuse to start without
  it, rather than half-working: the reference policy is defined as the
  adapter being switchable off.
- **Audio RL.** Out of scope by design — this fork's RL targets text +
  image/video only; the audio encoder stays frozen throughout.
