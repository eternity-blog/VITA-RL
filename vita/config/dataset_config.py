AudioFolder = ""
FolderDict = {
    #### NaturalCap
    "sharegpt4": "",
}
#### NaturalCap
ShareGPT4V = {"chat_path": ""}
ShareGPT4V0 = {"chat_path": ""}

#### Smoke test (see tools/make_smoke_data.py)
# Not real training data: a tiny synthetic set used only to verify that the
# training pipeline runs end to end. Enabled by setting VITA_SMOKE_DATA_DIR,
# so the default configuration is unchanged from upstream.
import os as _os

_SMOKE_DIR = _os.environ.get("VITA_SMOKE_DATA_DIR", "")
if _SMOKE_DIR:
    # The loader joins AudioFolder with a literal "audio" path segment
    # (data_utils_video_audio_neg_patch.py), so this points at the parent
    # of the audio/ directory rather than at audio/ itself.
    AudioFolder = _SMOKE_DIR
    FolderDict["smoke"] = _os.path.join(_SMOKE_DIR, "images")
    SmokeTest = {"chat_path": _os.path.join(_SMOKE_DIR, "smoke_train.json")}
else:
    SmokeTest = {"chat_path": ""}

#### DPO smoke test (see tools/make_dpo_smoke_data.py)
# Synthetic preference pairs, again only for verifying that the DPO path runs.
# Enabled by setting VITA_DPO_DATA_DIR.
_DPO_DIR = _os.environ.get("VITA_DPO_DATA_DIR", "")
if _DPO_DIR:
    # Same AudioFolder convention as above: parent of audio/, not audio/.
    AudioFolder = _DPO_DIR
    FolderDict["dpo_smoke"] = _os.path.join(_DPO_DIR, "images")
    DPOSmokeTest = {"chat_path": _os.path.join(_DPO_DIR, "dpo_train.json")}
else:
    DPOSmokeTest = {"chat_path": ""}

#### RLAIF-V (see tools/make_rlaif_v_data.py)
# Real preference pairs, unlike the smoke sets above: one image, one question,
# and two responses ranked by AI feedback. Enabled by VITA_RLAIF_DATA_DIR.
#
# tools/make_selfsample_data.py writes the same filename and record shape, so
# on-policy self-sampled data loads through this same variable -- point
# VITA_RLAIF_DATA_DIR at its output directory instead. The two differ in who
# wrote the responses (OmniLMM-12B vs VITA itself), which is the whole point
# of the comparison, but not in format.
_RLAIF_DIR = _os.environ.get("VITA_RLAIF_DATA_DIR", "")
if _RLAIF_DIR:
    # Same AudioFolder convention as above: parent of audio/, not audio/.
    AudioFolder = _RLAIF_DIR
    FolderDict["rlaif_v"] = _os.path.join(_RLAIF_DIR, "images")
    # make_selfsample_data.py tags its records set="selfsample"; register the
    # same image folder under that name so either source resolves.
    FolderDict["selfsample"] = _os.path.join(_RLAIF_DIR, "images")
    RLAIFV = {"chat_path": _os.path.join(_RLAIF_DIR, "rlaif_v_train.json")}
else:
    RLAIFV = {"chat_path": ""}

#### RLAIF-V as SFT (see tools/make_rlaif_v_sft_data.py)
# The same source as RLAIFV above, but keeping only the `chosen` response and
# discarding `rejected`. SFT needs no ranking -- only good answers -- so this
# sidesteps the judge problem that stalled four rounds of DPO. Enabled by
# VITA_SFT_DATA_DIR.
_SFT_DIR = _os.environ.get("VITA_SFT_DATA_DIR", "")
if _SFT_DIR:
    AudioFolder = _SFT_DIR
    FolderDict["rlaif_v_sft"] = _os.path.join(_SFT_DIR, "images")
    RLAIFVSFT = {"chat_path": _os.path.join(_SFT_DIR, "rlaif_v_sft_train.json")}
else:
    RLAIFVSFT = {"chat_path": ""}

#### GRPO smoke test (see tools/make_grpo_smoke_data.py)
# Prompt-only records; the policy writes its own completions during
# training. Enabled by setting VITA_GRPO_DATA_DIR.
_GRPO_DIR = _os.environ.get("VITA_GRPO_DATA_DIR", "")
if _GRPO_DIR:
    GRPOSmokeTest = {"chat_path": _os.path.join(_GRPO_DIR, "grpo_train.json")}
else:
    GRPOSmokeTest = {"chat_path": ""}

#### GRPO multimodal smoke test (see tools/make_grpo_mm_smoke_data.py)
# Same idea as GRPOSmokeTest but each record carries an image, so the
# trainer exercises the vision-fusion path (encode_images -> mm_projector ->
# prepare_inputs_labels_for_multimodal splices tile features into the token
# embeddings once per batch, then _rollout / _sequence_logps reuse them for
# all G completions). Enabled by VITA_GRPO_MM_DATA_DIR.
#
# The FolderDict entry is what lets GRPOPromptDataset._load_image_tiles resolve
# each record's `image` filename against <dir>/images -- the text smoke needs
# no such entry because it has no images to look up.
_GRPO_MM_DIR = _os.environ.get("VITA_GRPO_MM_DATA_DIR", "")
if _GRPO_MM_DIR:
    FolderDict["grpo_mm_smoke"] = _os.path.join(_GRPO_MM_DIR, "images")
    GRPOMMSmoke = {"chat_path": _os.path.join(_GRPO_MM_DIR, "grpo_mm_train.json")}
else:
    GRPOMMSmoke = {"chat_path": ""}

#### RLAIF-V for GRPO (see tools/make_rlaif_v_grpo_data.py)
# The same RLAIF-V source as RLAIFV/RLAIFVSFT, but prompt-only: GRPO writes
# its own completions and scores them, so only the image + question are kept.
# The chosen answer is mined for keywords and shipped in reward_meta for the
# `keyword` reward -- a proxy for groundedness, not a judge (see the script's
# docstring for the tradeoff). Enabled by VITA_RLAIF_GRPO_DATA_DIR.
#
# Records carry set="rlaif_v_grpo", so FolderDict is keyed the same way as the
# smoke set -- GRPOPromptDataset._load_image_tiles resolves each record's
# `image` filename against <dir>/images through this entry.
_RLAIF_GRPO_DIR = _os.environ.get("VITA_RLAIF_GRPO_DATA_DIR", "")
if _RLAIF_GRPO_DIR:
    FolderDict["rlaif_v_grpo"] = _os.path.join(_RLAIF_GRPO_DIR, "images")
    RLAIFVGRPO = {"chat_path": _os.path.join(_RLAIF_GRPO_DIR, "rlaif_v_grpo_train.json")}
else:
    RLAIFVGRPO = {"chat_path": ""}

#### CLEVR counting as SFT (see tools/make_clevr_sft_data.py)
# The control arm for the CLEVR GRPO experiment: the same prompts, but the
# gold solution is handed over as a supervised target instead of being used
# only to judge self-generated rollouts. Enabled by VITA_CLEVR_SFT_DATA_DIR.
_CLEVR_SFT_DIR = _os.environ.get("VITA_CLEVR_SFT_DATA_DIR", "")
if _CLEVR_SFT_DIR:
    AudioFolder = _CLEVR_SFT_DIR
    FolderDict["clevr_sft"] = _os.path.join(_CLEVR_SFT_DIR, "images")
    CLEVRSFT = {"chat_path": _os.path.join(_CLEVR_SFT_DIR, "clevr_sft_train.json")}
else:
    CLEVRSFT = {"chat_path": ""}

#### CLEVR counting for GRPO (see tools/make_clevr_grpo_data.py)
# R1-V's recipe (CLEVR-70k counting) adapted to this trainer: a *verifiable*
# reward. Each record carries reward_meta={"answer": "3"} and is scored by
# the binary `answer` reward plus the R1-style `format` reward -- no proxy
# (keyword/judge) in the loop, which is the point: RLAIF-V's proxy rewards
# rank stylistic variation, this ranks right against wrong.
# Enabled by VITA_CLEVR_GRPO_DATA_DIR.
_CLEVR_GRPO_DIR = _os.environ.get("VITA_CLEVR_GRPO_DATA_DIR", "")
if _CLEVR_GRPO_DIR:
    FolderDict["clevr_grpo"] = _os.path.join(_CLEVR_GRPO_DIR, "images")
    CLEVRGRPO = {"chat_path": _os.path.join(_CLEVR_GRPO_DIR, "clevr_grpo_train.json")}
else:
    CLEVRGRPO = {"chat_path": ""}
