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

#### GRPO smoke test (see tools/make_grpo_smoke_data.py)
# Prompt-only records; the policy writes its own completions during
# training. Enabled by setting VITA_GRPO_DATA_DIR.
_GRPO_DIR = _os.environ.get("VITA_GRPO_DATA_DIR", "")
if _GRPO_DIR:
    GRPOSmokeTest = {"chat_path": _os.path.join(_GRPO_DIR, "grpo_train.json")}
else:
    GRPOSmokeTest = {"chat_path": ""}
