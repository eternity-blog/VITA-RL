from .dataset_config import *

NaturalCap0 = [ShareGPT4V0]
NaturalCap = [ShareGPT4V]

DataConfig = {
    "Pretrain_video": NaturalCap0,
    # Upstream scripts pass these two --dataset_use values but never defined
    # them, so those scripts fail with a KeyError. Map them to the same
    # (unconfigured) dataset as Pretrain_video; fill in dataset_config.py to
    # actually use them.
    "Pretrain_video0": NaturalCap0,
    "Pretrain_audio": NaturalCap0,
    # Synthetic pipeline smoke test -- see tools/make_smoke_data.py.
    "SmokeTest": [SmokeTest],
    # Synthetic preference pairs for DPO -- see tools/make_dpo_smoke_data.py.
    "DPOSmokeTest": [DPOSmokeTest],
    # Real preference pairs from RLAIF-V -- see tools/make_rlaif_v_data.py.
    "RLAIFV": [RLAIFV],
    # The same source used as plain SFT on the chosen responses.
    "RLAIFVSFT": [RLAIFVSFT],
    # Prompt-only records for GRPO -- see tools/make_grpo_smoke_data.py.
    "GRPOSmokeTest": [GRPOSmokeTest],
    # Image+text prompt-only records for GRPO -- see
    # tools/make_grpo_mm_smoke_data.py. Same idea as GRPOSmokeTest but each
    # record carries an image, so the trainer exercises the vision-fusion
    # path (encode_images -> mm_projector -> prepare_inputs_labels_for_
    # multimodal splices tile features into the token embeddings once per
    # batch, then _rollout / _sequence_logps reuse them for all G
    # completions). Enabled by VITA_GRPO_MM_DATA_DIR.
    "GRPOMMSmoke": [GRPOMMSmoke],
    # Prompt-only image+text records from RLAIF-V for GRPO -- see
    # tools/make_rlaif_v_grpo_data.py. Same source as RLAIFV/RLAIFVSFT but
    # the policy writes its own completions; the gold answer is mined for
    # keywords and shipped in reward_meta for the `keyword` reward.
    # Enabled by VITA_RLAIF_GRPO_DATA_DIR.
    "RLAIFVGRPO": [RLAIFVGRPO],
    # CLEVR counting with a verifiable exact-match answer reward (R1-V's
    # recipe) -- see tools/make_clevr_grpo_data.py. Enabled by
    # VITA_CLEVR_GRPO_DATA_DIR.
    "CLEVRGRPO": [CLEVRGRPO],
    # The SFT control arm for CLEVRGRPO: same prompts, gold solution as the
    # supervised target -- see tools/make_clevr_sft_data.py. Enabled by
    # VITA_CLEVR_SFT_DATA_DIR.
    "CLEVRSFT": [CLEVRSFT],
}

NoPatchSets = ["khair", "jester"]
