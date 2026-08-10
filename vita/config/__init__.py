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
}

NoPatchSets = ["khair", "jester"]
