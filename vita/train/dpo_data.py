"""Preference-pair dataset for DPO.

Wraps the existing LazySupervisedDataset rather than reimplementing it. That
class carries ~1500 lines of image tiling, audio fbank extraction and prompt
assembly which DPO needs unchanged; the only difference is that each record
must yield two responses to the same prompt instead of one.

The trick is to hand the wrapped dataset a rewritten record. For a sample

    {"conversations": [human, gpt(chosen)], "rejected": "...", "image": ...}

we build two views that differ only in the final assistant turn, run each
through the untouched pipeline, and pair up the results. Everything that is
not the response -- prompt tokens, image tiles, audio -- is identical by
construction, which is exactly what DPO assumes.

Record format (SFT format plus one key):

    {
      "set": "dpo_smoke",
      "id": "pref_0001",
      "conversations": [
        {"from": "human", "value": "<image>\\nDescribe this image."},
        {"from": "gpt",   "value": "<the preferred response>"}
      ],
      "rejected": "<the dispreferred response>",
      "image": "vita_newlog.jpg"
    }
"""
import copy
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence

from vita.constants import IGNORE_INDEX
from vita.util.data_utils_video_audio_neg_patch import (
    DataCollatorForSupervisedDataset,
    LazySupervisedDataset,
)


class DPODataset(torch.utils.data.Dataset):
    """Yields a chosen/rejected pair per record.

    Each __getitem__ returns the two encoded sequences plus one copy of the
    shared image/audio tensors -- the pair looks at the same media, so
    duplicating it would only waste encoder time and memory.
    """

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super().__init__()
        self.inner = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)

        missing = [
            r.get("id", f"index {i}")
            for i, r in enumerate(self.inner.list_data_dict)
            if "rejected" not in r
        ]
        if missing:
            raise ValueError(
                f"{len(missing)} record(s) have no 'rejected' field, e.g. {missing[:3]}. "
                "A DPO dataset needs a dispreferred response per record; see "
                "tools/make_dpo_smoke_data.py for the expected format."
            )

    def __len__(self):
        return len(self.inner.list_data_dict)

    @property
    def modality_lengths(self):
        return self.inner.modality_lengths

    def _encode(self, index: int, use_rejected: bool) -> Dict[str, torch.Tensor]:
        """Run one view of a record through the unmodified SFT pipeline."""
        record = self.inner.list_data_dict[index]
        if use_rejected:
            record = copy.deepcopy(record)
            # Replace the last assistant turn with the dispreferred response.
            for turn in reversed(record["conversations"]):
                if turn["from"] == "gpt":
                    turn["value"] = record["rejected"]
                    break
            else:
                raise ValueError(f"record {record.get('id', index)} has no 'gpt' turn")

        original = self.inner.list_data_dict[index]
        self.inner.list_data_dict[index] = record
        try:
            return self.inner[index]
        finally:
            self.inner.list_data_dict[index] = original

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        chosen = self._encode(i, use_rejected=False)
        rejected = self._encode(i, use_rejected=True)

        out = {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_labels": chosen["labels"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_labels": rejected["labels"],
        }
        # Media is shared, so keep one copy (from the chosen view).
        for key in ("image", "audio", "audio_lengths", "audio_lengths_for_llm"):
            if key in chosen:
                out[key] = chosen[key]
        return out


@dataclass
class DPODataCollator:
    """Collates preference pairs into one batch of 2B sequences.

    Layout is [chosen_0..chosen_{B-1}, rejected_0..rejected_{B-1}]: the
    trainer splits it back in half after the forward pass. Packing both
    halves into a single batch means one forward instead of two, and lets the
    existing SFT collator handle all the padding and media stacking.

    Media tensors are duplicated across the two halves because the model
    consumes one image per sequence, but each pair still only decodes its
    image once, in the dataset.
    """

    tokenizer: transformers.PreTrainedTokenizer

    def __post_init__(self):
        self._inner = DataCollatorForSupervisedDataset(tokenizer=self.tokenizer)

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        flat = []
        for side in ("chosen", "rejected"):
            for inst in instances:
                item = {
                    "input_ids": inst[f"{side}_input_ids"],
                    "labels": inst[f"{side}_labels"],
                }
                for key in ("image", "audio", "audio_lengths", "audio_lengths_for_llm"):
                    if key in inst:
                        item[key] = inst[key]
                flat.append(item)

        batch = self._inner(flat)
        # Carried through so compute_loss knows where the split is, rather
        # than assuming an even division of the batch.
        batch["num_pairs"] = torch.tensor(len(instances))
        return batch


def make_dpo_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    return dict(
        train_dataset=DPODataset(tokenizer=tokenizer, data_args=data_args),
        eval_dataset=None,
        data_collator=DPODataCollator(tokenizer=tokenizer),
    )
