"""Prompt-only dataset for GRPO.

Unlike SFT and DPO, GRPO needs no responses in the data -- the policy writes
its own during training and the reward function scores them. A record is
therefore just a question plus whatever the reward rules need to grade an
answer to it.

This does not go through LazySupervisedDataset. That class exists to build
supervised targets from a full conversation, and every path in it assumes a
final assistant turn to compute labels for. Here there is nothing to
supervise: we need the prompt tokenized and stopped right where the model
should start writing. Reusing it would mean fighting it.

First version is text-only, so no image or audio handling appears below.
The multimodal version will need prompt embeds cached per record and
encode_images_deduped to share them across the group.

Record format:

    {
      "set": "grpo_smoke",
      "id": "p_0001",
      "conversations": [{"from": "human", "value": "Describe a cat."}],
      "reward_meta": {"keywords": ["cat"], "target_len": [15, 60]}
    }
"""
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import transformers

from vita import conversation as conversation_lib
from vita.config import DataConfig


class GRPOPromptDataset(torch.utils.data.Dataset):
    """Yields tokenized prompts ending at the assistant's turn."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args

        dataset_list = DataConfig[str(data_args.dataset_use)]
        records: List[dict] = []
        for entry in dataset_list:
            path = entry["chat_path"]
            if not path:
                raise ValueError(
                    f"dataset '{data_args.dataset_use}' has an empty chat_path; "
                    "is VITA_GRPO_DATA_DIR set?"
                )
            with open(path) as f:
                records.extend(json.load(f))

        for i, r in enumerate(records):
            if not r.get("conversations"):
                raise ValueError(f"record {r.get('id', i)} has no conversations")
            if r["conversations"][0].get("from") != "human":
                raise ValueError(
                    f"record {r.get('id', i)} must open with a human turn"
                )

        random.shuffle(records)
        self.records = records

    def __len__(self):
        return len(self.records)

    @property
    def modality_lengths(self):
        # VITATrainer._get_train_sampler reads this when
        # group_by_modality_length is on. Text-only, so all negative.
        return [-len(r["conversations"][0]["value"].split()) for r in self.records]

    def __getitem__(self, i) -> Dict:
        record = self.records[i]

        # Build the prompt with an empty assistant turn so get_prompt emits
        # the "<|im_start|>assistant\n" opener and stops -- that is exactly
        # where generation should begin.
        conv = conversation_lib.default_conversation.copy()
        for turn in record["conversations"]:
            role = conv.roles[0] if turn["from"] == "human" else conv.roles[1]
            conv.append_message(role, turn["value"])
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt("lang")

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
        return {
            "input_ids": input_ids,
            "prompt_text": record["conversations"][0]["value"],
            "reward_meta": record.get("reward_meta", {}),
            "id": record.get("id", str(i)),
        }


@dataclass
class GRPOPromptCollator:
    """Left-pads prompts so every sequence in the batch ends at the same
    position.

    Generation appends to the right, so a right-padded batch would have the
    model continue from the middle of one sequence and past the pad tokens
    of another. Left padding puts every prompt's final token flush against
    the generation boundary, which is the standard arrangement for batched
    decoding and the reason this collator exists rather than reusing the SFT
    one (that one right-pads, correctly, for teacher forcing).
    """

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict:
        ids = [inst["input_ids"] for inst in instances]
        max_len = max(x.shape[0] for x in ids)
        pad_id = self.tokenizer.pad_token_id

        padded, mask = [], []
        for x in ids:
            gap = max_len - x.shape[0]
            padded.append(
                torch.cat([torch.full((gap,), pad_id, dtype=x.dtype), x]) if gap else x
            )
            mask.append(
                torch.cat([torch.zeros(gap, dtype=torch.bool), torch.ones(x.shape[0], dtype=torch.bool)])
            )

        return {
            "input_ids": torch.stack(padded),
            "attention_mask": torch.stack(mask),
            "prompt_texts": [inst["prompt_text"] for inst in instances],
            "reward_metas": [inst["reward_meta"] for inst in instances],
            "ids": [inst["id"] for inst in instances],
        }


def make_grpo_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    return dict(
        train_dataset=GRPOPromptDataset(tokenizer=tokenizer, data_args=data_args),
        eval_dataset=None,
        data_collator=GRPOPromptCollator(tokenizer=tokenizer),
    )
