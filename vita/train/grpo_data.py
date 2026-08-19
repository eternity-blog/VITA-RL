"""Prompt dataset for GRPO, with optional image (multimodal) support.

Unlike SFT and DPO, GRPO needs no responses in the data -- the policy writes
its own during training and the reward function scores them. A record is
therefore just a question plus whatever the reward rules need to grade an
answer to it.

This does not go through LazySupervisedDataset. That class exists to build
supervised targets from a full conversation, and every path in it assumes a
final assistant turn to compute labels for. Here there is nothing to
supervise: we need the prompt tokenized and stopped right where the model
should start writing. Reusing it would mean fighting it.

The text-only path tokenizes the prompt with the plain tokenizer and the
trainer looks up `embed_tokens(input_ids)` to get prompt embeddings -- no
image, no fusion.

The image path (a record carrying an `"image"` field) loads the image, tiles
it with the same `dynamic_preprocess` + `image_processor` the SFT loader
uses, and builds the prompt with `patch_num` copies of the `<image>` token
(via `preprocess_multimodal`) so that `prepare_inputs_labels_for_multimodal`
can splice the vision features in place of each placeholder. The trainer
then fuses once per batch (the vision tower runs once, not once per rollout)
and the fused `inputs_embeds` becomes the `prompt_embeds` that the existing
rollout / log-prob code already operates on. That is the whole change: swap
`embed_tokens(input_ids)` for a fused `inputs_embeds`, and the rest of GRPO
(rollout, scoring, advantages, loss) is unchanged.

Record format (text-only, unchanged):

    {
      "set": "grpo_smoke",
      "id": "p_0001",
      "conversations": [{"from": "human", "value": "Describe a cat."}],
      "reward_meta": {"keywords": ["cat"], "target_len": [15, 60]}
    }

Record format (image+text):

    {
      "set": "grpo_mm_smoke",
      "id": "mm_0001",
      "conversations": [{"from": "human", "value": "<image>\nDescribe this image."}],
      "image": "vita_newlog.jpg",
      "reward_meta": {"keywords": ["logo"], "target_len": [20, 200]}
    }

The `<image>` literal in the human turn is what routes a record onto the
image path; exactly one is expected per record (multi-image is not supported
here -- the SFT loader's list-image branch is not needed for the grounded
tasks GRPO targets).
"""
import copy
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import transformers
from PIL import Image

from vita import conversation as conversation_lib
from vita.config import DataConfig, FolderDict
from vita.constants import DEFAULT_IMAGE_TOKEN
from vita.util.data_utils_video_audio_neg_patch import preprocess_multimodal
from vita.util.mm_utils import tokenizer_image_token


class GRPOPromptDataset(torch.utils.data.Dataset):
    """Yields tokenized prompts ending at the assistant's turn.

    Each item carries either text-only fields (`input_ids`, `prompt_text`,
    `reward_meta`, `id`) or, when the record has an image, those same fields
    plus an `image` tensor list of 448x448 tiles and a `has_image` flag. The
    collator stacks the tiles across the batch into the single
    `[total_tiles, 3, 448, 448]` tensor that
    `prepare_inputs_labels_for_multimodal` expects in the common
    same-shape case.
    """

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

        n_with_image = 0
        for i, r in enumerate(records):
            if not r.get("conversations"):
                raise ValueError(f"record {r.get('id', i)} has no conversations")
            if r["conversations"][0].get("from") != "human":
                raise ValueError(
                    f"record {r.get('id', i)} must open with a human turn"
                )
            if "image" in r:
                n_with_image += 1

        self.is_multimodal = n_with_image > 0
        # A batch is fed to one call of prepare_inputs_labels_for_multimodal,
        # which asserts one image feature per <image> token and consumes an
        # entry for every sequence. Mixing image and text records in a batch
        # would misalign that accounting, so require the dataset to be
        # homogeneous. (Text-only datasets have n_with_image == 0, which is
        # fine and is the original behaviour.)
        if self.is_multimodal and n_with_image != len(records):
            raise ValueError(
                f"dataset '{data_args.dataset_use}' mixes image and text-only "
                f"records ({n_with_image}/{len(records)} have images). Batch the "
                "two kinds separately, or drop the text-only records."
            )

        if self.is_multimodal:
            if getattr(data_args, "image_processor", None) is None:
                raise ValueError(
                    "data_args.image_processor is None; the trainer must set it "
                    "from vision_tower.image_processor before building the data "
                    "module (train.py does this for SFT/DPO; GRPO must too)."
                )
            self.image_processor = data_args.image_processor
            # Resolve the image folder per record's `set` exactly as SFT does:
            # FolderDict maps a set name to a directory of images.
            self.folder_dict = {k: v for k, v in FolderDict.items() if v}
            # dynamic_preprocess parameters live on data_args (defaults in
            # DataArguments: min=1, max=12, use_thumbnail=True).
            self.min_dynamic_patch = getattr(data_args, "min_dynamic_patch", 1)
            self.max_dynamic_patch = getattr(data_args, "max_dynamic_patch", 12)
            self.use_thumbnail = getattr(data_args, "use_thumbnail", True)
            self.image_aspect_ratio = getattr(data_args, "image_aspect_ratio", None)

        random.shuffle(records)
        self.records = records

    def __len__(self):
        return len(self.records)

    @property
    def modality_lengths(self):
        # VITATrainer._get_train_sampler reads this when
        # group_by_modality_length is on. Image records are positive, text
        # negative, so the sampler groups like with like.
        out = []
        for r in self.records:
            cur = sum(len(c["value"].split()) for c in r["conversations"])
            out.append(cur if "image" in r else -cur)
        return out

    # -- image loading (mirrors LazySupervisedDataset's single-image path) --

    def _load_image_tiles(self, record: dict) -> List[torch.Tensor]:
        """Return a list of [3, 448, 448] tiles for one record's image.

        Reproduces the single-image branch of
        data_utils_video_audio_neg_patch.py (the `image` and `not audio`
        case, single file not list). Both the `pad` aspect-ratio path
        (expand2square then tile) and the default path (tile the raw image)
        are supported, matching what the SFT loader does for the same
        `--image_aspect_ratio` flag.
        """
        image_file = record["image"]
        set_id = record.get("set", None)
        if set_id is None or set_id not in self.folder_dict:
            raise ValueError(
                f"record {record.get('id')} has set={set_id!r} which is not in "
                f"FolderDict (have {list(self.folder_dict)}); register it in "
                "vita/config/dataset_config.py."
            )
        image_folder = self.folder_dict[set_id]
        image = Image.open(
            os.path.join(image_folder, image_file.replace("\\", "/"))
        ).convert("RGB")

        processor = self.image_processor
        if "height" in processor.size.keys():
            image_size = processor.size["height"]
        elif "shortest_edge" in processor.size.keys():
            image_size = processor.size["shortest_edge"]
        else:
            raise NotImplementedError("image_processor.size has no height/shortest_edge")

        if self.image_aspect_ratio == "pad":
            # Inline expand2square: pad the shorter side to a square with the
            # processor's mean colour, so tiling is on a square canvas.
            width, height = image.size
            mean = tuple(int(x * 255) for x in processor.image_mean)
            if width == height:
                square = image
            elif width > height:
                square = Image.new(image.mode, (width, width), mean)
                square.paste(image, (0, (width - height) // 2))
            else:
                square = Image.new(image.mode, (height, height), mean)
                square.paste(image, ((height - width) // 2, 0))
            image = square

        tiles, _patch_num = self._dynamic_preprocess(image, image_size)
        return [
            processor.preprocess(t, return_tensors="pt")["pixel_values"][0]
            for t in tiles
        ]

    def _dynamic_preprocess(self, image, image_size):
        """Tile an image into image_size x image_size patches.

        Thin wrapper around the module-level dynamic_preprocess in
        data_utils_video_audio_neg_patch, kept here so the GRPO data path
        has no dependency on the SFT Dataset class internals.
        """
        from vita.util.data_utils_video_audio_neg_patch import dynamic_preprocess

        return dynamic_preprocess(
            image,
            min_num=self.min_dynamic_patch,
            max_num=self.max_dynamic_patch,
            image_size=image_size,
            use_thumbnail=self.use_thumbnail,
        )

    # -- prompt assembly --

    def _build_image_prompt(self, record: dict, n_tiles: int) -> str:
        """Build the image prompt string ending at `<|im_start|>assistant\\n`.

        Mirrors the SFT single-image path: `preprocess_multimodal` expands the
        one `<image>` literal into `n_tiles` copies (one per tile) so
        `tokenizer_image_token` emits one IMAGE_TOKEN_INDEX placeholder per
        tile, which is exactly what `prepare_inputs_labels_for_multimodal`
        splices the vision features into. `get_prompt("image")` then selects
        the image system prompt, matching what the base model was trained on.
        """
        conv = conversation_lib.default_conversation.copy()
        sources = [copy.deepcopy(record["conversations"])]
        sources = preprocess_multimodal(sources, self.data_args, patch_num=[n_tiles])
        for turn in sources[0]:
            role = conv.roles[0] if turn["from"] == "human" else conv.roles[1]
            conv.append_message(role, turn["value"])
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt("image")

    def __getitem__(self, i) -> Dict:
        record = self.records[i]
        has_image = "image" in record

        if has_image:
            tiles = self._load_image_tiles(record)
            prompt = self._build_image_prompt(record, len(tiles))
            input_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")
            return {
                "input_ids": input_ids,
                "image": tiles,
                "has_image": True,
                "prompt_text": record["conversations"][0]["value"],
                "reward_meta": record.get("reward_meta", {}),
                "id": record.get("id", str(i)),
            }

        # Text path builds the prompt straight from the raw conversation, with
        # NO preprocess_multimodal. That helper normalises "\n\n" -> "\n" on
        # every sentence when data_args.is_multimodal is True (the default),
        # which would change the tokenization and break the step-1
        # grpo/kl == 0 identity check (policy must equal reference). The
        # original text-only GRPO dataset built the prompt exactly this way;
        # we stay byte-identical to it so the text smoke test still passes.
        conv = conversation_lib.default_conversation.copy()
        for turn in record["conversations"]:
            role = conv.roles[0] if turn["from"] == "human" else conv.roles[1]
            conv.append_message(role, turn["value"])
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt("lang")
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
        return {
            "input_ids": input_ids,
            "has_image": False,
            "prompt_text": record["conversations"][0]["value"],
            "reward_meta": record.get("reward_meta", {}),
            "id": record.get("id", str(i)),
        }


@dataclass
class GRPOPromptCollator:
    """Left-pads prompts so every sequence ends at the same position, and
    stacks image tiles across the batch.

    Generation appends to the right, so a right-padded batch would have the
    model continue from the middle of one sequence and past the pad tokens of
    another. Left padding puts every prompt's final token flush against the
    generation boundary, which is the standard arrangement for batched
    decoding and the reason this collator exists rather than reusing the SFT
    one (that one right-pads, correctly, for teacher forcing).

    For image batches the tiles are flattened across the batch and stacked
    into one `[total_tiles, 3, 448, 448]` tensor -- the shape
    `prepare_inputs_labels_for_multimodal` takes in the common case where
    every tile is the same size (which they are: 448x448). The trainer fuses
    that once into `inputs_embeds`, which becomes the prompt embeddings the
    rollout code already consumes.
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

        batch = {
            "input_ids": torch.stack(padded),
            "attention_mask": torch.stack(mask),
            "prompt_texts": [inst["prompt_text"] for inst in instances],
            "reward_metas": [inst["reward_meta"] for inst in instances],
            "ids": [inst["id"] for inst in instances],
            "has_image": [inst.get("has_image", False) for inst in instances],
        }

        if any(inst.get("has_image") for inst in instances):
            # All instances in an image batch carry images (enforced by the
            # dataset), so flatten every tile into one list and stack.
            all_tiles = []
            for inst in instances:
                all_tiles.extend(inst["image"])
            if not all(x.shape == all_tiles[0].shape for x in all_tiles):
                # Different tile shapes would only happen with non-448 images;
                # fall back to a list, which prepare_inputs_labels_for_multimodal
                # also accepts (the ndim==5 / list branch).
                batch["images"] = all_tiles
            else:
                batch["images"] = torch.stack(all_tiles)

        return batch


def make_grpo_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    return dict(
        train_dataset=GRPOPromptDataset(tokenizer=tokenizer, data_args=data_args),
        eval_dataset=None,
        data_collator=GRPOPromptCollator(tokenizer=tokenizer),
    )
