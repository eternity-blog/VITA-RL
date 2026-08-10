"""Build on-policy preference pairs by sampling the model and ranking against
a reference answer.

Why this exists
---------------
DPO on RLAIF-V moved nothing, and the reason turned out to be specific: every
chosen/rejected pair in that dataset was written by OmniLMM-12B and judged by
OmniLMM-12B. VITA never produced those sentences, so the task DPO was posing
is "rank two responses from a different model" -- out of distribution, and
measurably so. The frozen base separates those pairs at 53.6%, barely above
chance, and a gradient built on that is mostly noise.

The fix is to make the *candidates* on-policy: VITA samples K responses to the
same image and question, so both sides of every pair sit exactly on its own
decision boundary. Separability then comes from the model's own spread rather
than from another model's phrasing.

Judging
-------
Ranking is done against RLAIF-V's `chosen` as a reference answer: of VITA's K
samples, the one most similar to the reference becomes `chosen`, the least
similar becomes `rejected`. Similarity is token-F1 over content words, which
rewards saying the same thing without demanding the same wording.

The division of labour matters. The reference decides *which of VITA's answers
is better*; it never enters the training data itself. So the pair stays
on-policy while the preference direction borrows an external signal.

Two earlier judges that did not work, kept here because the failure is
instructive:

  - **Self-consistency (majority vote).** Sample K, let the largest
    agreement group win. Measured on real prompts this is often simply
    wrong -- one case had 2/5 votes for "paper crafter" when the answer is
    leather, and another elected a bare "1800000" over a correct sentence.
    Open-ended VQA paraphrases also do not cluster: a third case had every
    sample in its own group, so there was no majority to speak of.
  - **Length or format heuristics.** Rejected outright: they hand DPO a
    shortcut that has nothing to do with answer quality.

Known limitation of the reference judge
---------------------------------------
Lexical similarity cannot tell which word carries the answer. Measured case:
for "leather crafter or paper crafter?", the wrong sample scored 0.50 against
the reference and the right one 0.31, because the wrong one happened to share
four incidental words (hole, punch, scissors, crafting) while the decisive
word `leather` counted once. Up-weighting terms the samples disagree about
(--contested-weight) does not fix it, because the incidental words are
contested too.

Spot-checked at 1 inverted pair in 10. That is label noise DPO can tolerate
far better than the 53.6%-separable off-policy data it replaces, but it is a
real ceiling: a semantic judge (NLI or an LLM grader) is the way past it.
**Run tools/probe_preference_separability.py on the output before training** --
that is the measurement that decides whether this data is worth the compute.

Usage:
    python tools/make_selfsample_data.py \\
        --parquet shard000.parquet --out-dir out/ \\
        --limit 2000 --group-size 6 --gpu 0
"""
import argparse
import hashlib
import io
import json
import os
import re
import sys
from collections import Counter

# Words too common to carry meaning when comparing two answers to the same
# question -- keeping them inflates every similarity score toward each other.
STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "as", "that",
    "this", "these", "those", "it", "its", "there", "and", "or", "but",
    "image", "picture", "photo", "shows", "appears", "likely", "seems",
}


def content_tokens(text):
    """Lowercase content words, state token and punctuation stripped."""
    t = text.strip()
    # VITA prefixes a state token onto every reply (data_utils:128-135).
    if t[:1] in ("☜", "☞", "☟"):
        t = t[1:]
    t = t.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return [w for w in t.split() if w not in STOP and len(w) > 1]


def similarity(a, b):
    """Token-F1 between two answers.

    F1 rather than raw overlap so that a long rambling answer cannot win by
    containing the reference's words among many others -- precision punishes
    that. This is the standard SQuAD-style match, which is the right shape
    here: we want "asserts the same content", not "same wording".
    """
    ta, tb = Counter(content_tokens(a)), Counter(content_tokens(b))
    if not ta or not tb:
        return 0.0
    overlap = sum((ta & tb).values())
    if not overlap:
        return 0.0
    prec, rec = overlap / sum(ta.values()), overlap / sum(tb.values())
    return 2 * prec * rec / (prec + rec)


def contested_terms(responses):
    """Words that some samples use and others do not.

    Plain bag-of-words similarity cannot see which word carries the answer.
    Measured case: for "leather crafter or paper crafter?", the wrong sample
    scored 0.40 against the reference and the right one 0.35, because both
    shared the incidental vocabulary (tools, hole, punch, scissors) and the
    decisive word was one token out of fifteen.

    A term that appears in some of the model's samples but not others is
    exactly where the samples disagree, which is what the reference should
    arbitrate. Terms every sample shares carry no information about which
    sample is better.
    """
    per = [set(content_tokens(r)) for r in responses]
    if len(per) < 2:
        return set()
    everywhere = set.intersection(*per)
    anywhere = set.union(*per)
    return anywhere - everywhere


def ranking_score(response, reference, contested, weight):
    """Similarity to the reference, with contested terms weighted up.

    `weight` multiplies the contribution of terms the samples disagree about,
    so agreeing with the reference on the decisive word outranks agreeing on
    boilerplate. weight=1 reduces exactly to plain token-F1.
    """
    tr, tf = Counter(content_tokens(response)), Counter(content_tokens(reference))
    if not tr or not tf:
        return 0.0

    def wt(tok):
        return weight if tok in contested else 1.0

    overlap = sum(min(tr[t], tf[t]) * wt(t) for t in tr.keys() & tf.keys())
    if not overlap:
        return 0.0
    denom_r = sum(c * wt(t) for t, c in tr.items())
    denom_f = sum(c * wt(t) for t, c in tf.items())
    prec, rec = overlap / denom_r, overlap / denom_f
    return 2 * prec * rec / (prec + rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, nargs="+")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default=os.environ.get("VITA_CKPT"))
    ap.add_argument("--limit", type=int, default=2000, help="pairs to emit")
    ap.add_argument("--group-size", type=int, default=6, help="samples per prompt")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--max-tiles", type=int, default=4,
                    help="dynamic_preprocess max_num. The default of 12 in "
                         "video_audio_demo.py maximises quality; 4 is much "
                         "faster and enough for sampling short VQA answers, "
                         "where CPU-side tile normalisation dominates.")
    ap.add_argument("--max-prompts", type=int, default=20000)
    ap.add_argument("--contested-weight", type=float, default=4.0,
                    help="how much to up-weight terms the samples disagree "
                         "about when scoring against the reference. 1.0 is "
                         "plain token-F1, which was measured picking the wrong "
                         "answer when the decisive word was one token in 15.")
    ap.add_argument("--min-margin", type=float, default=0.15,
                    help="minimum similarity gap between best and worst "
                         "sample. Below this the preference direction is "
                         "arbitrary and the pair only adds gradient noise.")
    ap.add_argument("--model-type", default="qwen2p5_instruct")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--shard", type=int, default=0,
                    help="this worker's index; with --num-shards N, worker i "
                         "takes every Nth prompt so N GPUs can run in parallel")
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    if not args.model:
        sys.exit("pass --model or set VITA_CKPT")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo)

    import pyarrow.parquet as pq
    import torch
    from PIL import Image
    from vita.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from vita.conversation import conv_templates
    from vita.model.builder import load_pretrained_model
    from vita.util.data_utils_video_audio_neg_patch import dynamic_preprocess
    from vita.util.mm_utils import get_model_name_from_path, tokenizer_image_token

    img_dir = os.path.join(args.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "audio"), exist_ok=True)

    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model, None, get_model_name_from_path(args.model),
        model_type=args.model_type, device_map="auto",
    )
    model.eval()
    # builder.py loads the towers in fp16 while the LLM may differ, and
    # InternViTVisionTower.forward re-casts to a dtype property that then
    # disagrees with the weights outside DeepSpeed. Pin everything.
    model.to(dtype=torch.float16)

    records, seen_images = [], {}
    stats = Counter()

    def sample_group(question, image):
        """Draw group_size responses to one (image, question).

        Mirrors video_audio_demo.py rather than calling image_processor
        directly: VITA tiles an image into p_num patches and asserts one
        image feature per <image> token, so the prompt must repeat the token
        p_num times. Passing a single <image> with multi-tile pixels trips a
        bare AssertionError in vita_arch with no message.

        Tiling and normalisation happen once per prompt, not once per sample.
        CLIPImageProcessor normalises on CPU and, at max_num=12, dominates
        the wall clock -- a py-spy dump of the naive version showed the
        process sitting in image_transforms.normalize, not in generate.
        """
        tiles, p_num = dynamic_preprocess(
            image, min_num=1, max_num=args.max_tiles,
            image_size=448, use_thumbnail=True,
        )
        assert len(p_num) == 1
        image_tensor = model.process_images(tiles, model.config).to(
            dtype=torch.float16, device="cuda"
        )

        conv = conv_templates[args.model_type].copy()
        conv.append_message(conv.roles[0],
                            DEFAULT_IMAGE_TOKEN * p_num[0] + "\n" + question)
        conv.append_message(conv.roles[1], None)
        # get_prompt takes the modality and asserts on it when the messages
        # contain <image>: it selects between two different system prompts
        # (conversation.py:137). Calling it bare raises an AssertionError with
        # no message, which is easy to misread as a tokenizer problem.
        prompt = conv.get_prompt("image")

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()

        # num_return_sequences draws the whole group in one forward pass and
        # reuses the encoded tiles across it, which is the point of doing the
        # vision work once.
        with torch.inference_mode():
            out = model.generate(
                input_ids,
                images=image_tensor,
                audios=None,
                do_sample=True,
                temperature=args.temperature,
                top_p=0.95,
                num_beams=1,
                num_return_sequences=args.group_size,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
        seqs = out if isinstance(out, torch.Tensor) else out.sequences
        return [t.strip() for t in tokenizer.batch_decode(seqs, skip_special_tokens=True)]

    for path in args.parquet:
        if len(records) >= args.limit:
            break
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=32):
            if len(records) >= args.limit or stats["prompts"] >= args.max_prompts:
                break
            for row in batch.to_pylist():
                if len(records) >= args.limit:
                    break
                stats["prompts"] += 1
                if (stats["prompts"] - 1) % args.num_shards != args.shard:
                    continue
                question = (row.get("question") or "").strip()
                if not question:
                    continue
                try:
                    raw = row["image"]["bytes"] if isinstance(row["image"], dict) else row["image"]
                    image = Image.open(io.BytesIO(raw)).convert("RGB")
                except Exception:
                    stats["bad_image"] += 1
                    continue

                try:
                    responses = sample_group(question, image)
                except Exception as exc:  # noqa: BLE001
                    stats["sample_failed"] += 1
                    if stats["sample_failed"] <= 3:
                        print(f"  sampling failed: {type(exc).__name__}: {exc}")
                    continue

                responses = [r for r in responses if len(r.strip()) >= 4]
                if len(responses) < 2:
                    stats["too_short"] += 1
                    continue

                reference = (row.get("chosen") or "").strip()
                if not reference:
                    stats["no_reference"] += 1
                    continue

                contested = contested_terms(responses)
                scored = sorted(
                    ((ranking_score(r, reference, contested, args.contested_weight), r)
                     for r in responses),
                    key=lambda x: x[0], reverse=True,
                )
                (best_s, chosen), (worst_s, rejected) = scored[0], scored[-1]

                # No usable preference direction: the best and worst samples
                # match the reference about equally well, so which one is
                # "better" is arbitrary. Training on it contributes -log(0.5)
                # and adds noise to the gradient.
                if best_s - worst_s < args.min_margin:
                    stats["flat"] += 1
                    continue
                if content_tokens(chosen) == content_tokens(rejected):
                    stats["identical"] += 1
                    continue

                digest = hashlib.sha1(raw).hexdigest()
                fname = seen_images.get(digest)
                if fname is None:
                    fname = f"{digest}.jpg"
                    dest = os.path.join(img_dir, fname)
                    if not os.path.exists(dest):
                        image.save(dest, "JPEG", quality=95)
                    seen_images[digest] = fname

                records.append({
                    "set": "selfsample",
                    "id": f"ss_{len(records):06d}",
                    "conversations": [
                        {"from": "human", "value": "<image>\n" + question},
                        {"from": "gpt", "value": chosen},
                    ],
                    "rejected": rejected,
                    "image": fname,
                    "sim_chosen": round(best_s, 3),
                    "sim_rejected": round(worst_s, 3),
                })
                stats["kept"] += 1
                if len(records) % 50 == 0:
                    print(f"  {len(records)} pairs "
                          f"(from {stats['prompts']} prompts, "
                          f"{stats['flat']} flat)")

    out_json = os.path.join(args.out_dir, "rlaif_v_train.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=1)

    print(f"\nwrote {len(records)} pairs -> {out_json}")
    print(f"unique images: {len(seen_images)}")
    print("stats:", dict(stats))
    if records:
        import statistics as st
        cl = [len(r["conversations"][1]["value"]) for r in records]
        rl = [len(r["rejected"]) for r in records]
        longer = sum(1 for a, b in zip(cl, rl) if a > b)
        print(f"mean chars: chosen={st.mean(cl):.0f} rejected={st.mean(rl):.0f}")
        print(f"chosen longer in {longer/len(records):.1%} "
              f"(50% = no length bias; a large deviation means DPO can cheat)")


if __name__ == "__main__":
    main()
