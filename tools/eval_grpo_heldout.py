"""Compare two checkpoints on held-out GRPO prompts by what they generate.

POPE/MMBench are discriminative (yes-no / multiple choice) and barely move
when a policy shifts a little; this script measures the thing GRPO actually
trains -- free-form generation on the training distribution -- on prompts the
trainer never saw. Three numbers per model, plus a paired win rate:

  keyword_recall  fraction of the gold answer's content words mentioned
                  (same mining as the keyword reward, but *content only*:
                  no length/format terms that saturate)
  judge_score     JudgeReward against the gold answer (needs --judge)
  win_rate        per-prompt comparison of after vs before on keyword
                  recall (judge as tiebreak when available), with a
                  bootstrap 95% CI

Usage:
    python tools/eval_grpo_heldout.py \
        --before /path/to/VITA-1.5 \
        --after  /path/to/VITA-1.5-grpo \
        --data   /path/to/rlaif_v_grpo_eval.json \
        --image-root /path/to/images \
        --judge  /path/to/Qwen2.5-3B-Instruct \
        --out    /path/to/heldout_results.json
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_all(model_path, records, image_root, max_new_tokens, device):
    """Greedy-decode a response for every record; returns list[str].

    Uses the training-time input format (one square image, one <image>
    token) rather than VLMEvalKit's dynamic tiling, so the eval probes
    exactly the distribution GRPO trained on. The generate call itself
    mirrors VLMEvalKit's vita_qwen2 wrapper: VITA's generate requires the
    audio dummies and returns only the new tokens, usually prefixed with a
    state token (one of the three reply-format glyphs).
    """
    import torch
    from PIL import Image

    from vita.constants import IMAGE_TOKEN_INDEX
    from vita.conversation import conv_templates
    from vita.model.builder import load_pretrained_model
    from vita.util.mm_utils import get_model_name_from_path, tokenizer_image_token

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, get_model_name_from_path(model_path),
        model_type="qwen2p5_instruct", device_map=device,
    )
    # Same as the VLMEvalKit wrapper: parts of the audio encoder (buffers)
    # stay fp32 after loading and poison the dtype of the dummy audio pass.
    model.get_audio_encoder().to(dtype=torch.float16)
    model.eval()

    # The wrapper loads the model in half precision; match it explicitly
    # (model.dtype can report float32 when any buffer stayed fp32).
    audios = {
        "audios": torch.zeros(1, 400, 80).half().to(model.device),
        "lengths": torch.tensor([400.0]).half().to(model.device),
        "lengths_for_llm": torch.tensor([60], device=model.device),
    }

    outs = []
    for i, rec in enumerate(records):
        q = rec["conversations"][0]["value"]
        conv = conv_templates["qwen2p5_instruct"].copy()
        conv.append_message(conv.roles[0], q)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt("image")

        image = Image.open(os.path.join(image_root, rec["image"])).convert("RGB")
        image_tensor = image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ].half().to(model.device)

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(model.device)

        with torch.inference_mode():
            out = model.generate(
                input_ids,
                images=image_tensor,
                audios=audios,
                sf_masks=torch.tensor([0], device=model.device),
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        text = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        if text[:1] in ("\u261e", "\u261c", "\u261f"):
            text = text[1:]
        outs.append(text.strip())
        if (i + 1) % 50 == 0:
            print(f"  {model_path}: {i + 1}/{len(records)}", flush=True)

    del model
    torch.cuda.empty_cache()
    return outs


def keyword_recall(response, keywords):
    low = response.lower()
    if not keywords:
        return 0.0
    return sum(1 for k in keywords if k in low) / len(keywords)


def bootstrap_ci(wins, n=10000, seed=0):
    rng = random.Random(seed)
    if not wins:
        return 0.5, 0.5, 0.5
    m = len(wins)
    stats = sorted(
        sum(rng.choices(wins, k=m)) / m for _ in range(n)
    )
    return sum(wins) / m, stats[int(0.025 * n)], stats[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument(
        "--after",
        default=None,
        help="omit (or pass the same path as --before) for a baseline-only "
        "run: generates once, reports absolute metrics, no comparison",
    )
    ap.add_argument("--data", required=True, help="held-out prompt json")
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--judge", default=None, help="judge model path (optional)")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0, help="cap prompts (0 = all)")
    ap.add_argument("--out", default=None, help="write per-sample results json")
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as fh:
        records = json.load(fh)
    if args.limit:
        records = records[: args.limit]
    print(f"{len(records)} held-out prompts")

    baseline_only = not args.after or args.after == args.before
    gen_b = generate_all(args.before, records, args.image_root,
                         args.max_new_tokens, "cuda")
    gen_a = gen_b if baseline_only else generate_all(
        args.after, records, args.image_root, args.max_new_tokens, "cuda")

    judge = None
    if args.judge:
        from vita.train.rewards import JudgeReward
        judge = JudgeReward(args.judge)

    # Verifiable sets (e.g. CLEVR counting) carry reward_meta["answer"];
    # exact-match accuracy is then the primary metric and keyword recall is
    # skipped -- there are no keywords to recall.
    from vita.train.rewards import answer_reward

    verifiable = any(rec.get("reward_meta", {}).get("answer") for rec in records)

    rows, wins = [], []
    for rec, rb, ra in zip(records, gen_b, gen_a):
        meta = rec.get("reward_meta", {})
        kws = meta.get("keywords", [])
        kr_b = keyword_recall(rb, kws)
        kr_a = keyword_recall(ra, kws)
        row = {"id": rec["id"], "kr_before": kr_b, "kr_after": kr_a,
               "before": rb, "after": ra}
        if verifiable:
            row["acc_before"] = answer_reward("", rb, meta)
            row["acc_after"] = answer_reward("", ra, meta)
        if judge is not None:
            q = rec["conversations"][0]["value"].replace("<image>", "").strip()
            row["judge_before"] = judge(q, rb, meta)
            row["judge_after"] = judge(q, ra, meta)
        rows.append(row)

        # Win on accuracy when verifiable, else keyword recall; judge breaks
        # ties when present.
        if verifiable and row["acc_after"] != row["acc_before"]:
            wins.append(1 if row["acc_after"] > row["acc_before"] else 0)
        elif not verifiable and kr_a != kr_b:
            wins.append(1 if kr_a > kr_b else 0)
        elif judge is not None and abs(row.get("judge_after", 0) - row.get("judge_before", 0)) > 1e-6:
            wins.append(1 if row["judge_after"] > row["judge_before"] else 0)
        # exact ties are dropped from win rate (standard practice)

    n = len(rows)
    if verifiable:
        acc_b = sum(r["acc_before"] for r in rows) / n
        acc_a = sum(r["acc_after"] for r in rows) / n
        print(f"\nanswer_accuracy: {acc_b:.4f} -> {acc_a:.4f} "
              f"({acc_a - acc_b:+.4f})")
    kr_b_mean = sum(r["kr_before"] for r in rows) / n
    kr_a_mean = sum(r["kr_after"] for r in rows) / n
    print(f"keyword_recall: {kr_b_mean:.4f} -> {kr_a_mean:.4f} "
          f"({kr_a_mean - kr_b_mean:+.4f})")
    if judge is not None:
        jb = sum(r["judge_before"] for r in rows) / n
        ja = sum(r["judge_after"] for r in rows) / n
        print(f"judge_score:    {jb:.4f} -> {ja:.4f} ({ja - jb:+.4f})")
    if baseline_only:
        print("(baseline-only run: no comparison)")
    else:
        wr, lo, hi = bootstrap_ci(wins)
        print(f"win_rate:       {wr:.3f}  [95% CI {lo:.3f}, {hi:.3f}]  "
              f"(decided on {len(wins)}/{n} prompts)")
        if lo > 0.5:
            print("  -> after is better beyond noise")
        elif hi < 0.5:
            print("  -> after is WORSE beyond noise")
        else:
            print("  -> not separable from noise at n =", n)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=1)
        print(f"per-sample results -> {args.out}")


if __name__ == "__main__":
    main()
