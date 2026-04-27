"""Quick summary stats for the Stage-1 pilot output.

Writes a human-readable markdown report alongside phrases.jsonl. Not a
substitute for eyeballing raw VLM outputs — the report flags issues
(parse failures, suspiciously short phrases) but cannot judge semantics.
"""
import argparse
import json
import os
from collections import Counter, defaultdict
from statistics import mean, median

from prompts import N_PHRASES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phrases", default="data/food-101/pilot/phrases.jsonl")
    p.add_argument("--out", default="data/food-101/pilot/pilot_stats.md")
    p.add_argument("--top_k", type=int, default=20,
                   help="Top-K most-common phrases per class.")
    return p.parse_args()


def load_records(path):
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def main():
    args = parse_args()
    recs = load_records(args.phrases)
    if not recs:
        raise SystemExit(f"No records in {args.phrases}")

    per_class = defaultdict(list)
    for r in recs:
        per_class[r["class"]].append(r)

    all_phrases = []
    all_lengths = []
    parse_fail = 0
    per_class_counters: dict[str, Counter] = {}
    for cls, cls_recs in per_class.items():
        c = Counter()
        for r in cls_recs:
            ps = [p.lower().strip() for p in r.get("phrases", [])]
            if len(ps) < N_PHRASES:
                parse_fail += 1
            c.update(ps)
            all_phrases.extend(ps)
            all_lengths.extend(len(p.split()) for p in ps)
        per_class_counters[cls] = c

    n = len(recs)
    n_phrases = len(all_phrases)
    n_unique = len(set(all_phrases))
    dedup_rate = 1 - n_unique / n_phrases if n_phrases else 0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# Food-101 pilot — Stage 1 stats\n\n")
        f.write(f"Source: `{args.phrases}`\n\n")
        f.write("## Overall\n\n")
        f.write(f"- Images: **{n}**\n")
        f.write(f"- Total phrases: **{n_phrases}**\n")
        f.write(f"- Unique phrases (case-insensitive): **{n_unique}**\n")
        f.write(f"- Dedup rate: **{dedup_rate:.1%}**\n")
        f.write(f"- Phrase word-length — mean: {mean(all_lengths):.2f}, "
                f"median: {median(all_lengths):.0f}, "
                f"min: {min(all_lengths)}, max: {max(all_lengths)}\n")
        f.write(f"- Parse failures (images with < {N_PHRASES} phrases): "
                f"**{parse_fail} / {n}** ({parse_fail / n:.1%})\n\n")

        f.write("## Per-class breakdown\n\n")
        for cls in sorted(per_class):
            cls_recs = per_class[cls]
            cnt = per_class_counters[cls]
            n_cls = len(cls_recs)
            n_phr_cls = sum(cnt.values())
            n_uniq_cls = len(cnt)
            f.write(f"### `{cls}` — {n_cls} images, {n_phr_cls} phrases, "
                    f"{n_uniq_cls} unique\n\n")
            f.write(f"Top {args.top_k} phrases:\n\n")
            for phrase, count in cnt.most_common(args.top_k):
                f.write(f"- `{phrase}` ({count})\n")
            f.write("\n")

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
