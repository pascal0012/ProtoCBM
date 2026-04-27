"""Stage 0: sample a deterministic pilot subset of Food-101.

Reads data/food-101/meta/train.txt, filters to the requested class list,
takes the first N images per class (by order in train.txt), and writes a
manifest JSONL with one record per image.

Usage:
    python data_creation/food101/sample_pilot.py \
        --food101_root data/food-101 \
        --classes pizza sushi apple_pie caesar_salad donuts \
        --per_class 100 \
        --out data/food-101/pilot/manifest.jsonl
"""
import argparse
import json
import os
import sys
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--food101_root", default="data/food-101")
    p.add_argument("--classes", nargs="+", required=True,
                   help="Class names (must match data/food-101/meta/classes.txt).")
    p.add_argument("--per_class", type=int, default=100)
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--out", required=True)
    return p.parse_args()


def load_class_list(root):
    with open(os.path.join(root, "meta", "classes.txt")) as f:
        return [ln.strip() for ln in f if ln.strip()]


def main():
    args = parse_args()
    all_classes = set(load_class_list(args.food101_root))
    unknown = [c for c in args.classes if c not in all_classes]
    if unknown:
        sys.exit(f"Unknown Food-101 classes: {unknown}")

    class_to_id = {c: i for i, c in enumerate(sorted(args.classes))}

    split_file = os.path.join(args.food101_root, "meta", f"{args.split}.txt")
    by_class = defaultdict(list)
    with open(split_file) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            cls, stem = ln.split("/", 1)
            if cls in class_to_id:
                by_class[cls].append(stem)

    images_dir = os.path.join(args.food101_root, "images")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n_written = 0
    uid = 0
    with open(args.out, "w") as f:
        for cls in sorted(args.classes):
            stems = by_class.get(cls, [])[: args.per_class]
            if len(stems) < args.per_class:
                print(f"WARN: class '{cls}' has only {len(stems)} images in {args.split} "
                      f"(requested {args.per_class})", file=sys.stderr)
            for stem in stems:
                img_path = os.path.abspath(os.path.join(images_dir, cls, f"{stem}.jpg"))
                if not os.path.isfile(img_path):
                    print(f"WARN: missing image {img_path}", file=sys.stderr)
                    continue
                rec = {
                    "id": uid,
                    "class": cls,
                    "class_id": class_to_id[cls],
                    "img_path": img_path,
                }
                f.write(json.dumps(rec) + "\n")
                uid += 1
                n_written += 1

    print(f"Wrote {n_written} records to {args.out}")
    print(f"Classes: {sorted(args.classes)} (id map: {class_to_id})")


if __name__ == "__main__":
    main()
