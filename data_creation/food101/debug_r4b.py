"""Qwen2.5-VL smoke test — a few images per class with stage-1 prompt.

Verifies the pipeline works end-to-end before re-submitting the full 500-image
pilot. Run under SLURM (debug_r4b.slurm — retained name for continuity).

TODO: Add a stage-0 bad-image filter before stage-1. Food-101 contains
collages, logos, and menu photos (e.g. donuts/100076.jpg is a 3-panel
collage with a brand logo) that produce meta-descriptions instead of
dish predicates. A cheap yes/no VLM call ("Is this a single photograph
of one prepared dish?") should drop these before predicate extraction.
"""
import os
import sys

sys.path.insert(0, "data_creation/food101")

from extract_descriptions import load_qwen_vl, run_one
from prompts import STAGE1_PROMPT


IMAGES_PER_CLASS = 3

SAMPLES = {
    "pizza":         ["1008104.jpg", "1001116.jpg", "1008144.jpg"],
    "sushi":         ["100332.jpg",  "1005352.jpg", "1012499.jpg"],
    "donuts":        ["100576.jpg",  "1006079.jpg", "1007399.jpg"],
    "caesar_salad":  ["1000016.jpg", "1000435.jpg", "1011441.jpg"],
    "chicken_curry": ["1004867.jpg", "1014843.jpg", "101833.jpg"],
    "hamburger":     ["100057.jpg",  "100517.jpg",  "100719.jpg"],
}


def test(model, processor, device, img_path, prompt, label):
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    print(f"img:    {img_path}")
    print(f"prompt: {prompt!r}")
    raw = run_one(model, processor, device, img_path, prompt, max_new_tokens=512)
    print("\n--- raw response ---")
    print(raw)


def main():
    model, processor, device = load_qwen_vl("Qwen/Qwen2.5-VL-7B-Instruct", "auto")
    print(f"Loaded on {device}, param dtype: {next(model.parameters()).dtype}")

    root = "/mnt/beegfs/hdd/mirror/home/mo81doja/ProtoCBM/data/food-101/images"
    test_idx = 1
    for cls, files in SAMPLES.items():
        for fname in files[:IMAGES_PER_CLASS]:
            rel = f"{cls}/{fname}"
            path = os.path.join(root, rel)
            label = f"TEST {test_idx}: {cls} ({fname}) + stage-1 prompt"
            test_idx += 1
            if not os.path.isfile(path):
                print(f"(skip {rel}: not found)")
                continue
            test(model, processor, device, path, STAGE1_PROMPT, label)


if __name__ == "__main__":
    main()
