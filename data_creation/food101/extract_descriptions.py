"""Stage 1: extract 8 visual-property phrases per image via Qwen2.5-VL.

Reads a manifest JSONL (produced by sample_pilot.py), runs
Qwen2.5-VL-7B-Instruct on each image with the Stage-1 prompt, parses the
numbered-list response, and appends one JSONL record per image to the output
file. Resumes by skipping ids already present in the output.

Usage:
    python data_creation/food101/extract_descriptions.py \
        --manifest data/food-101/pilot/manifest.jsonl \
        --out data/food-101/pilot/phrases.jsonl \
        --checkpoint_every 25 \
        --limit -1
"""
import argparse
import json
import os
import sys

import torch
from tqdm import tqdm

from prompts import STAGE1_PROMPT, N_PHRASES, parse_phrases


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--checkpoint_every", type=int, default=25,
                   help="Flush output file every N records.")
    p.add_argument("--limit", type=int, default=-1,
                   help="Hard cap on number of images (debug). -1 = no cap.")
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "bfloat16", "float16", "float32"])
    return p.parse_args()


def load_manifest(path):
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def load_done_ids(out_path):
    if not os.path.isfile(out_path):
        return set()
    done = set()
    with open(out_path) as f:
        for ln in f:
            try:
                rec = json.loads(ln)
                done.add(rec["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def load_qwen_vl(model_name, dtype_str):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    if dtype_str == "auto":
        dtype = "auto"
    else:
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}[dtype_str]

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, dtype=dtype, device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)
    device = next(model.parameters()).device
    return model, processor, device


def run_one(model, processor, device, img_path, prompt, max_new_tokens):
    from qwen_vl_utils import process_vision_info

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{img_path}"},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]


def main():
    args = parse_args()

    manifest = load_manifest(args.manifest)
    if args.limit > 0:
        manifest = manifest[: args.limit]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = load_done_ids(args.out)
    todo = [r for r in manifest if r["id"] not in done]
    print(f"Manifest: {len(manifest)} | already done: {len(done)} | todo: {len(todo)}",
          flush=True)
    if not todo:
        print("Nothing to do.")
        return

    model, processor, device = load_qwen_vl(args.model, args.dtype)
    print(f"Loaded {args.model} on {device} "
          f"(param dtype: {next(model.parameters()).dtype})", flush=True)

    n_since_flush = 0
    n_parse_fail = 0
    with open(args.out, "a") as fout:
        for rec in tqdm(todo, desc="Qwen2.5-VL"):
            try:
                raw = run_one(
                    model, processor, device, rec["img_path"],
                    STAGE1_PROMPT, args.max_new_tokens,
                )
            except Exception as e:
                import traceback
                print(f"generate error on id={rec['id']}: {e}", file=sys.stderr)
                if rec["id"] < 3:  # full TB for first few failures
                    traceback.print_exc()
                continue

            phrases = parse_phrases(raw, n=N_PHRASES)
            if len(phrases) < N_PHRASES:
                n_parse_fail += 1

            out_rec = {
                "id": rec["id"],
                "class": rec["class"],
                "class_id": rec["class_id"],
                "img_path": rec["img_path"],
                "raw_response": raw,
                "phrases": phrases,
            }
            fout.write(json.dumps(out_rec) + "\n")
            n_since_flush += 1
            if n_since_flush >= args.checkpoint_every:
                fout.flush()
                os.fsync(fout.fileno())
                n_since_flush = 0

    print(f"Done. Parse-failures (<{N_PHRASES} phrases): {n_parse_fail} / {len(todo)}")


if __name__ == "__main__":
    main()
