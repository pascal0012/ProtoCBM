# Food-101 concept dataset pipeline

Produces concept-annotated Food-101 data for ProtoCBM, per the spec in
[docs/food-101.md](../../docs/food-101.md).

**Status**: Stage 0–1 pilot only. Stages 2–5 (clustering, concept eval,
filtering, class selection, localization) are deferred until raw pilot
outputs are reviewed.

## Progress log (most recent first)

### ✅ Qwen2.5-VL smoke test passed (job 47766)

All 3 test images produced clean, coherent output. Stage-1 prompt tests
(pizza, sushi) returned perfect 8-phrase numbered lists that parse cleanly;
simple-prompt test (donuts) returned a coherent English description.

**Next step (next session):**
1. Submit the full pilot — expected ~20–35 min at bf16 on one GPU:
   ```bash
   sbatch slurm/annotate_food101_pilot.slurm
   ```
2. When the job finishes: review `data/food-101/pilot/phrases.jsonl` and
   `data/food-101/pilot/pilot_stats.md` (see "Reviewing pilot quality" below).

### ❌ R-4B pivot (job 47475, 47766 predecessor)

First attempt used `YannQi/R-4B` per the spec. Three sequential issues
unblocked and then blocked the pipeline:

1. **transformers 5.0 tied-weights API mismatch** — R-4B's remote code
   declares `_tied_weights_keys` as a list (old API); transformers 5 expects
   a dict. Worked around with a monkey-patch that rewrites the class
   attribute before `from_pretrained`.
2. **bfloat16 dtype mismatch** in the vision tower. Switched default
   `--dtype` to `float32` (matches R-4B's model card).
3. **Model-card snippet itself returns gibberish** under transformers 5.0
   even with tying confirmed intact (debug job 47750 showed
   `lm_head.weight` and `embed_tokens.weight` loaded as real, distinct
   checkpoint tensors with plausible norms). Diagnosed as a deeper R-4B vs
   transformers 5.0 incompatibility — not fixable without downgrading
   transformers, which risks breaking the rest of the repo.

**Decision**: pivoted from R-4B → `Qwen/Qwen2.5-VL-7B-Instruct`. Native
`Qwen2_5_VLForConditionalGeneration` class in transformers 5.x, no
`trust_remote_code`, no monkey-patches. Requires `qwen-vl-utils[decord]`
(added to [environment.yml](../../environment.yml) and installed at SLURM
prelude).

## Pilot layout (5 classes × 100 images)

Classes: `pizza`, `sushi`, `apple_pie`, `caesar_salad`, `donuts`.

Pipeline (run via SLURM):

```bash
sbatch slurm/annotate_food101_pilot.slurm
```

Or run stages manually:

```bash
# Stage 0 — sample manifest (deterministic, picks first 100/class from train.txt)
python data_creation/food101/sample_pilot.py \
    --food101_root data/food-101 \
    --classes pizza sushi apple_pie caesar_salad donuts \
    --per_class 100 \
    --out data/food-101/pilot/manifest.jsonl

# Stage 1 — Qwen2.5-VL phrase extraction (GPU, ~20–35 min for 500 images)
python data_creation/food101/extract_descriptions.py \
    --manifest data/food-101/pilot/manifest.jsonl \
    --out data/food-101/pilot/phrases.jsonl

# Stats summary (CPU, <1 min)
python data_creation/food101/analyze_pilot.py
```

## Outputs

Under `data/food-101/pilot/`:

- `manifest.jsonl` — one record per image: `{id, class, class_id, img_path}`.
- `phrases.jsonl` — per image: `{id, class, class_id, img_path, raw_response, phrases}`.
  `raw_response` is the full VLM output string (kept for manual debugging of
  parse failures).
- `pilot_stats.md` — markdown report: phrase counts, dedup rate, parse-failure
  rate, per-class top-K phrases.

## Reviewing pilot quality

The stats file flags structural issues (parse failures, very short phrases)
but cannot judge semantics. Before proceeding to Stage 2, manually inspect:

1. ~20 random lines of `phrases.jsonl`, cross-referenced with the image.
2. Per-class top-20 phrase lists in `pilot_stats.md` — these should look like
   plausible ingredient/texture/color words, *not* the dish name.
3. Parse-failure rate in `pilot_stats.md` — target < 5%. If higher, revisit
   the prompt in [prompts.py](prompts.py) or the regex in `parse_phrases`.

## VLM choice

Uses `Qwen/Qwen2.5-VL-7B-Instruct` via the native
`Qwen2_5_VLForConditionalGeneration` class (no `trust_remote_code`). R-4B was
tried first but produces gibberish under transformers 5.0 — see the
"Progress log" above for the diagnostic trail. The spec's R-4B recommendation
is kept as a documentation reference only.

## Files

| File | Purpose |
| ---- | ------- |
| `sample_pilot.py` | Stage 0: build manifest from `data/food-101/meta/train.txt`. |
| `extract_descriptions.py` | Stage 1: run Qwen2.5-VL, parse 8 phrases per image. |
| `prompts.py` | Single-source prompt + parser (imported by Stage 1 + analyze). |
| `analyze_pilot.py` | Compute stats, write markdown report. |
| `debug_r4b.py` | 3-image smoke test (retained name for continuity). |
| `debug_tied.py` | One-off R-4B tied-weights diagnostic (can be deleted). |
