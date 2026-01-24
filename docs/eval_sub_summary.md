# SUB Benchmark Evaluation Script

Evaluates trained concept-based models (like ProtoCBM) on the **SUB (Synthetic Attribute Substitutions) benchmark dataset** to test visual concept learning.

## eval_sub_attributes.py

**Purpose:** Evaluates **attribute prediction accuracy** to test whether models learn true visual concepts or just memorize class-attribute correlations.

**What it does:**
1. Loads a trained model and the SUB dataset
2. For each image where an attribute was substituted (e.g., a Cardinal with blue breast instead of red):
   - **New attribute detection**: Does the model predict the substituted attribute as PRESENT?
   - **Original attribute detection**: Does the model predict the original attribute as ABSENT?
3. Computes:
   - **Adaptation rate** - model correctly sees new=present AND original=absent (true visual understanding)
   - **Memorization rate** - model sees new=absent AND original=present (relies on class priors)
   - Per-attribute breakdowns
4. Generates visualization grids showing model predictions on SUB images

**Key question answered:** *Does the model actually see visual features, or does it just predict attributes based on which bird species it thinks it's looking at?*

---

## Config Options

| Option | Default | Description |
|--------|---------|-------------|
| `sub_data_dir` | `data/SUB` | Path to SUB dataset |
| `sub_limit` | `None` | Optionally limit number of samples for quick testing |
| `use_majority_voting` | `false` | Denoise ground-truth attributes using majority voting per class |
| `save_majority_csv` | `false` | Save majority-voted attributes to CSV file |
| `n_vis_images` | `10` | Number of images per visualization grid |
| `n_vis_grids` | `10` | Number of visualization grids to generate |

---

## Majority Voting (Attribute Denoising)

The CUB dataset attribute annotations can be noisy (inconsistent labels for the same class). The `use_majority_voting` option applies class-level denoising:

- For each class, count how many samples have each attribute
- If **>50%** of samples have the attribute → set to **1** for all samples
- If **≤50%** of samples have the attribute → set to **0** for all samples

This reduces noise from inconsistent per-image annotations and provides cleaner ground-truth for evaluation.

---

## Output Files

All outputs are saved to `{log_dir}/eval_sub/`:

| File/Folder | Description |
|-------------|-------------|
| `sub_attribute_eval.txt` | Main results with metrics and per-attribute breakdown |
| `majority_voted_attributes.csv` | (Optional) Denoised attributes if `save_majority_csv: true` |
| `prediction_grids/` | Visualization grids showing model predictions |

### Visualization Grids

The script generates image grids showing:
- The SUB image with substituted attribute
- Bird class name and attribute change (e.g., "breast color: red → blue")
- Model predictions for old and new attributes with probabilities
- Color-coded boxes:
  - **Green**: Model adapts (correctly predicts new=present, old=absent)
  - **Red**: Model memorizes (incorrectly predicts new=absent, old=present)
  - **Yellow**: Mixed results

---

## Example Config

```yaml
# SUB Dataset settings
sub_data_dir: data/SUB
sub_limit: null              # Set to integer for quick testing
use_majority_voting: true    # Enable denoising
save_majority_csv: true      # Export to {log_dir}/eval_sub/majority_voted_attributes.csv

# Visualization settings
n_vis_images: 10             # Images per grid
n_vis_grids: 10              # Number of grids to generate
```

---

## Metrics Explained

### Main Metrics

1. **New Attribute Detection Accuracy**: How often the model predicts the substituted attribute as present. High accuracy indicates the model responds to visual evidence.

2. **Original Attribute Hallucination Rate**: How often the model still predicts the original (now absent) attribute as present. Low rate indicates the model doesn't rely on class priors.

### Adaptation vs Memorization Analysis

| Category | Condition | Interpretation |
|----------|-----------|----------------|
| **Adapts** | new=1, original=0 | Model correctly recognizes the visual change |
| **Memorizes** | new=0, original=1 | Model ignores visual evidence, relies on class priors |
| **Mixed** | Other combinations | Model partially adapts |

A model that truly learns visual concepts should have high adaptation rate and low memorization rate.