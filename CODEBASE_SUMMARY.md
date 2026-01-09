# ProtoCBM Codebase Summary

## Project Overview
Research implementation of Concept Bottleneck Models (CBMs) for interpretable bird species classification.

**Task**: Fine-grained classification (200 bird species) with concept-based explanations
**Dataset**: CUB-200-2011 with 112 visual attributes
**Models**: Standard CBM vs ProtoCBM (prototype-based with attention)

---

## Core Concepts

### Training Modes
- **XC**: Image → Concepts only
- **CY**: Concepts → Class only
- **XCY**: Image → Concepts → Class (end-to-end)
- **XY**: Image → Class (baseline, no concepts)

### Key Architecture Components
- **Backbone**: Inception3 or DINO for feature extraction
- **Concept Mapper**:
  - CBMMapper: Simple linear/MLP
  - ProtoMod: Prototype-based with attention maps
- **Classifier**: Final bird species prediction

---

## Directory Structure

```
ProtoCBM/
├── configs/              # YAML experiment configurations
├── models/               # Model architectures (backbones, concept_mapper, model_connector)
├── cub/                  # Dataset handling and preprocessing
├── utils_protocbm/       # Training/eval utilities, mappings
├── localization/         # Localization metrics and visualizations
├── saliency/             # Attention/saliency map generation
├── losses.py             # ProtoModLoss for ProtoCBM
├── train_cbm.py          # Train standard CBM
├── train_protocbm.py     # Train ProtoCBM
├── eval.py               # Standard evaluation
├── eval_comparison.py    # Compare argmax vs aggregated localization
└── data/                 # CUB dataset (not in repo)
```

---

## Key Scripts

### Training
```bash
# Train ProtoCBM
python train_protocbm.py --config configs/protocbm.yaml

# Train standard CBM
python train_cbm.py --config configs/cbm.yaml
```

### Evaluation
```bash
# Standard evaluation
python eval.py --config configs/eval_protocbm.yaml

# Compare localization methods
python eval_comparison.py --config configs/eval_protocbm_comparison.yaml
```

### SLURM Submission
```bash
sbatch run.slurm      # For training
sbatch eval.slurm     # For evaluation
```

---

## Evaluation Features

### Localization Methods Comparison ([eval_comparison.py](eval_comparison.py))

Compares two localization strategies:

**1. Argmax Method** (original)
- Selects highest activated attribute per body part
- Uses that attribute's attention map for localization

**2. Aggregated Method** (new)
- Sums all attention maps per body part
- Finds maximum activation in aggregated map

**Output Structure**:
```
outputs/<log_dir>/comparison_<method>/
├── comparison_eval.txt                  # Statistics
├── argmax_method/
│   ├── b{N}_id{0}_part_viz.png         # Keypoint visualizations
│   └── individual_maps/                 # Per-attribute activation maps
│       ├── beak/
│       │   ├── <img>_bill_color.png
│       │   └── <img>_bill_shape.png
│       ├── head/, wing/, tail/, ...
│       └── (15 body part folders)
└── aggregated_method/
    ├── b{N}_id{0}_part_viz.png         # Keypoint visualizations
    └── individual_maps/                 # Aggregated per-part maps
        ├── beak/
        │   └── <img>_beak_aggregated.png
        ├── head/, wing/, tail/, ...
        └── (15 body part folders)
```

Each visualization shows:
- Left: Image with heatmap overlay
- Right: Pure heatmap

### Visualization Functions ([localization/visualise.py](localization/visualise.py))

**`save_individual_activation_maps()`**
- Saves all attribute activation maps grouped by body part
- Used for argmax method visualization

**`save_aggregated_activation_maps()`**
- Saves aggregated attention per body part
- Used for aggregated method visualization

**`visualize_keypoint_distances()`**
- Shows predicted vs ground-truth keypoint locations
- Displays distance metrics per part

---

## Configuration Parameters

### Essential Settings
```yaml
mode: XCY                    # Training mode
n_attributes: 112            # Number of concepts
concept_mapper: protomod     # "cbm" or "protomod"
backbone: inception          # "inception" or "dino"
batch_size: 64
```

### ProtoCBM Specific
```yaml
proto_n_vectors: 4           # Prototypes per concept
loss_weight_attribute_reg: 0.8886
loss_weight_map_compactness: 0.01
loss_weight_attribute_decorrelation: 0.0088
```

### Evaluation Settings
```yaml
saliency_method: attention   # "attention", "gradcam", etc.
vis_every_n: 5               # Visualize every Nth batch
checkpoint: path/to/model.pth
```

---

## Data Setup

Required data structure:
```
data/
├── CUB_200_2011/
│   ├── images/              # Bird images
│   ├── parts/               # Part keypoint annotations
│   ├── attributes/          # Attribute annotations
│   └── part_segmentations/  # Part masks (for evaluation)
└── CUB_processed/
    └── class_attr_data_10/
        ├── train.pkl
        ├── val.pkl
        └── test.pkl
```

**Attribute-Part Mappings** ([utils_protocbm/mappings.py](utils_protocbm/mappings.py)):
- 112 attributes mapped to 15 anatomical parts
- Parts: beak, head (crown/forehead), neck (nape), body (back/belly/breast/throat), wing, tail, leg, eye

---

## Key Metrics

### Classification
- **Class Accuracy**: Bird species top-1 accuracy
- **Concept Accuracy**: Binary accuracy for 112 attributes

### Localization
- **Keypoint Distance**: Euclidean distance from predicted to ground-truth keypoints
- **Segmentation IoU**: Overlap between attention maps and part segmentation masks

---

## Important Implementation Details

1. **Model Loading**: Set `checkpoint: ""` in config and ensure model file is named `xcy{seed}.pth` in log_dir
2. **Attribute Indexing**: Multiple systems exist (CUB, CBM, part-based) - use mapping utilities
3. **Part Segmentations**: Only available for first 70 bird classes
4. **Random Seeds**: Each run uses specified seed for reproducibility
5. **Configuration-Driven**: All behavior controlled via YAML configs

---

## Loss Function (ProtoCBM)

**ProtoModLoss** combines:
- **L_attribute_reg**: Concept prediction accuracy (MSE)
- **L_cpt**: Compactness of attention maps
- **L_decorrelation**: Encourages prototype diversity (per part group)

---

## Quick Start

```bash
# 1. Setup environment
conda env create -f environment.yml
conda activate protocbm

# 2. Download data (see Data Setup section)

# 3. Train model
python train_protocbm.py --config configs/protocbm.yaml

# 4. Evaluate with comparison
python eval_comparison.py --config configs/eval_protocbm_comparison.yaml

# Or submit to SLURM
sbatch eval.slurm
```

---

## References

1. **Concept Bottleneck Models**: Koh et al. (ICML 2020)
2. **CUB Dataset**: Wah et al. (2011)
3. **Part Segmentation Evaluation**: Behzadi-Khormouji & Oramas (WACV 2023)

---

**Last Updated**: 2026-01-09
**Status**: Active research project
