# Heatmap Visualization Guide

This guide explains how to use the single-image visualization script and the changes made to the aggregated heatmap visualization.

## Changes to `eval_comparison.py`

### Direct Checkpoint Path Support

The `eval_comparison.py` script now supports loading models from a direct checkpoint path instead of requiring a specific directory structure.

### Usage

You can now specify the checkpoint path directly in your config YAML file:

```yaml
# Option 1: Direct path to checkpoint file
checkpoint: "path/to/your/model_checkpoint.pth"

# Option 2: Leave empty to auto-load from log_dir based on seed
checkpoint: ""  # Will load {log_dir}/best_model_{seed}.pth
```

### Example Config

```yaml
log_dir: outputs/ProtoCBM/models
checkpoint: "outputs/ProtoCBM/models/my_custom_model.pth"  # Direct path
seed: 1  # Only used if checkpoint is empty
```

If the checkpoint path is specified and the file exists, it will be used. Otherwise, the script falls back to the default behavior of loading `{log_dir}/best_model_{seed}.pth`.

## New Script: `visualize_single_image.py`

This script allows you to visualize heatmaps for a single image using both localization methods (argmax and aggregation).

### Usage

```bash
python visualize_single_image.py --image_path <path_to_image> --config <path_to_config>
```

### Arguments

- `--image_path` (required): Path to the input image you want to visualize
- `--config` (optional): Path to the YAML config file (default: `configs/eval_protocbm_comparison.yaml`)
- `--output_dir` (optional): Directory to save visualizations (default: `./outputs/single_image_viz`)

### Example

```bash
python visualize_single_image.py \
    --image_path data/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg \
    --config configs/eval_protocbm_comparison.yaml \
    --output_dir outputs/my_viz
```

### Output Structure

The script creates the following directory structure:

```
output_dir/
├── argmax_method/
│   ├── b0_id0_part_viz.png                 # Keypoint visualization
│   └── individual_maps/
│       ├── body_part_1/
│       │   ├── image_attribute1_scaled.png
│       │   ├── image_attribute1_original.png
│       │   └── ...
│       └── body_part_2/
│           └── ...
└── aggregated_method/
    ├── b0_id0_part_viz.png                 # Keypoint visualization
    └── individual_maps/
        ├── body_part_1/
        │   ├── image_bodypart1_aggregated_scaled.png
        │   └── image_bodypart1_aggregated_original.png
        └── body_part_2/
            └── ...
```

## Changes to `save_aggregated_activation_maps`

The `save_aggregated_activation_maps` function in [localization/visualise.py](localization/visualise.py) has been modified to support raw heatmap values without automatic rescaling.

### New Parameter: `normalize_heatmaps`

- **`normalize_heatmaps=False`** (recommended): Uses raw heatmap values without rescaling. This shows the actual aggregated activation values, making it easier to compare across different body parts.
- **`normalize_heatmaps=True`**: Normalizes each heatmap individually to [0, 1] range. This was the previous default behavior.

### Key Changes

1. **Raw values by default**: When `normalize_heatmaps=False`, the heatmaps are displayed with their actual values, not normalized per-heatmap.
2. **Colorbar added**: The original-size heatmap plots now include a colorbar showing the value range.
3. **Fixed vmin**: When using raw values, `vmin=0` is set to ensure consistent scaling.
4. **Updated titles**: Plot titles indicate whether raw values or normalized values are being used.

### Why This Matters

The issue with automatic per-heatmap normalization is that it can mask differences in activation strength between body parts. For example:

- Body part A might have aggregated values ranging from [0, 100]
- Body part B might have aggregated values ranging from [0, 10]

With per-heatmap normalization, both would be displayed with the same color intensity, making them appear equally activated. With raw values, you can see that body part A is actually 10x more activated than body part B.

### Cache Considerations

If you're seeing unexpected results, the issue might be related to cached heatmaps. The raw value visualization should help diagnose this:

- If all heatmaps have similar value ranges, the cache is likely working correctly
- If some heatmaps have drastically different ranges (e.g., some in [0, 1] and others in [0, 100]), this could indicate a caching issue where some values are from normalized versions and others are raw

## Updated Files

1. **[visualize_single_image.py](visualize_single_image.py)**: New script for single-image visualization
2. **[localization/visualise.py](localization/visualise.py)**: Modified `save_aggregated_activation_maps` function
3. **[eval_comparison.py](eval_comparison.py)**: Updated to use `normalize_heatmaps=False` for raw value visualization


### test images
- data/CUB_200_2011/images/020.Yellow_breasted_Chat/Yellow_Breasted_Chat_0026_21845.jpg
- data/CUB_200_2011/images/152.Blue_headed_Vireo/Blue_Headed_Vireo_0019_156311.jpg
- data/CUB_200_2011/images/158.Bay_breasted_Warbler/Bay_Breasted_Warbler_0020_159737.jpg
- data/CUB_200_2011/images/196.House_Wren/House_Wren_0042_187098.jpg
- data/CUB_200_2011/images/200.Common_Yellowthroat/Common_Yellowthroat_0125_190902.jpg