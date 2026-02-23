# ProtoCBM
This is the codebase for the project seminar "Project Lab Multimodal Artificial Intelligence" at the technical University Darmstad. 

## Getting Started
The environment can be installed using the following command.
```
conda env create --name protocbm --file=environment.yml
```

## Dataset
The following chapter explains how to download and store the Training and validation data.

Shoud you want to store the data in a different location please modify the `BASE_DIR` variable in the `config.py` file.

### Training Data
To train the model we use the CUB_200_2011 dataset [[1]](#1). In order to train our model we use the download link provided from the GitHub repo of [[2]](#2). 

The dataset provided by the authors of [[2]](#2) is separated into two components. Where the first one is called `CUB_proecessed`. This part contains pickel files with information about the training and testing split of their model. To ensure similar results we use the provided split. The second part of the provided data is the `CUB_Dataset`. It contains the original images, metadata information (part positions, labels, etc.). Again to ensure reproducability of the different training runs we use the provided dataset instead of downloading it from the original website.

The data can be dowloaded from the following [Link](https://worksheets.codalab.org/worksheets/0x362911581fcd4e048ddfd84f47203fd2).


After downloading and uncompressig the files have to be stored in the following locations. 
- The `processed_data` should be stored in `./data/CUB_processed/`
- The `CUB_200_2011` folder shoudl be stored in `./data/CUB_200_2011`


### Evaluation Data
For the evaluation we use the part annotations introduced in [[3]](#3).

``` bash
wget https://github.com/hamedbehzadi/CUB70-PartSegmentationDataset/raw/main/AnnotationMasksPerclass.tar.xz
```

After downloading the files shoul be extracted to CUB data directory in the part_segmentation folder.

```
mkdir -p data/CUB_200_2011/part_segmentations/ &&
tar -xJf AnnotationMasksPerclass.tar.xz -C data/CUB_200_2011/part_segmentations/
```

Waterbirds: ```https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz```

#### Out of Distribution Data (SUB Benchmark)

In order to evaluate the concept accuracy of our model we further test it on the SUB Benchmark [[4]](#4). The SUB dataset contains synthetic images of birds with substituted visual attributes (e.g. a bird whose breast color has been changed from white to red). This tests whether concept-based models can detect attribute changes rather than simply memorizing class-attribute associations.

The dataset is hosted on HuggingFace: https://huggingface.co/datasets/Jessica-bader/SUB

It will be **downloaded automatically** the first time you run a SUB evaluation script and saved locally to `data/SUB/`. To use a pre-downloaded copy, set the `sub_data_dir` config parameter.

##### Config Setup

Add the following parameters to your evaluation config (e.g. `configs/eval_protocbm.yaml` or `configs/eval_cbm_sub.yaml`):

```yaml
# SUB Dataset settings
sub_data_dir: data/SUB          # Path to local SUB dataset
sub_limit: null                 # Optional: limit number of samples (for debugging)
use_majority_voting: true       # Denoise ground truth via majority voting
save_majority_csv: false        # Save majority-voted attributes to CSV
```


## Train the Model

The repository contains trainings scripts for two distinct models. Namely the `CBM` and the `ProtoCBM` models. For trainings these models we need the respective script `train_protocbm.py` or `train_cbm.py` and the correct config file.

### Training with Distance Loss

To train the ProtoCBM model with the localization distance loss, add the following parameters to your config file (e.g. `configs/protocbm.yaml`):

```yaml
distance_loss: true
distance_loss_weight: 0.1
```

This enables a loss term that penalizes the distance between predicted attention map locations and ground-truth part keypoints from the CUB dataset. The `distance_loss_weight` controls the relative weight of this term in the total loss.


## Evaluate the Model

The repository contains different evaluation methods to test the effectiveness of the trained model.

### Evaluate on CUB Test Set (`eval.py`)

The main evaluation script measures classification accuracy, attribute accuracy, part segmentation IoU, and keypoint localization distance on the CUB test set.

#### Joint / Standard Models

```bash
CONFIG=configs/protocbm/eval/eval_proto_final.yaml SCRIPT=eval.py sbatch run.slurm
```

#### Independent Models

Independent CBM models use two separate checkpoints: an XC model (image → concept scores) and a CY model (concept scores → class predictions). Use the provided config:

```bash
CONFIG=configs/protocbm/eval/eval_independent.yaml SCRIPT=eval.py sbatch run.slurm
```

The key parameters in `configs/protocbm/eval/eval_independent.yaml`:

```yaml
mode: independent

# Paths to the two checkpoints
xc_checkpoint: weights/protoCBM-models/independent/xc_sigmoid.pth
cy_checkpoint: weights/protoCBM-models/independent/cy_sigmoid.pth
```

### Evaluate on SUB-Benchmark

See the [Out of Distribution Data (SUB Benchmark)](#out-of-distribution-data-sub-benchmark) section under Dataset for setup instructions and the full list of evaluation scripts.


## References
<a id="1">[1]</a> 
Wah, C., Branson, S., Welinder, P., Perona, P., & Belongie, S. (2011). The caltech-ucsd birds-200-2011 dataset.  
<a id="2">[2]</a> Koh, P. W., Nguyen, T., Tang, Y. S., Mussmann, S., Pierson, E., Kim, B., & Liang, P. (2020, November). Concept bottleneck models. In International conference on machine learning (pp. 5338-5348). PMLR.   
<a id="3">[3]</a> Behzadi-Khormouji, H., & Oramas, J. (2023). A protocol for evaluating model interpretation methods from visual explanations. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 1421-1429).

<a id="4">[4]</a> Bader, J., Girrbach, L., Alaniz, S., & Akata, Z. (2025). Sub: Benchmarking cbm generalization via synthetic attribute substitutions. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 23188-23198).
