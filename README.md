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

### Training Datat
To train the model we use the CUB_200_2011 dataset [[1]](#1). In order to train our model we use the download link provided from the GitHub repo of [[2]](#2). 

The dataset provided by the authors of [[2]](#2) is separated into two components. Where the first one is called `CUB_proecessed`. This part contains pickel files with information about the training and testing split of their model. To ensure similar results we also use the provided split. The second part of the provided data is the `CUB_Dataset`. It contains the original image, metadata information (part positions, labels, etc.). Again to ensure equality of the different training runs we also use the provided dataset instead of downloading it from the Caltech website.

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

#### Out of Distribution Data
In order to evaluate the concept accuracy of our model we further test it on the SUB Benchmark [[4]](#4). The dataset cont 

Link to the dataset:
https://huggingface.co/datasets/Jessica-bader/SUB


To run the evaluation on the SUB dataset we first have to add some new parameters to the configuration file (if not already included).



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



### Evaluate on SUB-Benchmark

To evaluate either ProtoCBM or vanilla CBM on the SUB benchmark either create a new config or modify an existing config with the following parameters.

```yaml
sub_data_dir: data/SUB
sub_limit: null
use_majority_voting: true
save_majority_csv: false
```

After setting up the config the benchmark can be run with the following script:

```bash
python eval_sub_attributes.py --config configs/protocbm.yaml
```


## References
<a id="1">[1]</a> 
Wah, C., Branson, S., Welinder, P., Perona, P., & Belongie, S. (2011). The caltech-ucsd birds-200-2011 dataset.  
<a id="2">[2]</a> Koh, P. W., Nguyen, T., Tang, Y. S., Mussmann, S., Pierson, E., Kim, B., & Liang, P. (2020, November). Concept bottleneck models. In International conference on machine learning (pp. 5338-5348). PMLR.   
<a id="3">[3]</a> Behzadi-Khormouji, H., & Oramas, J. (2023). A protocol for evaluating model interpretation methods from visual explanations. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 1421-1429).

<a id="4">[4]</a> Bader, J., Girrbach, L., Alaniz, S., & Akata, Z. (2025). Sub: Benchmarking cbm generalization via synthetic attribute substitutions. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 23188-23198).
