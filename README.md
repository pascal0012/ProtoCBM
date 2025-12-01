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
The training data can be 
curl -L -o ~/Downloads/cub2002011.zip\
https://worksheets.codalab.org/bundles/0xd013a7ba2e88481bbc07e787f73109f5

Additionally to the CUB_200_2011 dataset we use the preprocessed training data from the CUB paper. The processed data can be downloaded using the following link.
https://worksheets.codalab.org/bundles/0x5b9d528d2101418b87212db92fea6683

The `processed_data` should be stored in the following directory: `./data/CUB_processed/`


### Evaluation Data
For the evaluation we use the part annotations introduced in [A Protocol for Evaluating Model Interpretation Methods from Visual Explanations](https://openaccess.thecvf.com/content/WACV2023/papers/Behzadi-Khormouji_A_Protocol_for_Evaluating_Model_Interpretation_Methods_From_Visual_Explanations_WACV_2023_paper.pdf).

``` bash
wget https://github.com/hamedbehzadi/CUB70-PartSegmentationDataset/raw/main/AnnotationMasksPerclass.tar.xz
```

## Execution


## BibTeX Citation
```
@inproceedings{behzadi2023protocol,
  title     = {A protocol for evaluating model interpretation methods 
               from visual explanations},
  author    = {Behzadi-Khormouji, Hamed and Oramas, Jos{\'e}},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on 
               Applications of Computer Vision},
  pages     = {1421--1429},
  year      = {2023}
}
```