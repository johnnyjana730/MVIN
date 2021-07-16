# MVIN
MVIN: Learning Multiview Items for Recommendation, SIGIR 2020

This repository is the implementation of MVIN ([arXiv](https://arxiv.org/abs/2005.12516)):
> Chang-You Tai, Meng-Ru Wu, Yun-Wei Chu, Shao-Yu Chu, and Lun-Wei Ku. SIGIR 2020. MVIN: Learning Multiview Items for Recommendation

<img src="https://github.com/johnnyjana730/MVIN/blob/master/img.PNG">

## Introduction
We propose the multi-view item network (MVIN), a GNN-based recommendation model which provides superior recommend-ations bydescribing items from a unique mixed view from user and entity angles.

## Citation 
If you want to use our codes and datasets in your research, please cite:
```
@inproceedings{10.1145/3404835.3462980,
author = {Tai, Chang-You and Huang, Chien-Kun and Huang, Liang-Ying and Ku, Lun-Wei},
title = {Knowledge Based Hyperbolic Propagation},
year = {2021},
isbn = {9781450380379},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3404835.3462980},
doi = {10.1145/3404835.3462980},
booktitle = {Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {1945–1949},
numpages = {5},
keywords = {recommendation, hyperbolic embedding learning, graph neural network, knowledge graph},
location = {Virtual Event, Canada},
series = {SIGIR '21}
}
```
## Files in the folder

- `data/`: datasets
  - `MovieLens-1M/`
  - `amazon-book_20core/`
  - `last-fm_50core/`
- `src/model/`: implementation of MVIN.
- `output/`: storing log files
- `misc/`: storing users being evaluating, popular items, and sharing embeddings.

## Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:
* tensorflow == 1.12.0
* numpy == 1.15.4
* scipy == 1.1.0
* sklearn == 0.20.0

## Build Environment(conda)
```
$ cd MVIN
$ conda deactivate
$ conda env create -f requirements.yml
$ conda activate MVIN
```

## Example to Run the Codes

* MVIN
```
$ cd src/bash/
$ bash main_run.sh "MVIN" $dataset $gpu

```
* other baseline models
```
$ cd src/bash/
$ bash main_run.sh $model $dataset $gpu
```
* some arguments:

* `model`
  * It specifies the type of model.
  * Here we provide four options, including MVIN and five baseline models:
    * `MVIN` (by default), proposed in [MVIN: Learning Multiview Items for Recommendation](https://arxiv.org/abs/2005.12516), SIGIR 2020. Usage: `model=MVIN`.
    * `KGAT`, proposed in [KGAT: Knowledge Graph Attention Network for Recommendation](https://arxiv.org/abs/1905.07854), KDD 2019. Usage: `model=KGAT`.
    * `KGCN`, proposed in [Knowledge Graph Convolutional Networks for Recommender Systems](https://arxiv.org/abs/1904.12575), WWW 2019. Usage: `model=KGCN`.
    * `RippleNet`, proposed in [RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems](https://arxiv.org/pdf/1803.03467.pdf), CIKM 2018. Usage: `model=RippleNet`.
  * You can find other baselines in Github.
  
* `dataset`
  * It specifies the dataset.
  * Here we provide three options, including  * `amazon-book`, `movie`, or `last_fm`.

* `gpu`
  * It specifies the gpu, e.g. * `0`, `1`, and `2`.

 
## Example to Run the Attention Codes
```
$ cd src/bash/
$ bash main_att_case_st.sh $gpu
```

# Issue

* `main_run.sh syntax error near unexpected token elif`
```
$ sed -i -e 's/\r$//' *.sh
```
