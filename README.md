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
@inproceedings{Tai2020MVIN,
 title={SIGIR 2020. MVIN: Learning Multiview Items for Recommendation},
 author={Chang-You Tai, Meng-Ru Wu, Yun-Wei Chu, Shao-Yu Chu, and Lun-Wei Ku},
 year={2020}
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
$ conda create -f requirements.yml
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
  * $dataset: one of "amazon-book", "movie", "last_fm"
  * $gpu: gpu number
  * $model: "KGCN", "RippleNet"

* `alg_type`
  * It specifies the type of graph convolutional layer.
  * Here we provide three options:
    * `kgat` (by default), proposed in [KGAT: Knowledge Graph Attention Network for Recommendation](xx), KDD2019. Usage: `--alg_type kgat`.
    * `gcn`, proposed in [Semi-Supervised Classification with Graph Convolutional Networks](https://openreview.net/pdf?id=SJU4ayYgl), ICLR2018. Usage: `--alg_type gcn`.
    * `graphsage`, propsed in [Inductive Representation Learning on Large Graphs.](https://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf), NeurIPS2017. Usage: `--alg_type graphsage`.
    
 

 
## Example to Run the Attention Codes
```
$ cd src/bash/
$ bash main_att_case_st.sh $gpu
```
