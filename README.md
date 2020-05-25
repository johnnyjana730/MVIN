# MVIN
MVIN: Learning Multiview Items for Recommendation, SIGIR 2020

This repository is the implementation of MVIN ([arXiv]()):
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
- `src/`: implementation of MVIN.
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
- MVIN  
```
$ cd src/model/bash/
$ bash main_az_book.sh 1 # e.g. az_book, gpu=1
```
  
## Example to Run the Attention Codes
```
$ cd src/model/bash/
$ bash main_att_case_st.sh 1 # e.g. gpu=1
```
