# MVIN
MVIN: Learning Multiview Items for Recommendation, SIGIR 2020

This repository is the implementation of MVIN ([arXiv]()):
> Chang-You Tai, Meng-Ru Wu, Yun-Wei Chu, Shao-Yu Chu, and Lun-Wei Ku. SIGIR 2020. MVIN: Learning Multiview Items for Recommendation

## Introduction
We propose MVIN,

## Citation 
If you want to use our codes and datasets in your research, please cite:
```
@inproceedings{Tai2019GraphSWAT,
 title={GraphSW: a training protocol based on stage-wise training for GNN-based Recommender Model},
 author={Chang-You Tai and Meng-Ru Wu and Yun-Wei Chu and Shao-Yu Chu},
 year={2019}
}
```
## Files in the folder

- `data/`: datasets
  - `Book-Crossing/`
  - `MovieLens-1M/`
  - `amazon-book_20core/`
  - `last-fm_50core/`
  - `music/`
  - `yelp2018_20core/`
- `src/`: implementation of GraphSW.
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
$ cd graph-stage
$ conda deactivate
$ conda create -f requirements.yml
$ conda activate graph-stage
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
