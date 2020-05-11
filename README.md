# GraphSW

This repository is the implementation of GraphSW ([arXiv](https://arxiv.org/abs/1908.05611)):
> Chang-You Tai, Meng-Ru Wu, Yun-Wei Chu, Shao-Yu Chu, and Lun-Wei Ku.2018. GraphSW: a training protocol based on stage-wise training for GNN-based Recommender Model.
## Introduction
We propose GraphSW, a strategy based on stage-wise training framework which would only access to a subset of the entities in KG in everystage. During the following stages, the learned embedding from previous stages is provided to the network in the next stage and the model can learn the information gradually from the KG.
![image](https://github.com/mengruwu/graphsw-dev/blob/master/framwork.png)
![image](https://github.com/mengruwu/graphsw-dev/blob/master/performance.png)
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
- RippleNet
  ```
  $ cd src/RippleNet/
  $ bash RippleNet_{dataset}.sh # e.g. music
  ```
- KGCN
  ```
  $ cd src/KGCN/
  $ bash main_{dataset}.sh # e.g. music
  ```
