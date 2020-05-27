#!/bin/bash

# cmd_min="conda deactivate"
# $cmd_min
# cmd_min="source activate py36"
# $cmd_min

export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH="."

dataset="amazon-book_20core"
aggregator="sum"
n_epochs=20
neighbor_sample_size=8
dim=16
n_iter=1
batch_size=1024
l2_weight=1e-7
lr=5e-3
tolerance=4
early_decrease_lr=2
early_stop=2

log_name=${dataset}_KGCN_hop_${n_iter}_b_${batch_size}
save_model_name=${log_name}

cmd_min="python ../model/KGCN/main.py --log_name $log_name --save_model_name $save_model_name --dataset $dataset --aggregator $aggregator --n_epochs $n_epochs --neighbor_sample_size $neighbor_sample_size 
	  --dim $dim --n_iter $n_iter --batch_size $batch_size --l2_weight $l2_weight --lr $lr --early_decrease_lr $early_decrease_lr --early_stop $early_stop"
$cmd_min
