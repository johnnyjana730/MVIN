export CUDA_VISIBLE_DEVICES=1

dataset="last-fm_50core"
# model hyper-parameter setting
dim=16
n_hop=1
n_memory=8
l2_weight=1e-7

# learning parameter setting
lr=4e-3
batch_size=512
tolerance=2
early_stop=4

# log file setting
emb_name="h${n_hop}_d${dim}_m${n_memory}_nosw_top_k_debug"
log_name="${dataset}_${emb_name}"

# command
cmd="python3 ../model/RippleNet/main.py
    --dataset $dataset
    --dim $dim
    --n_memory $n_memory
    --l2_weight $l2_weight
    --lr $lr
    --emb_name $emb_name
    --log_name $log_name"
$cmd

# NEIGHBOR_SIZE=(64 32 16 8)

# for NEIGHBOR in ${NEIGHBOR_SIZE[@]}
# do

#     n_memory=$NEIGHBOR
# 	cmd="python3 main.py
# 	    --dataset $dataset
# 	    --dim $dim
# 	    --n_memory $n_memory
# 	    --l2_weight $l2_weight
#         --batch_size $batch_size
# 	    --lr $lr
# 	    --emb_name $emb_name
# 	    --log_name $log_name"
# 	$cmd
# done
