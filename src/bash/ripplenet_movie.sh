export CUDA_VISIBLE_DEVICES=1

dataset="MovieLens-1M"
# model hyper-parameter setting
dim=16
n_hop=1
n_memory=8
l2_weight=1e-7

# learning parameter setting
lr=5e-3
tolerance=2
early_stop=3
batch_size=1024

topk_eval=False

emb_name="h${n_hop}_d${dim}_m${n_memory}_nosw_top_k_new_set"
log_name="${dataset}_${emb_name}"

cmd="python3 ../model/RippleNet/main.py
    --dataset $dataset
    --dim $dim
    --n_memory $n_memory
    --l2_weight $l2_weight
    --batch_size $batch_size
    --lr $lr
    --n_hop $n_hop
    --early_stop $early_stop
    --tolerance $tolerance
    --emb_name $emb_name
    --log_name $log_name"
$cmd
