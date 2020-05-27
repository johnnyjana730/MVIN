export CUDA_VISIBLE_DEVICES=$1

dataset="amazon-book_20core"
# model hyper-parameter setting
dim=16
n_hop=1
n_memory=32
l2_weight=1e-7

# learning parameter setting
lr=3e-3
tolerance=2
early_stop=5
batch_size=1024

# log file setting
emb_name="h${n_hop}_d${dim}_m${n_memory}_no_sw"
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
