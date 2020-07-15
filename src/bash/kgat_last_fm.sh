export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH="."

alg_type=bi
dataset=last-fm_50core
regs=[1e-5,1e-5]
layer_size=[16,16]
lr=0.0002
epoch=400
verbose=1
save_flag=1
pretrain=-1
batch_size=512
node_dropout=[0.1]
use_att=True
use_kge=True

model_type=kgat

embed_size=16
mess_dropout=[0.1]
layer_size=[16]

echo $CUDA_VISIBLE_DEVICES

filename=../model/KGAT/main.py

cmd="python3 ${filename} --model_type $model_type --alg_type $alg_type --dataset $dataset --regs $regs --layer_size $layer_size
		--embed_size $embed_size --lr $lr --epoch $epoch --verbose $verbose --save_flag $save_flag --pretrain $pretrain --batch_size $batch_size 
		--node_dropout $node_dropout --mess_dropout $mess_dropout --use_att $use_att --use_kge $use_kge"
$cmd

# DIM_SIZE=(16 8 32)
# lr_list=(0.01 0.0005 0.005)

# for DIM_S in ${DIM_SIZE[@]}
# do
# 	for LR_R in ${lr_list[@]}
# 	do
# 		embed_size=$DIM_S
# 		lr=$LR_R
# 		cmd_min="python3 ${filename} --model_type $model_type --alg_type $alg_type --dataset $dataset --regs $regs --layer_size $layer_size
# 				--embed_size $embed_size --lr $lr --epoch $epoch --verbose $verbose --save_flag $save_flag --pretrain $pretrain --batch_size $batch_size 
# 				--node_dropout $node_dropout --mess_dropout $mess_dropout --use_att $use_att --use_kge $use_kge"
# 		$cmd_min

# 	done
# done
