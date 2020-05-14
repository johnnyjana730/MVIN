import argparse
import numpy as np
import os
import gc
from path import Path_SW, Path
from parameter_ablation import parameter_env
from data_loader_user_set import load_data
from train import train
from train_util import Train_info_record_sw_emb,Train_info_record_emb_sw_ndcg
from parser import parse_args


def selc_data_and_run(load_pretrain_emb = False, tags = 0, show_loss = False, show_topk = False, record_info = False):
	if load_pretrain_emb == True: args.load_pretrain_emb = True

	data = load_data(args)
	trn_info.update_cur_train_info(args, record_info)
	train(args, data, trn_info,  show_loss, show_topk)
	trn_info.train_over(tags)
	args.load_pretrain_emb = False
	del data
	gc.collect()

def env_by_item(args, trn_info):
	if args.top_k == True:	show_topk_mode = True
	else: show_topk_mode = False
	exp_times = 3
	if args.abla_exp == True: exp_times = 3
	for epoch in range(exp_times):
		if epoch == exp_times - 1: record_info = True
		else: record_info = False

		args.epoch = epoch
		args.SW_stage = 0
		selc_data_and_run(load_pretrain_emb = False, show_topk = show_topk_mode, record_info = record_info)

		# stage_wise traing
		if args.SW == True:
			for repeat in range(5):
				args.SW_stage += 1
				selc_data_and_run(load_pretrain_emb = True, show_topk = show_topk_mode)
				if trn_info.sw_early_stop >= 3: break

		trn_info.record_final_score(record_info = record_info)
		trn_info.counter_add()

if __name__ == "__main__":
	args = parse_args()

	if args.SW == True: path = Path_SW(args)
	else: path = Path(args)
	args.path = path

	if args.top_k == True:	trn_info = Train_info_record_emb_sw_ndcg(args)
	else: trn_info = Train_info_record_sw_emb(args)
	env_by_item(args, trn_info)
