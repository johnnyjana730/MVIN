import argparse
import numpy as np
import os
import gc
from path import Path_SW, Path
from parameter_ablation import parameter_env
from data_loader_user_set import load_data
from train import train
from train_util import Train_info_record_sw_emb,Train_info_record_emb_sw_ndcg

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
parser.add_argument('--n_epochs', type=int, default=20, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--p_hop', type=int, default=1, help='mix hop')
parser.add_argument('--user_agg_hop', type=int, default=0, help='mix hop')
parser.add_argument('--n_memory', type=int, default=16, help='size of ripple set for each hop')
parser.add_argument('--dim', type=int, default=8, help='dimension of user and entity embeddings')
parser.add_argument('--h_hop', type=int, default=3, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
parser.add_argument('--l2_agg_weight', type=float, default=1e-6, help='weight of l2 regularization')

parser.add_argument('--kge_weight', type=float, default=1e-2, help='weight of the KGE term')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--tolerance', type=int, default=2, help='')
parser.add_argument('--early_decrease_lr', type=int, default=2, help='')
parser.add_argument('--early_stop', type=int, default=3, help='')

parser.add_argument('--update_item_emb', type=str, default='transform_matrix', help='transform_matrix')
parser.add_argument('--h0_att', type=str, default='st_att_h_set', help='st_att_h_set')
parser.add_argument('--model_select', type=str, default='KGCN', help='select_model')
parser.add_argument('--n_mix_hop', type=int, default=2, help='mix hop')

parser.add_argument('--load_pretrain_emb', type=bool, default=False, help='number of iterations when computing entity representation')
parser.add_argument('--save_default_model', type=bool, default=False, help='save default model')
parser.add_argument('--save_final_model', type=bool, default=True, help='save default model')
parser.add_argument('--save_record_user_list', type=bool, default=False, help='number of iterations when computing entity representation')
parser.add_argument('--show_topk_mode', type=bool, default=False, help='save default model')
parser.add_argument('--use_neighbor_rate', type=list, default=0, help='use_neighbor_rate')
parser.add_argument('--save_model_name', type=str, default="model1", help='save default model')
parser.add_argument('--new_load_data', type=bool, default=False, help='size of training dataset')
parser.add_argument('--log_name', type=str, default='', help='save default log')

parser.add_argument('--SW_stage', type=int, default=0, help='save default log')

parser.add_argument('--top_k', type=int, default=0, help='top_k')
parser.add_argument('--ablation', type=str, default='all', help='stage_wise')
parser.add_argument('--abla_exp', type=int, default=0, help='stage_wise')
parser.add_argument('--SW', type=int, default=1, help='stage_wise')
parser.add_argument('--User_orient', type=int, default=1, help='size of training dataset')
parser.add_argument('--User_orient_rela', type=int, default=1, help='size of training dataset')
parser.add_argument('--User_orient_kg_eh', type=int, default=1, help='size of training dataset')
parser.add_argument('--PS_W_ft', type=int, default=1, help='size of training dataset')
parser.add_argument('--PS_O_ft', type=int, default=1, help='size of training dataset')
parser.add_argument('--wide_deep', type=int, default=1, help='size of training dataset')
parser.add_argument('--PS_only', type=int, default=0, help='size of training dataset')
parser.add_argument('--HO_only', type=int, default=0, help='size of training dataset')


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
	args = parser.parse_args()
	args = parameter_env(args)
	args.top_k = (args.top_k == 1)

	if args.SW == True: path = Path_SW(args)
	else: path = Path(args)
	args.path = path


	if args.top_k == True:	trn_info = Train_info_record_emb_sw_ndcg(args)
	else: trn_info = Train_info_record_sw_emb(args)
	env_by_item(args, trn_info)
