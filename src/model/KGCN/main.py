import argparse
import numpy as np
import os
import gc
from path import Path
from data_loader import load_data
from train import train
from train_util import Train_info_record_sw_emb, Train_info_record_emb_sw_ndcg

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
parser.add_argument('--n_epochs', type=int, default=20, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=8, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=3, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--tolerance', type=int, default=5, help='')
parser.add_argument('--early_decrease_lr', type=int, default=2, help='')
parser.add_argument('--early_stop', type=int, default=3, help='')

parser.add_argument('--load_pretrain_emb', type=bool, default=False, help='number of iterations when computing entity representation')
parser.add_argument('--save_default_model', type=bool, default=False, help='save default model')
parser.add_argument('--save_final_model', type=bool, default=True, help='save default model')
parser.add_argument('--save_record_user_list', type=bool, default=False, help='number of iterations when computing entity representation')
parser.add_argument('--show_topk_mode', type=bool, default=False, help='topk evaluation')
parser.add_argument('--use_neighbor_rate', type=list, default=0, help='use_neighbor_rate')
parser.add_argument('--save_model_name', type=str, default="model1", help='save default model')
parser.add_argument('--log_name', type=str, default='', help='save default log')

def selc_data_and_run(load_pretrain_emb = False, tags = 0, show_loss = False, show_topk = False):
	if load_pretrain_emb == True: args.load_pretrain_emb = True

	data = load_data(args)
	trn_info.update_cur_train_info(args)
	train(args, data, trn_info,  show_loss, show_topk)
	trn_info.train_over(tags)
	
	args.load_pretrain_emb = False

	del data
	gc.collect()

def env_by_item(args, trn_info):
	for epoch in range(3):
		args.epoch = epoch
		selc_data_and_run(load_pretrain_emb = False, show_topk = args.show_topk_mode)

		# stage_wise traing
		if args.stage_wise == True:
			for repeat in range(1):
				selc_data_and_run(load_pretrain_emb = True, show_topk = args.show_topk_mode)
				if trn_info.sw_early_stop >= 3: break
		trn_info.record_final_score()
	trn_info.counter_add()

if __name__ == "__main__":
	args = parser.parse_args()
	path = Path(args.dataset)
	args.path = path

	# stage_wise traing
	args.stage_wise = True
	trn_info = Train_info_record_sw_emb(args)
	env_by_item(args, trn_info)

	# non stage_wise training
	args.stage_wise = False
	args.n_epochs = 50
	trn_info = Train_info_record_sw_emb(args)
	env_by_item(args, trn_info)
