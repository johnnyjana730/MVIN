import argparse
import numpy as np
from data_loader import load_data
from train import train
from train_util import Train_info_record
import gc
import tensorflow as tf
import pandas as pd
from path import Path

# np.random.seed(555)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_fract', type=float, default=1., help='fraction of gpu to use')

parser.add_argument('--n_round', type=int, default=3, help='number of rounds')

parser.add_argument('--dataset', type=str, default='Book-Crossing', help='which dataset to use')
parser.add_argument('--dim', type=int, default=16, help='dimension of entity and relation embeddings')
parser.add_argument('--n_hop', type=int, default=1, help='maximum hops')
parser.add_argument('--kge_weight', type=float, default=1e-2, help='weight of the KGE term')
parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of the l2 regularization term')
parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--n_epoch', type=int, default=80, help='the number of epochs')
parser.add_argument('--n_memory', type=int, default=16, help='size of ripple set for each hop')
parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                    help='how to update item at the end of each hop')
parser.add_argument('--using_all_hops', type=bool, default=True,
                    help='whether using outputs of all hops or just the last hop when making prediction')

parser.add_argument('--tolerance', type=int, default=8, help='number of epoch start to count early stop')
parser.add_argument('--early_stop', type=int, default=4, help='by epoch')
parser.add_argument('--stage_tolerance', type=int, default=2, help='number of stage start to count early stop')
parser.add_argument('--stage_early_stop', type=int, default=5, help='by stage')

parser.add_argument('--load_emb', type=bool, default=False)
parser.add_argument('--log_name', type=str, default='')
parser.add_argument('--emb_name', type=str, default='', help='transD, transE, transR, transH, none')
parser.add_argument('--save_user_list', type=bool, default=False, help='save the user list of topk evaluation')

parser.add_argument('--topk_eval', type=bool, default=False, help='whether use precision, recall, or ndcg as evaluation method')
parser.add_argument('--k_list', type=int, nargs='+', default=[1, 2, 5, 10, 25, 50, 100], help='whether use precision, recall, or ndcg as evaluation method')

parser.add_argument('--n_pop_item_eval', type=int, default=500, help='popular items for topk evalution')
parser.add_argument('--n_user_eval', type=int, default=250, help='users for topk evalution')

parser.add_argument('--new_load_data', type=bool, default=True, help='users for topk evalution')
# misc
parser.add_argument('--show_save_dataset_info', type=bool, default=False, help='show and save dataset information')

def run(args, logger, tag, load_emb=False, refresh_score=True, refresh_interaction=True):
	args.load_emb = load_emb
	data = load_data(args)
	logger.update_cur_train_info(args, refresh_score=refresh_score, refresh_interaction=refresh_interaction, user_ere_interaction_dict=data[-3], all_user_entity_count=data[-2])
	train(args, data, logger)
	logger.check_refresh_state()
	if not args.load_emb or logger.check_early_stop(args.stage_early_stop):
		logger.train_over(tag)

	gc.collect()

def main_sw(args, logger):
	tags = ['origin', 'graphsw']
	logger.init_scores(tags)
	tolerance = args.tolerance
	early_stop = args.early_stop
	for _round in range(args.n_round):
		args.round = _round

		# w/o SW	
		args.tolerance = tolerance
		args.early_stop = early_stop
		run(args, logger, tags[0])

		# SW
		args.tolerance = 0
		args.early_stop = 2
		stage = 0
		logger.start_early_stop()
		# while True:
		# 	run(args, logger, tags[1], load_emb=True, refresh_score=(stage == 0), refresh_interaction=False)
		# 	stage += 1
		# 	if logger.check_early_stop(args.stage_early_stop):
		# 		break

		logger.record_final_score()

if __name__ == "__main__":
	args = parser.parse_args()
	path = Path(args.dataset)
	args.path = path
	
	logger = Train_info_record(args)

	main_sw(args, logger)
