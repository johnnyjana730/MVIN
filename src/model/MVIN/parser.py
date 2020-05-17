import argparse

def parse_args():

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

	parser.add_argument('--attention_cast_st', type=int, default=0, help='attention cast study')

	args = parser.parse_args()

	args.top_k = (args.top_k == 1)
	args.attention_cast_st = (args.attention_cast_st == 1)

	return args