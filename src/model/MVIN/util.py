import os
import sys

import tensorflow as tf

import numpy as np
from time import time
from model import MVIN
from train_util import Early_stop_info, Eval_score_info, Train_info_record_sw_emb
from metrics import ndcg_at_k, map_at_k, recall_at_k, hit_ratio_at_k, mrr_at_k, precision_at_k
import pickle


def topk_settings(args, show_topk, train_data, eval_data, test_data, n_item, save_record_user_list, save_user_list_name):
    if show_topk:
        user_num = 250
        k_list = [1, 2, 5, 10, 25, 50, 100]
        train_record = get_user_record(train_data, True)
        test_record = get_user_record(test_data, False)
        eval_record = get_user_record(eval_data, False)

        if True or os.path.exists(args.path.misc + 'user_list_' + save_user_list_name + "_" + str(user_num) + '.pickle') == False:
            user_list = list(set(train_record.keys()) & (set(test_record.keys() & set(eval_record.keys()))))

            user_counter_dict = {user:len(train_record[user]) for user in user_list}
            user_counter_dict = sorted(user_counter_dict.items(), key=lambda x: x[1], reverse=True)
            user_counter_dict = user_counter_dict[:user_num]
            user_list = [user_set[0] for user_set in user_counter_dict]

            if len(user_list) > user_num:
                user_list = np.random.choice(user_list, size=user_num, replace=False)
            with open(args.path.misc + 'user_list_' + save_user_list_name + "_" + str(user_num) + '.pickle', 'wb') as fp:
                pickle.dump(user_list, fp)
        print('user_list_load')
        with open (args.path.misc + 'user_list_' + save_user_list_name + "_" + str(user_num) + '.pickle', 'rb') as fp:
            user_list = pickle.load(fp)

        item_set = set(list(range(n_item)))
        return user_list, train_record, eval_record, test_record, item_set, k_list
    else:
        return [None] * 6


def ctr_eval(args, user_path, sess, model, data, user_triplet_set, batch_size):
    start = 0
    auc_list = []
    acc_list = []
    f1_list = []
    while start + batch_size <= data.shape[0]:
        auc, acc,  f1 = model.eval(sess, get_feed_dict(args, user_path, model, data, user_triplet_set, start, start + args.batch_size))
        auc_list.append(auc)
        acc_list.append(acc)
        f1_list.append(f1)
        start += batch_size

    return auc_list, acc_list, f1_list, float(np.mean(auc_list)), float(np.mean(acc_list)), float(np.mean(f1_list))


def ctr_eval_case_study(args, user_path, sess, model, data, user_triplet_set, user_history_dict, entity_index_2_name, rela_index_2_name,
    user_list, item_set_most_pop, batch_size):
    start = 0
    auc_list = []
    acc_list = []
    f1_list = []
    nb_size = args.neighbor_sample_size

    if args.SW_stage == 4:
        mixhop_parameter_path = f"{args.path.case_st}{args.log_name}_ep_{str(args.epoch)}_st_{str(args.SW_stage)}.log"
        eval_log_save = open(mixhop_parameter_path, 'w')
        text_space = "*" * 50 + "\n"
        eval_log_save.write(f"{text_space} case_study \n")

        while start + batch_size <= data.shape[0]:
            user_indices, labels, item_indices, entities_data, relations_data, importance_list_0, importance_list_1 = model.eval_case_study(sess, get_feed_dict(args, user_path, model, data, 
                user_triplet_set, start, start + args.batch_size))

            for b_i in range(batch_size):
                if user_indices[b_i] in user_list and  item_indices[b_i] in item_set_most_pop and labels[b_i] == 0:
                    eval_log_save.write(f"{'*'* 50}\n")

                    eval_log_save.write(f"user_indices = {user_indices[b_i]}, item_indices = {item_indices[b_i]}, labels = {labels[b_i]}\n")
                    eval_log_save.write(f"{'*'* 20} first_layer  {'*'* 20}\n")
                    eval_log_save.write(f"et_index 0 = {','.join('%s' %dt for dt in entities_data[0][b_i,:].tolist())}\n")
                    eval_log_save.write(f"rela_index 0 = {','.join('%s' %dt for dt in relations_data[0][b_i,:].tolist())}\n")
                    eval_log_save.write(f"et_index 1 = {','.join('%s' %dt for dt in entities_data[1][b_i,:].tolist())}\n")
                    eval_log_save.write(f"{'*'* 20} second_layer  {'*'* 20}\n")

                    for k in range(nb_size):

                        eval_log_save.write(f"entities 0 = {str(k)}\n")
                        eval_log_save.write(f"et_index 0 = {','.join('%s' %dt for dt in [entities_data[1][b_i,k].tolist()])}\n")
                        eval_log_save.write(f"rela_index 0 = {','.join('%s' %dt for dt in relations_data[1][b_i,nb_size*k: nb_size*(k+1)].tolist())}\n")
                        eval_log_save.write(f"et_index 1 = {','.join('%s' %dt for dt in entities_data[2][b_i,nb_size*k: nb_size*(k+1)].tolist())}\n")

                    eval_log_save.write(f"{'*'* 20} entity_relation_name  {'*'* 20}\n")

                    user_interact_items = index_2_name_title(user_history_dict[user_indices[b_i]], entity_index_2_name)
                    item_name = entity_index_2_name[str(item_indices[b_i])] if str(item_indices[b_i]) in entity_index_2_name else str(item_indices[b_i])

                    eval_log_save.write(f"item_name = {item_name}\n")
                    eval_log_save.write(f"user_interact_items = {','.join(user_interact_items)}\n")

                    entities_name = [index_2_name_title(et_data[b_i,:], entity_index_2_name) for et_data in entities_data]
                    relation_name = [index_2_name(rela_data[b_i,:], rela_index_2_name) for rela_data in relations_data]
                    eval_log_save.write(f"{'*'* 20} first_layer  {'*'* 20}\n")

                    eval_log_save.write(f"et_index 0 = {','.join('%s' %dt for dt in entities_name[0])}\n")
                    rea_pair = ['rela = %s, enti = %s, att = %s' % (pair[0],pair[1],pair[2]) for pair in zip(relation_name[0], entities_name[1], importance_list_0[b_i,:][0])]
                    rea_pair = '\n'.join(rea_pair)
                    eval_log_save.write(f"er rela pair 0 = {rea_pair}\n")
                    eval_log_save.write(f"{'*'* 20} second_layer  {'*'* 20}\n")

                    for k in range(nb_size):
                        eval_log_save.write(f"entities 0 = {str(k)}\n")
                        eval_log_save.write(f"et_index 0 = {','.join('%s' %dt for dt in [entities_name[1][k]])}\n")
                        rea_pair = ['rela = %s, enti = %s, att = %s' % (pair[0],pair[1],pair[2]) for pair in zip(relation_name[1][nb_size*k: nb_size*(k+1)], entities_name[2][nb_size*k: nb_size*(k+1)], importance_list_1[b_i,k])]
                        rea_pair = '\n'.join(rea_pair)
                        eval_log_save.write(f"er rela pair 1 = {rea_pair}\n")
            start += batch_size

def index_2_name(list_array, dictionary):
    return [dictionary[str(et)] if str(et) in dictionary else et for et in list_array]

def index_2_name_title(list_array, dictionary):
    return [dictionary[str(et)] if str(et) in dictionary else et for et in list_array]

def topk_eval(sess, args, user_triplet_set, model, user_list, train_record, eval_record, test_record, item_set, k_list, batch_size, mode = 'test'):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}
    MAP_list = {k: [] for k in k_list}
    hit_ratio_list = {k: [] for k in k_list}
    ndcg_list = {k: [] for k in k_list}

    
    for user in user_list:
        if mode == 'eval': ref_user = eval_record
        else: ref_user = test_record
        if user in ref_user:
            test_item_list = list(item_set - train_record[user])
            item_score_map = dict()
            start = 0
            while start + batch_size <= len(test_item_list):
                data = []

                user_list_tmp = [user] * batch_size
                item_list = test_item_list[start:start + batch_size]
                labels_list = [1] * batch_size

                items, scores = model.get_scores(sess, get_feed_dict_top_k(args, model, user_list_tmp, item_list, labels_list, user_triplet_set))

                for item, score in zip(items, scores):
                    item_score_map[item] = score
                start += batch_size

            # padding the last incomplete minibatch if exists
            if start < len(test_item_list):

                user_list_tmp = [user] * batch_size
                item_list = test_item_list[start:] + [test_item_list[-1]] * (batch_size - len(test_item_list) + start)
                labels_list = [1] * batch_size

                # items, scores = model.get_scores(
                #     sess, {model.user_indices: [user] * batch_size,
                           # model.item_indices: test_item_list[start:] + [test_item_list[-1]] * (
                           #         batch_size - len(test_item_list) + start)})  

                items, scores = model.get_scores(sess, get_feed_dict_top_k(args, model, user_list_tmp, item_list, labels_list, user_triplet_set))


                for item, score in zip(items, scores):
                    item_score_map[item] = score

            item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
            item_sorted = [i[0] for i in item_score_pair_sorted]


            for k in k_list:
                precision_list[k].append(precision_at_k(item_sorted,ref_user[user],k))
                recall_list[k].append(recall_at_k(item_sorted,ref_user[user],k))

            # ndcg
            r_hit = []
            for i in item_sorted[:k]:
                if i in ref_user[user]:
                    r_hit.append(1)
                else:
                    r_hit.append(0)
            for k in k_list:
                ndcg_list[k].append(ndcg_at_k(r_hit,k))

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]
    ndcg = [np.mean(ndcg_list[k]) for k in k_list]

    return precision, recall, ndcg, None, None


def get_feed_dict(args, user_path, model, data, user_triplet_set, start, end):
    feed_dict = {model.user_indices: data[start:end, 0],
                 model.item_indices: data[start:end, 1],
                 model.labels: data[start:end, 2]}

    for i in range(max(1,args.p_hop)):
        feed_dict[model.memories_h[i]] = [user_triplet_set[user][i][0] for user in data[start:end, 0]]
        feed_dict[model.memories_r[i]] = [user_triplet_set[user][i][1] for user in data[start:end, 0]]
        feed_dict[model.memories_t[i]] = [user_triplet_set[user][i][2] for user in data[start:end, 0]]

    return feed_dict

def get_feed_dict_top_k(args, model, user_list, item, label, user_triplet_set):
    feed_dict = {model.user_indices: user_list,
                 model.item_indices: item,
                 model.labels: label}

    for i in range(max(1,args.p_hop)):
        feed_dict[model.memories_h[i]] = [user_triplet_set[user][i][0] for user in user_list]
        feed_dict[model.memories_r[i]] = [user_triplet_set[user][i][1] for user in user_list]
        feed_dict[model.memories_t[i]] = [user_triplet_set[user][i][2] for user in user_list]

    return feed_dict

def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict
