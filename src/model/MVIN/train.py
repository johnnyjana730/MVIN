import os
import sys

import tensorflow as tf

import numpy as np
from time import time
from model import MVIN
from train_util import Early_stop_info, Eval_score_info, Train_info_record_sw_emb
from metrics import ndcg_at_k, map_at_k, recall_at_k, hit_ratio_at_k, mrr_at_k, precision_at_k
import pickle

# np.random.seed(1)

def train(args, data, trn_info, show_loss, show_topk):
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    adj_entity, adj_relation, user_triplet_set = data[7], data[8], data[9]
    user_path, user_path_top_k, item_set_most_pop = data[10], data[11], data[12]

    early_st_info = Early_stop_info(args,show_topk)
    eval_score_info = Eval_score_info()
    
    # top-K evaluation settings
    user_list, train_record, eval_record, test_record, item_set, k_list = topk_settings(args, show_topk, train_data, eval_data, test_data, n_item, args.save_record_user_list, args.save_model_name)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = MVIN(args, n_user, n_entity, n_relation, adj_entity, adj_relation)

        # Model_parameter = [v for v in tf.global_variables()]
        # for model_variable in Model_parameter:
        #     print("model_variable variable = ", model_variable.name)

        sess.run(tf.global_variables_initializer())
        trn_info.logger.info('Parameters:' + '\n'.join([v.name for v in tf.global_variables()]))
        
        # stage_wise_var = [v for v in tf.global_variables() if 'Adam' not in v.name and 'adam' not in v.name]
        stage_wise_var = [v for v in tf.global_variables() if 'STWS' in v.name and 'Adam' not in v.name and 'adam' not in v.name]
        # stage_wise_var = [v for v in tf.global_variables() if ('STWS' in v.name or 'agg' in v.name) and 'Adam' not in v.name]
        # stage_wise_var = [v for v in tf.global_variables() if 'agg' not in v.name and 'enti_mlp_matrix' not in v.name 
                            # and 'adam' not in v.name and 'Adam' not in v.name]
        trn_info.logger.info('SW Parameters:' + '\n'.join([v.name for v in stage_wise_var]))
        for SW_varitable in stage_wise_var:
            print("stage_wise variable = ", SW_varitable.name)

        saver = tf.train.Saver(stage_wise_var)

        if args.load_pretrain_emb == True:
            saver.restore(sess, f"{args.path.emb}_sw_para_{args.save_model_name}" + '_parameter')

        for step in range(args.n_epochs):
            # training
            np.random.shuffle(train_data)
            start = 0
            # skip the last incomplete minibatch if its size < batch size
            t_start = time()
            while start + args.batch_size <= train_data.shape[0]:
                _, loss = model.train(sess, get_feed_dict(args, user_path, model, train_data, user_triplet_set, start, start + args.batch_size))
                start += args.batch_size
            t_flag = time()

            # top-K evaluation
            if show_topk:
                precision, recall, ndcg, MAP, hit_ratio = topk_eval(
                    sess, args, user_triplet_set, model, user_list, train_record, eval_record, test_record, item_set_most_pop, k_list, args.batch_size, mode = 'eval')
                n_precision_eval = [round(i, 6) for i in precision]
                n_recall_eval = [round(i, 6) for i in recall]
                n_ndcg_eval = [round(i, 6) for i in ndcg]

                precision, recall, ndcg, MAP, hit_ratio = topk_eval(
                    sess, args, user_triplet_set, model, user_list, train_record, eval_record, test_record, item_set_most_pop, k_list, 
                    args.batch_size, mode = 'test')

                n_precision_test = [round(i, 6) for i in precision]
                n_recall_test = [round(i, 6) for i in recall]
                n_ndcg_test = [round(i, 6) for i in ndcg]

                eval_score_info.eval_ndcg_recall_pecision = [n_ndcg_eval, n_recall_eval, n_precision_eval]
                eval_score_info.test_ndcg_recall_pecision = [n_ndcg_test, n_recall_test, n_precision_test]

                trn_info.update_score(step, eval_score_info)
                
                if early_st_info.update_score(step,n_recall_eval[2],sess,model,saver) == True: break
            else:
                # CTR evaluation
                _, _, _, auc, acc, f1 = ctr_eval(args, user_path, sess, model, train_data, user_triplet_set, args.batch_size)
                eval_score_info.train_auc_acc_f1 = auc, acc, f1
                _, _, _, auc, acc, f1 = ctr_eval(args, user_path, sess, model, eval_data, user_triplet_set, args.batch_size)
                eval_score_info.eval_auc_acc_f1 = auc, acc, f1
                test_auc_list, test_acc_list, test_f1_list, auc, acc, f1 = ctr_eval(args, user_path, sess, model, test_data, user_triplet_set, args.batch_size)
                eval_score_info.test_auc_acc_f1 = auc, acc, f1

                satistic_list = [test_auc_list, test_acc_list, test_f1_list]

                trn_info.update_score(step, eval_score_info)

                if early_st_info.update_score(step,eval_score_info.eval_st_score(),sess,model,saver, satistic_list= satistic_list) == True: break

    tf.reset_default_graph()

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


def get_feed_dict(args, user_path, model, data, user_triplet_set, start, end):
    feed_dict = {model.user_indices: data[start:end, 0],
                 model.item_indices: data[start:end, 1],
                 model.labels: data[start:end, 2]}

    for i in range(max(1,args.p_hop)):
        feed_dict[model.memories_h[i]] = [user_triplet_set[user][i][0] for user in data[start:end, 0]]
        feed_dict[model.memories_r[i]] = [user_triplet_set[user][i][1] for user in data[start:end, 0]]
        feed_dict[model.memories_t[i]] = [user_triplet_set[user][i][2] for user in data[start:end, 0]]

    return feed_dict

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


def get_feed_dict_top_k(args, model, user_list, item, label, user_triplet_set):
    feed_dict = {model.user_indices: user_list,
                 model.item_indices: item,
                 model.labels: label}

    for i in range(max(1,args.p_hop)):
        feed_dict[model.memories_h[i]] = [user_triplet_set[user][i][0] for user in user_list]
        feed_dict[model.memories_r[i]] = [user_triplet_set[user][i][1] for user in user_list]
        feed_dict[model.memories_t[i]] = [user_triplet_set[user][i][2] for user in user_list]

    return feed_dict

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
