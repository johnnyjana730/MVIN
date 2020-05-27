# import tensorflow.compat.v1 as tf
import tensorflow as tf
import numpy as np
from model import RippleNet
from train_util import Early_stop_info, ndcg_at_k
from metrics import ndcg_at_k, map_at_k, recall_at_k, hit_ratio_at_k, mrr_at_k, precision_at_k
from collections import defaultdict
from time import time
from functools import partial
import pickle
import os

def train(args, data_info, logger):
    train_data, eval_data, test_data = data_info[0], data_info[1], data_info[2]
    n_item, n_user = data_info[3], data_info[4]
    n_entity, n_relation = data_info[5], data_info[6]
    ripple_set, item_set_most_pop = data_info[7], data_info[-1]
    if args.show_save_dataset_info:
        print(f'train({len(train_data)}), eval({len(eval_data)}), test({len(test_data)})')

    # train_dataset = get_dataset(train_data, ripple_set, n_hop=args.n_hop, batch_size=args.batch_size)
    # eval_dataset = get_dataset(eval_data, ripple_set, n_hop=args.n_hop, batch_size=args.batch_size)
    # test_dataset = get_dataset(test_data, ripple_set, n_hop=args.n_hop, batch_size=args.batch_size)
    # if args.topk_eval:
    #     topk_dataset = get_dataset(topk_data, ripple_set, n_hop=args.n_hop, batch_size=args.batch_size)

    if args.topk_eval:
        user_list, train_record, eval_record, test_record, item_set, k_list = topk_settings(args, train_data, eval_data, test_data, n_item)

    # init early stop controller
    early_stop = Early_stop_info(args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_fract

    with tf.Session(config=config) as sess:
        model = RippleNet(args, n_entity, n_relation)

        init = tf.global_variables_initializer()
        sess.run(init)

        # if args.load_emb == True:
        #     print('load pretrained emb ...')
        #     model.initialize_pretrained_embeddings(sess)

        for epoch in range(80):
            scores = {t: {} for t in ['train', 'eval', 'test']}
            
            np.random.shuffle(train_data)
            start = 0

            t_start = time()
            while start < train_data.shape[0]:
                _, loss = model.train(
                    sess, get_feed_dict(args, model, train_data, ripple_set, start, start + args.batch_size))
                start += args.batch_size
            t_flag = time()

            # evaluation
            scores['train'] = evaluation(sess, args, model, train_data, ripple_set, args.batch_size)
            scores['eval'] = evaluation(sess, args, model, eval_data, ripple_set, args.batch_size)
            scores['test'] = evaluation(sess, args, model, test_data, ripple_set, args.batch_size)
            
            early_stop_score = 0.
            if args.topk_eval:
                # topk evaluation
                # topk_scores = topk_evaluation(sess, model, topk_data, ripple_set, args.k_list)
                precision, recall, ndcg, MAP, hit_ratio = topk_eval(
                    sess, args, ripple_set, model, user_list, train_record, eval_record, test_record, item_set_most_pop, k_list, args.batch_size, mode = 'eval')
                n_precision_eval = [round(i, 6) for i in precision]
                n_recall_eval = [round(i, 6) for i in recall]
                n_ndcg_eval = [round(i, 6) for i in ndcg]

                for t in ['eval']:
                    scores[t]['p'] = n_precision_eval
                    scores[t]['r'] = n_recall_eval
                    scores[t]['ndcg'] = n_ndcg_eval

                precision, recall, ndcg, MAP, hit_ratio = topk_eval(
                    sess, args, ripple_set, model, user_list, train_record, eval_record, test_record, item_set_most_pop, k_list, args.batch_size, mode = 'test')
                n_precision_test = [round(i, 6) for i in precision]
                n_recall_test = [round(i, 6) for i in recall]
                n_ndcg_test = [round(i, 6) for i in ndcg]

                for t in ['test']:
                    scores[t]['p'] = n_precision_test
                    scores[t]['r'] = n_recall_test
                    scores[t]['ndcg'] = n_ndcg_test
                    # for m in ['p', 'r', 'ndcg']:
                    #     scores[t][m] = topk_scores[t][m]
                # print('scores = ', scores)
                early_stop_score = scores['eval']['r'][2]
            # else:
            early_stop_score = scores['eval']['auc']

            logger.update_score(epoch, scores)
            
            print('training time: %.1fs' % (t_flag - t_start), end='') 
            print(', total: %.1fs.' % (time() - t_start))

            # if early_stop_score >= early_stop.best_score:
            #     print('save embs ...', end='\r')
            #     model.save_pretrained_emb(sess)

            if early_stop.update_score(epoch, early_stop_score) == True: break
        
    tf.reset_default_graph()

def get_feed_dict(args, model, data, ripple_set, start, end):
    feed_dict = dict()
    feed_dict[model.items] = data[start:end, 1]
    feed_dict[model.labels] = data[start:end, 2]
    for i in range(args.n_hop):
        feed_dict[model.memories_h[i]] = [ripple_set[user][i][0] for user in data[start:end, 0]]
        feed_dict[model.memories_r[i]] = [ripple_set[user][i][1] for user in data[start:end, 0]]
        feed_dict[model.memories_t[i]] = [ripple_set[user][i][2] for user in data[start:end, 0]]
    return feed_dict

# def topk_settings(args, train_data, eval_data, test_data, n_item):
#     train_record = get_user_record(train_data)
#     eval_record = get_user_record(eval_data)
#     test_record = get_user_record(test_data)

#     user_list_path = f'{args.path.misc}user_list_{args.n_user_eval}.pickle'
#     pop_item_path = f'{args.path.misc}pop_item_{args.n_pop_item_eval}.pickle'

#     if not os.path.isfile(user_list_path):
#         print('save user list ...')
#         user_list = list(set(train_record.keys()) & set(eval_record.keys()) & set(test_record.keys()))
#         user_counter = { u: len(train_record[u]) for u in user_list }
#         user_counter_sorted = sorted(user_counter.items(), key=lambda x: x[1], reverse=True)
#         user_list = [u for u, _ in user_counter_sorted[:args.n_user_eval]]

#         with open(user_list_path, 'wb') as f:
#             pickle.dump(user_list, f)
#     else:
#         print('load user list ...')
#         with open(user_list_path, 'rb') as f:
#             user_list = pickle.load(f)

#     with open(pop_item_path, 'rb') as f:
#         item_set = set(pickle.load(f))

#     data = []
#     for user in user_list:
#         data += [[user, item, 1] for item in (item_set - train_record[user])]
#     data = np.array(data)
    
#     return user_list, train_record, eval_record, test_record, item_set, k_list

def topk_settings(args, train_data, eval_data, test_data, n_item):
    user_num = 250
    k_list = [1, 2, 5, 10, 25, 50, 100]
    train_record = get_user_record(train_data, True)
    test_record = get_user_record(test_data, False)
    eval_record = get_user_record(eval_data, False)

    if True or os.path.exists(args.path.misc + 'user_list_250' + '.pickle') == False:
        user_list = list(set(train_record.keys()) & set(test_record.keys() & (eval_record.keys())))
        user_counter_dict = {user:len(train_record[user]) for user in user_list}
        user_counter_dict = sorted(user_counter_dict.items(), key=lambda x: x[1], reverse=True)
        user_counter_dict = user_counter_dict[:user_num]
        user_list = [user_set[0] for user_set in user_counter_dict]

        if len(user_list) > user_num:
            user_list = np.random.choice(user_list, size=user_num, replace=False)
        with open(args.path.misc + 'user_list_250' + '.pickle', 'wb') as fp:
            pickle.dump(user_list, fp)
    print('user_list_load')
    with open (args.path.misc + 'user_list_250' + '.pickle', 'rb') as fp:
        user_list = pickle.load(fp)

    item_set = set(list(range(n_item)))
    return user_list, train_record, eval_record, test_record, item_set, k_list


def evaluation(sess, args, model, data, ripple_set, batch_size):
    start = 0
    auc_list = []
    acc_list = []
    f1_list = []
    while start < data.shape[0]:
        auc, acc, f1 = model.eval(sess, get_feed_dict(args, model, data, ripple_set, start, start + batch_size))
        auc_list.append(auc)
        acc_list.append(acc)
        f1_list.append(f1)
        start += batch_size
    return {
        'auc': float(np.mean(auc_list)),
        'acc': float(np.mean(acc_list)),
        'f1': float(np.mean(f1_list))
    }

def get_feed_dict_top_k(args, model, user_list, item, label, user_triplet_set):
    feed_dict = {model.items: item, model.labels: label}

    for i in range(args.n_hop):
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


def topk_evaluation(sess, model, dataset, ripple_set, k_list):
    topk_scores = {
       t: {
           m: {
               k: [] for k in k_list
           } for m in ['p', 'r', 'ndcg']
       } for t in ['eval', 'test']
    }

    user_item_score_map = defaultdict(list)
    
    start = 0
    try:
        while start < dataset.shape[0]:
            users, items, scores = model.get_scores(sess, get_feed_dict(args, model, dataset, ripple_set, start, start + batch_size))
            user_item_score_map[u].append((i, s)) 

    except:
        pass

    # model.iter_init(sess, dataset)
    # try:
    #     while True:
    #         users, items, scores = model.get_scores(sess)
    #         for u, i, s in zip(users, items, scores):
    #             user_item_score_map[u].append((i, s))    
    # except tf.errors.OutOfRangeError:
    #     pass

    print()

    for u, item_score_pair in user_item_score_map.items():
        item_score_pair_sorted = sorted(item_score_pair, key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        eval_r_hit, test_r_hit = [], []
        for i in item_sorted[:k_list[-1]]:
            eval_r_hit.append(1 if i in eval_user_dict[u] else 0)
            test_r_hit.append(1 if i in test_user_dict[u] else 0)
            
        for k in k_list:
            eval_hit_num = len(set(item_sorted[:k]) & eval_user_dict[u])
            topk_scores['eval']['p'][k].append(eval_hit_num / k)
            topk_scores['eval']['r'][k].append(eval_hit_num / len(eval_user_dict[u]))
            topk_scores['eval']['ndcg'][k].append(ndcg_at_k(eval_r_hit, k))

            test_hit_num = len(set(item_sorted[:k]) & test_user_dict[u])
            topk_scores['test']['p'][k].append(test_hit_num / k)
            topk_scores['test']['r'][k].append(test_hit_num / len(test_user_dict[u]))
            topk_scores['test']['ndcg'][k].append(ndcg_at_k(test_r_hit, k))

    for t in ['eval', 'test']:
        for m in ['p', 'r', 'ndcg']:
            topk_scores[t][m] = [np.around(np.mean(topk_scores[t][m][k]), decimals=4) for k in k_list]
    print('topk_scores = ', topk_scores)
    input()
    return topk_scores

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
