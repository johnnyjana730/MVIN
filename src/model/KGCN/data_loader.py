import numpy as np
import collections
import os
import pickle
import random
import csv
import pandas as pd
from collections import defaultdict
from collections import Counter 
import time
import multiprocessing
import itertools
from multiprocessing import Pool, cpu_count
import sys

def load_data(args):
    n_user, n_item, train_data, eval_data, test_data, item_set_most_pop = load_rating(args)
    kg, n_entity, n_relation, adj_entity, adj_relation, user_path, user_path_top_k = load_kg(args)
    print('data loaded.')

    all_user_entity_count = get_all_user_entity_count(args,train_data, kg, adj_entity, adj_relation, hop=args.n_iter)
    return n_user, n_item, n_entity, n_relation, train_data, eval_data, test_data, adj_entity, adj_relation, user_path, user_path_top_k, item_set_most_pop

def load_rating(args):
    print('reading rating file ...')

    rating_file = args.path.data + 'ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
        np.save(rating_file + '.npy', rating_np)

    n_user = max(set(rating_np[:, 0])) + 1
    n_item = max(set(rating_np[:, 1])) + 1

    # ************ topk_settings **************
    if os.path.exists(f"{args.path.misc}KGNN_pop_item_set_500.pickle") == False:
        item_count = {}
        for i in range(rating_np.shape[0]):
            item = rating_np[i, 1]
            if item not in item_count:
                item_count[item] = 0
            item_count[item] += 1
        item_count = sorted(item_count.items(), key=lambda x: x[1], reverse=True)
        item_count = item_count[:500]
        item_set_most_pop = [item_set[0] for item_set in item_count]
        with open(f"{args.path.misc}KGNN_pop_item_set_500.pickle", 'wb') as fp:
            pickle.dump(item_set_most_pop, fp)
        
    with open(f"{args.path.misc}KGNN_pop_item_set_500.pickle", 'rb') as fp:
        item_set_most_pop = pickle.load(fp)

    item_set_most_pop = set(item_set_most_pop)
    # ************topk_settings**************

    train_data, eval_data, test_data = load_pre_data(args)

    return n_user, n_item, train_data, eval_data, test_data, item_set_most_pop

def add_negative_sample(train_data, eval_data, test_data, item_set_most_pop):
    user_history_dict = dict()
    for data_set in [train_data, eval_data, test_data]:
        for interaction in data_set:
            user = interaction[0]
            item = interaction[1]
            label = interaction[2]
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)

    train_add_neg = dict()

    for user, item_set in user_history_dict.items():
        neg_set = list(item_set_most_pop - item_set)
        neg_set = random.sample(neg_set, 10)
        tmp = []
        for item_neg in neg_set:
            tmp.append([user, item_neg, 0])

        train_data = np.append(train_data, np.array(tmp), axis=0)
    return train_data

def get_all_user_entity_count(args,train_data, kg, adj_entity, adj_relation, hop=0):
    item_pool = set(train_data[:, 1].tolist())

    if args.load_pretrain_emb == True and args.save_final_model == True:
        with open(f'{args.path.misc}total_kg_exploration_{args.save_model_name}.pickle', 'rb') as handle:
            total_kg_exploration = pickle.load(handle)
    else:
        total_kg_exploration = set()

    item_pool_tmp = item_pool
    for _ in range(hop):
        h_r_t_set = set([tuple([item,tail,rela]) for item in item_pool_tmp for tail, rela in zip(adj_entity[item,:].tolist(), adj_relation[item,:].tolist())])
        item_pool_tmp = set([tail for item in item_pool_tmp for tail in adj_entity[item,:].tolist()])
        total_kg_exploration |= h_r_t_set

    if args.save_final_model == True:
        with open(f'{args.path.misc}total_kg_exploration_{args.save_model_name}.pickle', 'wb') as handle:
            pickle.dump(total_kg_exploration, handle, protocol=pickle.HIGHEST_PROTOCOL)

    total_kg = set()
    item_pool_tmp = item_pool
    for _ in range(hop):
        h_r_t_set = set([tuple([item,tail,rela]) for item in item_pool_tmp for tail, rela in kg[item]])
        item_pool_tmp = set([tail for item in item_pool_tmp for tail, _ in kg[item]])
        total_kg |= h_r_t_set


    all_neighbor_num = len(total_kg)
    use_neighbor_num = len(total_kg_exploration)
    use_neighbor_rate = use_neighbor_num / all_neighbor_num
    args.use_neighbor_rate = [all_neighbor_num, use_neighbor_num, round(use_neighbor_rate, 6)]
    return

def load_pre_data(args):
    train_data = pd.read_csv(f'{args.path.data}train_pd.csv',index_col=None)
    train_data = train_data.drop(train_data.columns[0], axis=1)
    train_data = train_data[['user','item','like']].values

    eval_data = pd.read_csv(f'{args.path.data}eval_pd.csv',index_col=None)
    eval_data = eval_data.drop(eval_data.columns[0], axis=1)
    eval_data = eval_data[['user','item','like']].values

    test_data = pd.read_csv(f'{args.path.data}test_pd.csv',index_col=None)
    test_data = test_data.drop(test_data.columns[0], axis=1)
    test_data = test_data[['user','item','like']].values
    return train_data, eval_data, test_data

def dataset_split(rating_np, args):
    print('splitting dataset ...')

    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    if args.ratio < 1:
        train_indices = np.random.choice(list(train_indices), size=int(len(train_indices) * args.ratio), replace=False)

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data

def load_kg(args):
    print('reading KG file ...')

    # reading kg file
    kg_file = args.path.data + 'kg_final.npy'
    kg_np = np.load(kg_file)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))

    rating_file = args.path.data + 'ratings_final.npy'
    ratings_final = np.load(rating_file)

    kg = construct_kg(args, kg_np)

    adj_entity, adj_relation, user_path, user_path_top_k = None, None, None, None
    adj_entity, adj_relation = construct_adj(args, kg, n_entity)

    return kg, n_entity, n_relation, adj_entity, adj_relation, user_path, user_path_top_k


def construct_kg(args, kg_np):
    print('constructing knowledge graph ...')
    kg = dict()
    for triple in kg_np:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        # treat the KG as an undirected graph
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))

    return kg

def construct_adj(args, kg, entity_num):
    print('constructing adjacency matrix ...')
    # each line of adj_entity stores the sampled neighbor entities for a given entity
    # each line of adj_relation stores the corresponding sampled neighbor relations
    adj_entity = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    adj_relation = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    for entity in range(entity_num):
        neighbors = kg[entity]
        n_neighbors = len(neighbors)
        if n_neighbors >= args.neighbor_sample_size:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=True)
        adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
        adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])

    return adj_entity, adj_relation
