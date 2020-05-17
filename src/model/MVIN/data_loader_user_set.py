import numpy as np
import collections
import os
import sys
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
import multiprocessing as mp
from functools import partial

def load_data(args):
    n_user, n_item, train_data, eval_data, test_data, user_history_dict, item_set_most_pop = load_rating(args)
    
    if args.attention_cast_st == True and args.dataset[:6] == "amazon":
        entity_index_2_name, rela_index_2_name = load_enti_rela_name(args)
    else:
        entity_index_2_name, rela_index_2_name = None, None

    kg, n_entity, n_relation, adj_entity, adj_relation, user_path, user_path_top_k = load_kg(args, train_data)
    print('data loaded.')
    user_triplet_set = get_user_triplet_set(args, kg, user_history_dict)
    all_user_entity_count = get_all_user_entity_count(args,train_data, kg, adj_entity, adj_relation, hop=args.h_hop)
    return n_user, n_item, n_entity, n_relation, train_data, eval_data, test_data, adj_entity, adj_relation, user_triplet_set, \
            user_path, user_path_top_k, item_set_most_pop, user_history_dict, entity_index_2_name, rela_index_2_name

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

    if args.dataset == "MovieLens-1M": top_k = 500
    else: top_k = 500

    if os.path.exists(f"{args.path.misc}KGNN_pop_item_set_{top_k}.pickle") == False:
        item_count = {}
        for i in range(rating_np.shape[0]):
            item = rating_np[i, 1]
            if item not in item_count:
                item_count[item] = 0
            item_count[item] += 1
        item_count = sorted(item_count.items(), key=lambda x: x[1], reverse=True)
        item_count = item_count[:top_k]
        item_set_most_pop = [item_set[0] for item_set in item_count]
        with open(f"{args.path.misc}KGNN_pop_item_set_{top_k}.pickle", 'wb') as fp:
            pickle.dump(item_set_most_pop, fp)
        
    with open(f"{args.path.misc}KGNN_pop_item_set_{top_k}.pickle", 'rb') as fp:
        item_set_most_pop = pickle.load(fp)

    item_set_most_pop = set(item_set_most_pop)
    u_counter, i_counter = {}, {}
    if args.new_load_data == True:
        print('load new train eval test')
        train_data, eval_data, test_data = dataset_split(rating_np, args)
        with open(f"{args.path.data}train_data.pickle",'wb') as f:
            pickle.dump(train_data, f)
        with open(f"{args.path.data}eval_data.pickle",'wb') as f:
            pickle.dump(eval_data, f)
        with open(f"{args.path.data}test_data.pickle",'wb') as f:
            pickle.dump(test_data, f)
    else:
        train_data, eval_data, test_data = load_pre_data(args)

    user_history_dict = dict()
    for i in range(train_data.shape[0]):
        user = train_data[i][0]
        item = train_data[i][1]
        rating = train_data[i][2]
        # print('user, item, rating', user, item, rating)
        # input()
        if rating == 1:
            if user not in user_history_dict:
                user_history_dict[user] = []
            user_history_dict[user].append(item)

    train_indices = [i for i in range(train_data.shape[0]) if train_data[i][0] in user_history_dict]
    eval_indices = [i for i in range(eval_data.shape[0]) if eval_data[i][0] in user_history_dict]
    test_indices = [i for i in range(test_data.shape[0]) if test_data[i][0] in user_history_dict]

    train_data = train_data[train_indices]
    eval_data = eval_data[eval_indices]
    test_data = test_data[test_indices]
        # for t_i in range(rating_np.shape[0]):
        #     user = rating_np[t_i][0]
        #     item = rating_np[t_i][1]
        #     rating = rating_np[t_i][2]

        #     if rating == 1:
        #         if user not in user_history_dict:
        #             user_history_dict[user] = [item]
        #         user_history_dict[user].append(item)
            
        #     if user not in u_counter:
        #         u_counter[user] = set()
        #     if item not in i_counter:
        #         i_counter[item] = set()

        #     u_counter[user].add(item)
        #     i_counter[item].add(user)
    return n_user, n_item, train_data, eval_data, test_data, user_history_dict, item_set_most_pop


def load_enti_rela_name(args):
    rating_file = args.path.data + 'meta_Books.json'
    if os.path.exists(rating_file) == False:
        return None, None

    if os.path.exists(f"{args.path.misc}new_meta_Books_dict.pickle") == False:
        new_meta_Books_dict = {}
        rating_file = args.path.data + 'meta_Books.json'
        with open(rating_file, 'r') as f:
            d = f.readlines()
            for line_json in d:
                mb_dict = json.loads(line_json)
                # print('mb_dict = ', mb_dict)
                if str(mb_dict['asin']) not in new_meta_Books_dict:
                    new_meta_Books_dict[str(mb_dict['asin'])] = {}
                try:
                    mb_dict_sp = mb_dict['title'].split('(')
                    new_meta_Books_dict[str(mb_dict['asin'])] = mb_dict_sp[0]
                    # new_meta_Books_dict[str(mb_dict['asin'])]['title'] = mb_dict['title']
                    # new_meta_Books_dict[str(mb_dict['asin'])]['description'] = mb_dict['description']
                except:
                    pass
                # print(new_meta_Books_dict)
                # input()
        with open(f"{args.path.misc}new_meta_Books_dict.pickle",'wb') as f:
            pickle.dump(new_meta_Books_dict, f)
    else:
        print('load_random_adj')
        with open(f"{args.path.misc}new_meta_Books_dict.pickle",'rb') as f:
            new_meta_Books_dict = pickle.load(f)


    rating_file = args.path.data + 'ab2fb'
    fp = open(rating_file + '.txt', "r")
    freebase_2_entity = {}
    for line in iter(fp):
        line_tr = line.replace('\n', '').split('\t')
        # print('line_tr = ', line_tr)
        freebase_2_entity[line_tr[-1]] = line_tr[0]
    # print(freebase_2_entity)
    # input()
    fp.close()

    rating_file = args.path.data + 'entity_list'
    fp = open(rating_file + '.txt', "r")
    index_2_freebase = {}
    index_2_entity = {}
    for line in iter(fp):
        line_tr = line.replace('\n', '').split(' ')
        index_2_freebase[line_tr[-1]]  =  "".join(line_tr[:-1])
        if "".join(line_tr[:-1]) in new_meta_Books_dict:
            # print('find 1 = ', new_meta_Books_dict["".join(line_tr[:-1])])
            index_2_entity[line_tr[-1]] = new_meta_Books_dict["".join(line_tr[:-1])]
        else:
            index_2_entity[line_tr[-1]] = "".join(line_tr[:-1])
    fp.close()

    # case_st
    rating_file = args.path.data + 'item_list'
    fp = open(rating_file + '.txt', "r")
    for line in iter(fp):
        line_tr = line.replace('\n', '').split(' ')
        if len(line_tr) > 2:
            if line_tr[0] in new_meta_Books_dict:
                # print('find 2 = ', new_meta_Books_dict[line_tr[0]])
                index_2_entity[line_tr[1]] = new_meta_Books_dict[line_tr[0]]
            else:
                index_2_entity[line_tr[1]] = line_tr[0]
        else:
            pass
            # print('line_tr = ', line_tr)
    fp.close()

    rating_file = args.path.data + 'relation_list'
    fp = open(rating_file + '.txt', "r")
    rela_2_name = {}
    for line in iter(fp):
        # print(line.replace('http://rdf.freebase.com/ns/',''))
        line_tr = line.replace('http://rdf.freebase.com/ns/','')
        line_tr = line_tr.replace('http://www.w3.org/1999/02/22-rdf-','')
        line_tr = line_tr.replace('http://www.w3.org/2000/01/rdf-','')
        line_tr = line_tr.replace('\n', '').split(' ')

        # if len(line_tr) != 2:
        #     print("".join(line_tr[:-1]), line_tr[-1])
        rela_2_name[line_tr[-1]]  = " ".join(line_tr[:-1])
    fp.close()

    return index_2_entity, rela_2_name

def get_all_user_entity_count(args,train_data, kg, adj_entity, adj_relation, hop=0):
    # item_pool = set(train_data[:, 1].tolist())

    # if args.load_pretrain_emb == True and args.save_final_model == True:
    #     with open(f'{args.path.misc}total_kg_exploration_{args.save_model_name}.pickle', 'rb') as handle:
    #         total_kg_exploration = pickle.load(handle)
    # else:
    #     total_kg_exploration = set()

    # item_pool_tmp = item_pool
    # for _ in range(hop):
    #     h_r_t_set = set([tuple([item,tail,rela]) for item in item_pool_tmp for tail, rela in zip(adj_entity[item,:].tolist(),adj_relation[item,:].tolist())])
    #     item_pool_tmp = set([tail for item in item_pool_tmp for tail in adj_entity[item,:].tolist()])
    #     total_kg_exploration |= h_r_t_set

    # if args.save_final_model == True:
    #     with open(f'{args.path.misc}total_kg_exploration_{args.save_model_name}.pickle', 'wb') as handle:
    #         pickle.dump(total_kg_exploration, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # total_kg = set()
    # item_pool_tmp = item_pool
    # for _ in range(hop):
    #     h_r_t_set = set([tuple([item,tail,rela]) for item in item_pool_tmp for tail, rela in kg[item]])
    #     item_pool_tmp = set([tail for item in item_pool_tmp for tail, _ in kg[item]])
    #     total_kg |= h_r_t_set


    # all_neighbor_num = len(total_kg)
    # use_neighbor_num = len(total_kg_exploration)
    # use_neighbor_rate = use_neighbor_num/all_neighbor_num
    # args.use_neighbor_rate = [all_neighbor_num,use_neighbor_num,round(use_neighbor_rate, 6)]
    args.use_neighbor_rate = [0,0,0]

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


def load_kg(args, train_data):
    print('reading KG file ...')

    # reading kg file
    kg_file = args.path.data + 'kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int64)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))

    rating_file = args.path.data + 'ratings_final'
    if os.path.exists(rating_file + '.npy'):
        ratings_final = np.load(rating_file + '.npy')
    else:
        ratings_final = np.loadtxt(rating_file + '.txt', dtype=np.int64)
        np.save(rating_file + '.npy', ratings_final)

    user_num = len(set(ratings_final[:,0]))
    item_num = len(set(ratings_final[:,1]))

    average_user_num = len(ratings_final) / user_num
    arverage_item_num = len(ratings_final) / item_num

    kg, enti, rela = construct_kg(args,kg_np)

    # diiferent 1
    # print('n_entity, n_relation, enti, rela = ',n_entity, n_relation, enti, rela)

    # kg, enti, rela = construct_user_kg(kg, train_data, n_entity, n_relation)
    # n_entity = n_entity + n_entity
    # n_relation = n_relation + 1

    # diiferent 1
    # print('n_entity, n_relation, enti, rela = ', n_entity + user_num, n_relation, enti, rela)
    # input()

    adj_entity, adj_relation, user_path = None, None, None

    adj_entity, adj_relation = construct_adj(args, kg, n_entity)
    user_path_top_k = None

    return kg, n_entity, n_relation, adj_entity, adj_relation, user_path, user_path_top_k


def construct_kg(args,kg_np):
    print('constructing knowledge graph ...')
    kg = dict()
    enti = 0
    rela = 0
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

        enti = max(enti, head, tail)
        rela = max(rela, relation)
    return kg, enti, rela

def construct_user_kg(kg, train_data, item_num, n_relation):
    print('constructing user knowledge graph ...')
    enti = 0
    rela = 0
    for i in range(train_data.shape[0]):
        user = train_data[i][0]
        item = train_data[i][1]
        rating = train_data[i][2]

        if rating == 1:
            head = user + item_num
            tail = item
            if head not in kg:
                kg[head] = []
            kg[head].append((tail, n_relation))
            if tail not in kg:
                kg[tail] = []
            kg[tail].append((head, n_relation))

            enti = max(enti, head, tail)
            rela = max(rela, n_relation)
    return kg, enti, rela



def construct_adj(args, kg, entity_num, random_seed = 1):
    adj_entity, adj_relation = contruct_random_adj(args,kg,entity_num)
    return adj_entity, adj_relation


def contruct_random_adj(args,kg,entity_num):

    adj_entity = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    adj_relation = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    for entity in range(entity_num):
        if entity in kg:
            neighbors = kg[entity]
            n_neighbors = len(neighbors)
            if n_neighbors >= args.neighbor_sample_size: sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=False)
            else: sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=True)
            adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
            adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])

    return adj_entity, adj_relation
                        


def get_user_triplet_set(args, kg, user_history_dict):
    print('constructing ripple set ...')
    # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    user_triplet_set = collections.defaultdict(list)
    # entity_interaction_dict = collections.defaultdict(list)
    global g_kg
    g_kg = kg
    with mp.Pool(processes=min(mp.cpu_count(), 12)) as pool:
        job = partial(_get_user_triplet_set, p_hop=max(1,args.p_hop), n_memory=args.n_memory, n_neighbor=16)
        for u, u_r_set, u_interaction_list in pool.starmap(job, user_history_dict.items()):
            user_triplet_set[u] = np.array(u_r_set, dtype=np.int32)
            # entity_interaction_dict[u] = u_interaction_list
    del g_kg
    return user_triplet_set

def _get_user_triplet_set(user, history, p_hop=2, n_memory=32, n_neighbor=16):
    ret = []
    entity_interaction_list = []
    for h in range(max(1,p_hop)):
        memories_h = []
        memories_r = []
        memories_t = []

        if h == 0:
            tails_of_last_hop = history
        else:
            tails_of_last_hop = ret[-1][2]

        for entity in tails_of_last_hop:
            for tail_and_relation in random.sample(g_kg[entity], min(len(g_kg[entity]), n_neighbor)):
                memories_h.append(entity)
                memories_r.append(tail_and_relation[1])
                memories_t.append(tail_and_relation[0])

        # if the current ripple set of the given user is empty, we simply copy the ripple set of the last hop here
        # this won't happen for h = 0, because only the items that appear in the KG have been selected
        # this only happens on 154 users in Book-Crossing dataset (since both BX dataset and the KG are sparse)
        if len(memories_h) == 0:
            ret.append(ret[-1])
        else:
            # sample a fixed-size 1-hop memory for each user
            replace = len(memories_h) < n_memory
            indices = np.random.choice(len(memories_h), size=n_memory, replace=replace)
            memories_h = [memories_h[i] for i in indices]
            memories_r = [memories_r[i] for i in indices]
            memories_t = [memories_t[i] for i in indices]
            entity_interaction_list += zip(memories_h, memories_r, memories_t)
            ret.append([memories_h, memories_r, memories_t])
            
    return [user, ret, list(set(entity_interaction_list))]