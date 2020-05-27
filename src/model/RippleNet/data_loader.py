import collections
import os
import numpy as np
import pandas as pd
import pickle
import time
import multiprocessing as mp
from functools import partial
import random

def load_data(args):
    train_data, eval_data, test_data, user_history_dict, item_set_most_pop, n_item, n_user, n_interaction, avg_u_inte, avg_i_inte = load_rating(args)
    n_entity, n_relation, n_triple, kg = load_kg(args)
    if args.show_save_dataset_info:
        print('*' * 120)
        print(f'user({n_user}) item({n_item}) inte({n_interaction}) avg u inte({round(avg_u_inte, 1)}) avg i inte({round(avg_i_inte, 1)})')
        print(f'enti({n_entity}) rela({n_relation}) trip({n_triple})')
        print('*' * 120)
        with open(args.path.data + 'info.txt', 'w') as f:
            f.write(f'user({n_user}) item({n_item}) inte({n_interaction}) avg u inte({avg_u_inte}) avg i inte({avg_i_inte}) ')
            f.write(f'enti({n_entity}) rela({n_relation}) trip({n_triple})')
            
    ripple_set, user_ere_interaction_dict = get_ripple_set(args, kg, user_history_dict)
    all_user_entity_count = get_all_user_entity_count(user_history_dict, kg, hop=args.n_hop)

    return train_data, eval_data, test_data, n_item, n_user, n_entity, n_relation, ripple_set, user_ere_interaction_dict, all_user_entity_count, item_set_most_pop


def load_rating(args):
    print('reading rating file ...')
    # reading rating file
    rating_file = args.path.data + 'ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
        np.save(rating_file + '.npy', rating_np)

    # if args.topk_eval and not os.path.exists(f'{args.path.misc}pop_item_{args.n_pop_item_eval}.pickle'):
    #     with open(f'{args.path.misc}pop_item_{args.n_pop_item_eval}.pickle', 'wb') as f:
    #         item_counter = {}
    #         for _, i, _ in rating_np:
    #             if i not in item_counter:
    #                 item_counter[i] = 0
    #             item_counter[i] += 1

    #         item_counter = sorted(item_counter.items(), key=lambda x: x[1], reverse=True)
    #         pop_item = [i for i, _ in item_counter[:args.n_pop_item_eval]]
    #         pickle.dump(pop_item, f)

    if True or os.path.exists(f'{args.path.misc}pop_item_{args.n_pop_item_eval}.pickle') == True:
        item_count = {}
        for i in range(rating_np.shape[0]):
            item = rating_np[i, 1]
            if item not in item_count:
                item_count[item] = 0
            item_count[item] += 1
        item_count = sorted(item_count.items(), key=lambda x: x[1], reverse=True)
        item_count = item_count[:500]
        item_set_most_pop = [item_set[0] for item_set in item_count]
        with open(f'{args.path.misc}pop_item_{args.n_pop_item_eval}.pickle', 'wb') as fp:
            pickle.dump(item_set_most_pop, fp)

    item_set_most_pop = set(item_set_most_pop)
    # with open(f"{args.path.misc}KGNN_pop_item_set_500.pickle", 'rb') as fp:
    #     item_set_most_pop = pickle.load(fp)


    n_user, n_item, n_interaction = max(set(rating_np[:, 0])), max(set(rating_np[:, 1])), rating_np.shape[0]
    u_counter, i_counter = {}, {}
    # if True or not os.path.isfile(args.path.data + 'train_pd.csv'):
    if args.new_load_data == True:
        print('load new train eval test')
        train_data, eval_data, test_data, user_history_dict = dataset_split(rating_np)
        with open(args.path.data + 'train_data.pickle','wb') as f:
            pickle.dump(train_data, f)
        with open(args.path.data + 'eval_data.pickle','wb') as f:
            pickle.dump(eval_data, f)
        with open(args.path.data + 'test_data.pickle','wb') as f:
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

    print('bf train_data, eval_data, test_data = ', len(train_data), len(eval_data), len(test_data))
    train_indices = [i for i in range(train_data.shape[0]) if train_data[i][0] in user_history_dict]
    eval_indices = [i for i in range(eval_data.shape[0]) if eval_data[i][0] in user_history_dict]
    test_indices = [i for i in range(test_data.shape[0]) if test_data[i][0] in user_history_dict]

    train_data = train_data[train_indices]
    eval_data = eval_data[eval_indices]
    test_data = test_data[test_indices]

    # print('af train_data, eval_data, test_data = ', len(train_data), len(eval_data), len(test_data))
    # input()

    # for t_i in range(rating_np.shape[0]):
    #     user = rating_np[t_i][0]
    #     item = rating_np[t_i][1]
    #     rating = rating_np[t_i][2]

    #     if user not in u_counter: u_counter[user] = set()
    #     if item not in i_counter: i_counter[item] = set()
    #     u_counter[user].add(item)
    #     i_counter[item].add(user)
    # total_u_interaction = np.sum([len(item_set) for item_set in u_counter.values()])
    # total_i_interaction = np.sum([len(user_set) for user_set in i_counter.values()])
    # avg_u_inte = total_i_interaction / n_user
    # avg_i_inte = total_u_interaction / n_item

    return train_data, eval_data, test_data, user_history_dict, item_set_most_pop,  n_item, n_user, n_interaction, 0, 0

def load_pre_data(args):
    train_data = pd.read_csv(args.path.data + 'train_pd.csv',index_col=None)
    train_data = train_data.drop(train_data.columns[0], axis=1)
    train_data = train_data[['user','item','like']].values
    eval_data = pd.read_csv(args.path.data + 'eval_pd.csv',index_col=None)
    eval_data = eval_data.drop(eval_data.columns[0], axis=1)
    eval_data = eval_data[['user','item','like']].values
    test_data = pd.read_csv(args.path.data + 'test_pd.csv',index_col=None)
    test_data = test_data.drop(test_data.columns[0], axis=1)
    test_data = test_data[['user','item','like']].values
    return train_data, eval_data, test_data

def dataset_split(rating_np):
    print('splitting dataset ...')

    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(n_ratings, size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))

    # traverse training data, only keeping the users with positive ratings
    user_history_dict = dict()
    for i in train_indices:
        user = rating_np[i][0]
        item = rating_np[i][1]
        rating = rating_np[i][2]

        if rating == 1:
            if user not in user_history_dict:
                user_history_dict[user] = []
            user_history_dict[user].append(item)

    train_indices = [i for i in train_indices if rating_np[i][0] in user_history_dict]
    eval_indices = [i for i in eval_indices if rating_np[i][0] in user_history_dict]
    test_indices = [i for i in test_indices if rating_np[i][0] in user_history_dict]

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data, user_history_dict


def load_kg(args):
    print('reading KG file ...')

    kg_file = args.path.data + 'kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))
    n_triple = kg_np.shape[0]

    kg = construct_kg(kg_np)

    return n_entity, n_relation, n_triple, kg


def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg = collections.defaultdict(list)
    for head, relation, tail in kg_np:
        kg[head].append((tail, relation))
    return kg

def get_all_user_entity_count(user_history_dict, kg, hop=0):
    tails = set()
    for items in user_history_dict.values():
        tails.update(items)

    total = set()
    for _ in range(hop):            
        tmp = set([(i, r, t) for i in tails for t, r in kg[i]])
        tails = set([t for _, _, t in tmp])
        total |= tmp
    return len(total)

def get_ripple_set(args, kg, user_history_dict):
    print('constructing ripple set ...')
    # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    ripple_set = collections.defaultdict(list)
    user_ere_interaction_dict = collections.defaultdict(list)
    global g_kg
    g_kg = kg
    with mp.Pool(processes=min(mp.cpu_count(), 8)) as pool:
        job = partial(_get_ripple_set, n_hop=args.n_hop, n_memory=args.n_memory, n_neighbor=16)
        for u, u_r_set, u_interaction_list in pool.starmap(job, user_history_dict.items()):
            ripple_set[u] = np.array(u_r_set, dtype=np.int32)
            user_ere_interaction_dict[u] = u_interaction_list
    del g_kg
    return ripple_set, user_ere_interaction_dict

def _get_ripple_set(user, history, n_hop=2, n_memory=32, n_neighbor=16):
    ret = []
    ere_interaction_list = []
    for h in range(n_hop):
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
            ere_interaction_list += zip(memories_h, memories_r, memories_t)
            ret.append([memories_h, memories_r, memories_t])
            
    return [user, ret, list(set(ere_interaction_list))]
