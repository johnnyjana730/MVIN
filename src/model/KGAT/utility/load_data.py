'''
Created on Dec 18, 2018
Tensorflow Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import collections
import numpy as np
import pandas as pd
import random as rd
import os

class Data(object):
    def __init__(self, args, path):
        self.path = path
        self.args = args

        self.batch_size = args.batch_size

        train_file = path + '/train_pd.csv'
        eval_file = path + '/eval_pd.csv'
        test_file = path + '/test_pd.csv'

        kg_file = path + '/kg_final.txt'

        # ----------get number of users and items & then load rating data from train_file & test_file------------.
        self.n_train, self.n_test = 0, 0
        self.n_users, self.n_items = 0, 0

        # self.train_data, self.train_user_dict = self._load_ratings(train_file)
        # self.test_data, self.test_user_dict = self._load_ratings(test_file)

        rating_file = path + '/ratings_final'
        if os.path.exists(rating_file + '.npy'):
            rating_np = np.load(rating_file + '.npy')
        else:
            rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
            np.save(rating_file + '.npy', rating_np)

        n_user, n_item, n_interaction = max(set(rating_np[:, 0])), max(set(rating_np[:, 1])), rating_np.shape[0]

        item_count = {}
        for i in range(rating_np.shape[0]):
            item = rating_np[i, 1]
            if item not in item_count:
                item_count[item] = 0
            item_count[item] += 1
        item_count = sorted(item_count.items(), key=lambda x: x[1], reverse=True)
        item_count = item_count[:500]
        item_set_most_pop = [item_set[0] for item_set in item_count]
        self.item_set_most_pop = set(item_set_most_pop)


        self.train_ratings = self._load_ratings_csv(train_file)
        self.eval_ratings = self._load_ratings_csv(eval_file)
        self.test_ratings = self._load_ratings_csv(test_file)

        user_history_dict = {}
        for i in range(self.train_ratings.shape[0]):
            user = self.train_ratings[i][0]
            item = self.train_ratings[i][1]
            rating = self.train_ratings[i][2]
            if rating == 1:
                if user not in user_history_dict:
                    user_history_dict[user] = []
                user_history_dict[user].append(item)

        train_indices = [i for i in range(self.train_ratings.shape[0]) if self.train_ratings[i][0] in user_history_dict]
        eval_indices = [i for i in range(self.eval_ratings.shape[0]) if self.eval_ratings[i][0] in user_history_dict]
        test_indices = [i for i in range(self.test_ratings.shape[0]) if self.test_ratings[i][0] in user_history_dict]

        self.train_ratings = self.train_ratings[train_indices]
        self.eval_ratings = self.eval_ratings[eval_indices]
        self.test_ratings = self.test_ratings[test_indices]

        self.train_data, self.train_user_dict, self.train_user_neg_dict, self.train_ratings = self.load_pre_data(self.train_ratings)
        self.eval_data, self.eval_user_dict, self.eval_user_neg_dict, self.eval_ratings = self.load_pre_data(self.eval_ratings)
        self.test_data, self.test_user_dict, self.test_user_neg_dict, self.test_ratings = self.load_pre_data(self.test_ratings)

        self.user_list, self.train_record, self.eval_record, self.test_record, self.item_set, self.k_list = self._topk_settings(self.train_ratings, \
            self.eval_ratings, self.test_ratings, n_item)

        self.exist_users = self.train_user_dict.keys()

        self._statistic_ratings()

        # ----------get number of entities and relations & then load kg data from kg_file ------------.
        self.n_relations, self.n_entities, self.n_triples = 0, 0, 0
        self.kg_data, self.kg_dict, self.relation_dict = self._load_kg(kg_file)

        # ----------print the basic info about the dataset-------------.
        self.batch_size_kg = self.n_triples // (self.n_train // self.batch_size)
        self._print_data_info()

    def _get_user_record(self, data, is_train):
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


    def _topk_settings(self, train_data, eval_data, test_data, n_item):
        user_num = 250
        k_list = [1, 2, 5, 10, 25, 50, 100]
        train_record = self._get_user_record(train_data, True)
        test_record = self._get_user_record(test_data, False)
        eval_record = self._get_user_record(eval_data, False)

        user_list = list(set(train_record.keys()) & (set(test_record.keys()) & set(eval_record.keys())))
        user_counter_dict = {user:len(train_record[user]) for user in user_list}
        user_counter_dict = sorted(user_counter_dict.items(), key=lambda x: x[1], reverse=True)
        user_counter_dict = user_counter_dict[:user_num]
        user_list = [user_set[0] for user_set in user_counter_dict]

        if len(user_list) > user_num:
            user_list = np.random.choice(user_list, size=user_num, replace=False)

        item_set = set(list(range(n_item)))
        return user_list, train_record, eval_record, test_record, item_set, k_list

    # reading train & test interaction data.
    def _load_ratings(self, file_name):
        user_dict = dict()
        inter_mat = list()

        lines = open(file_name, 'r').readlines()
        for l in lines:
            tmps = l.strip()
            inters = [int(i) for i in tmps.split(' ')]

            u_id, pos_ids = inters[0], inters[1:]
            pos_ids = list(set(pos_ids))

            for i_id in pos_ids:
                inter_mat.append([u_id, i_id])

            if len(pos_ids) > 0:
                user_dict[u_id] = pos_ids
        return np.array(inter_mat), user_dict

    # reading train & test interaction data.
    def _load_ratings_csv(self, file_name):
        csv_data = pd.read_csv(file_name,index_col=None)
        csv_data = csv_data.drop(csv_data.columns[0], axis=1)
        csv_data = csv_data[['user','item','like']].values
        return csv_data


    def load_pre_data(self, csv_data):

        user_dict = dict()
        user_neg_dict = dict()
        inter_mat = list()

        for i in range(csv_data.shape[0]):
            user = csv_data[i][0]
            item = csv_data[i][1]
            rating = csv_data[i][2]
            if rating == 1:
                inter_mat.append([user, item])
                if user not in user_dict:
                    user_dict[user] = []
                user_dict[user].append(item)
            elif rating == 0:
                if user not in user_neg_dict:
                    user_neg_dict[user] = []
                user_neg_dict[user].append(item)

        return np.array(inter_mat), user_dict, user_neg_dict, csv_data

    def _statistic_ratings(self):
        self.n_users = max(max(self.train_data[:, 0]), max(self.test_data[:, 0])) + 1
        self.n_items = max(max(self.train_data[:, 1]), max(self.test_data[:, 1])) + 1
        self.n_train = len(self.train_data)
        self.n_test = len(self.test_data)

    # reading train & test interaction data.
    def _load_kg(self, file_name):
        def _construct_kg(kg_np):
            kg = collections.defaultdict(list)
            rd = collections.defaultdict(list)

            for head, relation, tail in kg_np:
                kg[head].append((tail, relation))
                rd[relation].append((head, tail))
            return kg, rd

        kg_np = np.loadtxt(file_name, dtype=np.int32)
        kg_np = np.unique(kg_np, axis=0)

        # self.n_relations = len(set(kg_np[:, 1]))
        # self.n_entities = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
        self.n_relations = max(kg_np[:, 1]) + 1
        self.n_entities = max(max(kg_np[:, 0]), max(kg_np[:, 2])) + 1
        self.n_triples = len(kg_np)

        kg_dict, relation_dict = _construct_kg(kg_np)

        return kg_np, kg_dict, relation_dict

    def _print_data_info(self):
        print('[n_users, n_items]=[%d, %d]' % (self.n_users, self.n_items))
        print('[n_train, n_test]=[%d, %d]' % (self.n_train, self.n_test))
        print('[n_entities, n_relations, n_triples]=[%d, %d, %d]' % (self.n_entities, self.n_relations, self.n_triples))
        print('[batch_size, batch_size_kg]=[%d, %d]' % (self.batch_size, self.batch_size_kg))

    def _generate_train_cf_batch(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_user_dict[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_i_id = np.random.randint(low=0, high=self.n_items,size=1)[0]

                if neg_i_id not in self.train_user_dict[u] and neg_i_id not in neg_items:
                    neg_items.append(neg_i_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items


    def _generate_train_cf_rating_batch(self):

        self.train_ratings

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state



    def create_sparsity_split(self):
        all_users_to_test = list(self.test_user_dict.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_user_dict[uid]
            test_iids = self.test_user_dict[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)


        return split_uids, split_state