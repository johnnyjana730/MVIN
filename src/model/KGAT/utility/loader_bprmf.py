'''
Created on Dec 18, 2018
Tensorflow Implementation of the Baseline Model, BPRMF, in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
from utility.load_data import Data

class BPRMF_loader(Data):
    def __init__(self, args, path):
        super().__init__(args, path)

    # def generate_train_batch(self):
    #     users, pos_items, neg_items = self._generate_train_cf_batch()

    #     batch_data = {}
    #     batch_data['users'] = users
    #     batch_data['pos_items'] = pos_items
    #     batch_data['neg_items'] = neg_items

    #     return batch_data

    def generate_rating_batch(self, data ,start, end):
        # heads, relations, pos_tails, neg_tails = self._generate_train_kg_batch()

        batch_data = {}
        batch_data['users'] = data[start:end, 0]
        batch_data['items'] = data[start:end, 1]
        batch_data['labels'] = data[start:end, 2]

        # batch_data['heads'] = heads
        # batch_data['relations'] = relations
        # batch_data['pos_tails'] = pos_tails
        # batch_data['neg_tails'] = neg_tails

        return batch_data

    # def generate_train_feed_dict(self, model, batch_data):
    #     feed_dict = {
    #         model.users: batch_data['users'],
    #         model.pos_items: batch_data['pos_items'],
    #         model.neg_items: batch_data['neg_items']
    #     }

    #     return feed_dict

    def generate_feed_rating_dict(self, model, batch_data):
        feed_dict ={
            # model.u: batch_data['users'],
            # model.pos_i: batch_data['pos_items'],
            # model.neg_i: batch_data['neg_items'],

            model.user_indices: batch_data['users'],
            model.item_indices: batch_data['items'],
            model.labels: batch_data['labels'],
            
            # model.h: batch_data['heads'],
            # model.r: batch_data['relations'],
            # model.pos_t: batch_data['pos_tails'],
            # model.neg_t: batch_data['neg_tails']
        }

        return feed_dict


    def generate_test_feed_dict(self, model, user_batch, item_batch, drop_flag=False):
        feed_dict = {
            model.users: user_batch,
            model.pos_items: item_batch
        }
        return feed_dict

