'''
Created on Dec 18, 2018
Tensorflow Implementation of the Baseline Model, BPRMF, in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import tensorflow as tf
import os
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score, roc_auc_score

# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class BPRMF(object):
    def __init__(self, data_config, pretrain_data, args):
        self.model_type = 'mf'
        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.regs = eval(args.regs)

        self.verbose = args.verbose

        # Placeholder definition
        self.users = tf.placeholder(tf.int32, shape=[None,], name='users')
        self.pos_items = tf.placeholder(tf.int32, shape=[None,], name='pos_items')
        self.neg_items = tf.placeholder(tf.int32, shape=[None,], name='neg_items')

        # Variable definition
        self.weights = self._init_weights()

        # Original embedding.

        self.user_indices = tf.placeholder(dtype=tf.int32, shape=[None], name='user_indices')
        self.item_indices = tf.placeholder(dtype=tf.int32, shape=[None], name='item_indices')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')

        u_e = tf.nn.embedding_lookup(self.weights['user_embedding'], self.user_indices)
        i_e = tf.nn.embedding_lookup(self.weights['item_embedding'], self.item_indices)
        # neg_i_e = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)
# 
        # All predictions for all users.
        # self.batch_predictions = tf.matmul(u_e, pos_i_e, transpose_a=False, transpose_b=True)

        # Optimization process.
        self.base_loss, self.reg_loss = self._create_bpr_loss(u_e, i_e)
        self.kge_loss = tf.constant(0.0, tf.float32, [1])
        self.loss = self.base_loss + self.kge_loss + self.reg_loss

        # self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.loss)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        self._statistics_params()


    def _init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        if self.pretrain_data is None:
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
            print('using xavier initialization')
        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                    name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                    name='item_embedding', dtype=tf.float32)
            print('using pretrained initialization')
        return all_weights


    def _create_bpr_loss(self, users, i_items):


        self.scores = tf.reduce_sum(tf.multiply(users, i_items), axis=1)
        self.scores_normalized = tf.sigmoid(self.scores)

        base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.scores))

        # neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(i_items)

        # maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))

        # mf_loss = tf.negative(tf.reduce_mean(maxi))
        reg_loss = self.regs[0] * regularizer
# 
        return base_loss, reg_loss


    def _statistics_params(self):
        # number of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            print("#params: %d" % total_parameters)

    def train(self, sess, feed_dict):
        return sess.run([self.opt, self.loss, self.base_loss, self.kge_loss, self.reg_loss], feed_dict)

    # def eval(self, sess, feed_dict):
    #     batch_predictions = sess.run(self.batch_predictions, feed_dict)
    #     return batch_predictions
    
    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        acc = np.mean(np.equal(scores, labels))
        return auc, acc, f1

    def get_scores(self, sess, feed_dict):
        return sess.run([self.item_indices, self.scores_normalized], feed_dict)