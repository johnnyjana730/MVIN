import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import time

class RippleNet(object):
    def __init__(self, args, n_entity, n_relation):
        # self._next_element = self.iterator.get_next()
        self._parse_args(args, n_entity, n_relation)
        self._build_inputs()
        self._build_embeddings()
        self._build_model()
        self._build_loss()
        self._build_train()

    def _parse_args(self, args, n_entity, n_relation):
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        self.n_memory = args.n_memory
        self.item_update_mode = args.item_update_mode
        self.using_all_hops = args.using_all_hops

        self.emb_path = args.path.emb

        if args.emb_name == '':
            self.emb_name = f'h${args.n_hop}_d${args.dim}_m${args.n_memory}_sw'
        else:
            self.emb_name = args.emb_name
    
        self.pretrained_enti_emb = f'{self.emb_path}enti_emb_{self.emb_name}.npy'
        self.pretrained_rela_emb = f'{self.emb_path}rela_emb_{self.emb_name}.npy'

    def _build_inputs(self):
        # self.items = tf.placeholder(dtype=tf.int32, shape=[None], name='items')
        # self.labels = tf.placeholder(dtype=tf.float64, shape=[None], name='labels')
        
        # d = self.iterator.get_next()
        # for k, v in d.items():
        #     setattr(self, k, v)
        # self.memories_h = [d['memories_h_' + str(hop)] for hop in range(self.n_hop)]
        # self.memories_r = [d['memories_r_' + str(hop)] for hop in range(self.n_hop)]
        # self.memories_t = [d['memories_t_' + str(hop)] for hop in range(self.n_hop)]
        
        self.items = tf.placeholder(dtype=tf.int32, shape=[None], name="items")
        self.labels = tf.placeholder(dtype=tf.float64, shape=[None], name="labels")
        self.memories_h = []
        self.memories_r = []
        self.memories_t = []

        for hop in range(self.n_hop):
            self.memories_h.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_h_" + str(hop)))
            self.memories_r.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_r_" + str(hop)))
            self.memories_t.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_t_" + str(hop)))

    def save_pretrained_emb(self, sess):
        enti_emb = sess.run(self.entity_emb_matrix)
        np.save(self.pretrained_enti_emb, enti_emb)

        rela_emb = sess.run(self.relation_emb_matrix)
        np.save(self.pretrained_rela_emb, rela_emb)
        # print(self.pretrained_enti_emb)
        # print(self.pretrained_rela_emb)

    def initialize_pretrained_embeddings(self, sess):
        # print(self.pretrained_enti_emb)
        # print(self.pretrained_rela_emb)
        embeddings = np.load(self.pretrained_rela_emb)
        _ = sess.run((self.relation_emb_init),
            feed_dict={self.relation_emb_placeholder: embeddings})

        embeddings = np.load(self.pretrained_enti_emb)
        _ = sess.run((self.entity_emb_init),
            feed_dict={self.entity_emb_placeholder: embeddings})


    def _build_embeddings(self):
        with tf.variable_scope("entity_emb_matrix"):
            self.entity_emb_placeholder = tf.placeholder(
                                                    dtype=tf.float64,
                                                    shape=[self.n_entity, self.dim]
                                                    )     
            self.entity_emb_matrix = tf.get_variable(
                                                    name="entity_emb_matrix",
                                                    dtype=tf.float64,
                                                    shape=[self.n_entity, self.dim],
                                                    initializer=tf.contrib.layers.xavier_initializer(),
                                                    )
            self.entity_emb_init = self.entity_emb_matrix.assign(self.entity_emb_placeholder)

        with tf.variable_scope("relation_emb_matrix"):
            self.relation_emb_placeholder = tf.placeholder(
                                                    dtype=tf.float64,
                                                    shape=[self.n_relation, self.dim, self.dim]
                                                    )     
            self.relation_emb_matrix = tf.get_variable(
                                                    name="relation_emb_matrix",
                                                    dtype=tf.float64,
                                                    shape=[self.n_relation, self.dim, self.dim],
                                                    initializer=tf.contrib.layers.xavier_initializer()
                                                    )
            self.relation_emb_init = self.relation_emb_matrix.assign(self.relation_emb_placeholder)

    def _build_model(self):
        # transformation matrix for updating item embeddings at the end of each hop
        self.transform_matrix = tf.get_variable(name="transform_matrix", shape=[self.dim, self.dim], dtype=tf.float64,
                                                initializer=tf.contrib.layers.xavier_initializer())

        # [batch size, dim]
        self.item_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.items)

        self.h_emb_list = []
        self.r_emb_list = []
        self.t_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            self.h_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_h[i]))

            # [batch size, n_memory, dim, dim]
            self.r_emb_list.append(tf.nn.embedding_lookup(self.relation_emb_matrix, self.memories_r[i]))

            # [batch size, n_memory, dim]
            self.t_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_t[i]))

        o_list = self._key_addressing()

        self.scores = tf.squeeze(self.predict(self.item_embeddings, o_list))
        self.scores_normalized = tf.sigmoid(self.scores)

    def _key_addressing(self):
        o_list = []
        for hop in range(self.n_hop):
            # [batch_size, n_memory, dim, 1]
            h_expanded = tf.expand_dims(self.h_emb_list[hop], axis=3)

            # [batch_size, n_memory, dim]
            Rh = tf.squeeze(tf.matmul(self.r_emb_list[hop], h_expanded), axis=3)

            # [batch_size, dim, 1]
            v = tf.expand_dims(self.item_embeddings, axis=2)

            # [batch_size, n_memory]
            probs = tf.squeeze(tf.matmul(Rh, v), axis=2)

            # [batch_size, n_memory]
            probs_normalized = tf.nn.softmax(probs)

            # [batch_size, n_memory, 1]
            probs_expanded = tf.expand_dims(probs_normalized, axis=2)

            # [batch_size, dim]
            o = tf.reduce_sum(self.t_emb_list[hop] * probs_expanded, axis=1)

            self.item_embeddings = self.update_item_embedding(self.item_embeddings, o)
            o_list.append(o)
        return o_list

    def update_item_embedding(self, item_embeddings, o):
        if self.item_update_mode == "replace":
            item_embeddings = o
        elif self.item_update_mode == "plus":
            item_embeddings = item_embeddings + o
        elif self.item_update_mode == "replace_transform":
            item_embeddings = tf.matmul(o, self.transform_matrix)
        elif self.item_update_mode == "plus_transform":
            item_embeddings = tf.matmul(item_embeddings + o, self.transform_matrix)
        else:
            raise Exception("Unknown item updating mode: " + self.item_update_mode)
        return item_embeddings

    def predict(self, item_embeddings, o_list):
        y = o_list[-1]
        if self.using_all_hops:
            for i in range(self.n_hop - 1):
                y += o_list[i]

        # [batch_size]
        scores = tf.reduce_sum(item_embeddings * y, axis=1)
        return scores

    def _build_loss(self):
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.scores))

        self.kge_loss = 0
        for hop in range(self.n_hop):
            h_expanded = tf.expand_dims(self.h_emb_list[hop], axis=2)
            t_expanded = tf.expand_dims(self.t_emb_list[hop], axis=3)
            hRt = tf.squeeze(tf.matmul(tf.matmul(h_expanded, self.r_emb_list[hop]), t_expanded))
            self.kge_loss += tf.reduce_mean(tf.sigmoid(hRt))
        self.kge_loss = -self.kge_weight * self.kge_loss

        self.l2_loss = 0
        for hop in range(self.n_hop):
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.h_emb_list[hop] * self.h_emb_list[hop]))
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.t_emb_list[hop] * self.t_emb_list[hop]))
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.r_emb_list[hop] * self.r_emb_list[hop]))
            if self.item_update_mode == "replace nonlinear" or self.item_update_mode == "plus nonlinear":
                self.l2_loss += tf.nn.l2_loss(self.transform_matrix)
        self.l2_loss = self.l2_weight * self.l2_loss

        self.loss = self.base_loss + self.kge_loss + self.l2_loss

    def _build_train(self):
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    # def eval(self, sess, feed_dict):
    #     labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
    #     if len(np.unique(labels)) == 1: # bug in roc_auc_score
    #         auc = accuracy_score(labels, np.rint(scores))
    #     else:
    #         auc = roc_auc_score(y_true=labels, y_score=scores)
    #     scores[scores >= 0.5] = 1
    #     scores[scores < 0.5] = 0

    #     f1 = f1_score(y_true=labels, y_pred=scores)
    #     acc = np.mean(np.equal(scores, labels))
    #     return auc, acc, f1

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        acc = np.mean(np.equal(scores, labels))
        return auc, acc, f1
    
    def get_scores(self, sess, feed_dict):
        return sess.run([self.items, self.scores_normalized], feed_dict)
