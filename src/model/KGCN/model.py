import tensorflow as tf
from aggregators import SumAggregator, ConcatAggregator, NeighborAggregator, LabelAggregator
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
# tf.set_random_seed(1)

class KGCN(object):
    def __init__(self, args, n_user, n_entity, n_relation, adj_entity, adj_relation):
        self._parse_args(args, adj_entity, adj_relation)
        self._build_inputs()
        self._build_model(n_user, n_entity, n_relation)
        self._build_train()

    @staticmethod
    def get_initializer():
        return tf.contrib.layers.xavier_initializer()

    def _parse_args(self, args, adj_entity, adj_relation):
        # [entity_num, neighbor_sample_size]
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation
        self.dataset = args.dataset

        self.load_pretrain_emb = args.load_pretrain_emb
        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.n_neighbor = args.neighbor_sample_size
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        self.save_model_name = args.save_model_name

        self.path = args.path

        self.pretrained_embeddings_action = f"{self.path.emb}rela_tr_emb_test_{self.save_model_name}.npy"
        self.pretrained_embeddings_entity = f"{self.path.emb}enti_tr_emb_test_{self.save_model_name}.npy"
        self.pretrained_embeddings_user = f"{self.path.emb}user_tr_emb_test_{self.save_model_name}.npy"

        if args.aggregator == 'sum':
            self.aggregator_class = SumAggregator
        elif args.aggregator == 'concat':
            self.aggregator_class = ConcatAggregator
        elif args.aggregator == 'neighbor':
            self.aggregator_class = NeighborAggregator
        else:
            raise Exception("Unknown aggregator: " + args.aggregator)

    def _build_inputs(self):
        self.user_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='user_indices')
        self.item_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='item_indices')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')
        self.lr_placeholder = tf.placeholder(tf.float32, [], name='learning_rate')

    def save_pretrain_emb_fuc(self,sess):

        enti_emb = sess.run(self.entity_emb_matrix)
        np.save(f"{self.path.emb}enti_tr_emb_test_{self.save_model_name}.npy", enti_emb)
        rela_emb = sess.run(self.relation_emb_matrix)
        np.save(f"{self.path.emb}rela_tr_emb_test_{self.save_model_name}.npy", rela_emb)
        user_emb = sess.run(self.user_emb_matrix)
        np.save(f"{self.path.emb}user_tr_emb_test_{self.save_model_name}.npy", user_emb)


    def initialize_pretrained_embeddings(self, sess):
        if self.pretrained_embeddings_action != '':
            if self.load_pretrain_emb == True:
                embeddings = np.load(self.pretrained_embeddings_action)
                print('load pretrained action emb', self.pretrained_embeddings_action)
                _ = sess.run((self.relation_embedding_init),
                             feed_dict={self.action_embedding_placeholder: embeddings})

        if self.pretrained_embeddings_entity != '':
            if self.load_pretrain_emb == True:
                print('load pretrained entity emb', self.pretrained_embeddings_entity)
                embeddings = np.load(self.pretrained_embeddings_entity)
                _ = sess.run((self.entity_embedding_init),
                             feed_dict={self.entity_embedding_placeholder: embeddings})

        if self.pretrained_embeddings_user != '':
            if self.load_pretrain_emb == True:
                print('load pretrained user emb')
                embeddings = np.load(self.pretrained_embeddings_user)
                _ = sess.run((self.user_embedding_init),
                              feed_dict={self.user_embedding_placeholder: embeddings})

    def _build_model(self, n_user, n_entity, n_relation):
        self.n_entity = n_entity

        with tf.variable_scope("user_emb_matrix"):
            self.user_embedding_placeholder = tf.placeholder(tf.float32, [n_user, self.dim])
            self.user_emb_matrix = tf.get_variable(
                shape=[n_user, self.dim], initializer=KGCN.get_initializer(), name='user_emb_matrix')
            self.user_embedding_init = self.user_emb_matrix.assign(self.user_embedding_placeholder)

        with tf.variable_scope("entity_emb_matrix"):
            self.entity_embedding_placeholder = tf.placeholder(tf.float32, [n_entity, self.dim])
            self.entity_emb_matrix = tf.get_variable(
                shape=[n_entity, self.dim], initializer=KGCN.get_initializer(), name='entity_emb_matrix')
            self.entity_embedding_init = self.entity_emb_matrix.assign(self.entity_embedding_placeholder)

        with tf.variable_scope("relation_emb_matrix"):
            self.action_embedding_placeholder = tf.placeholder(tf.float32,
                                    [n_relation, self.dim])
            self.relation_emb_matrix = tf.get_variable(
                shape=[n_relation, self.dim], initializer=KGCN.get_initializer(), name='relation_emb_matrix')
            self.relation_embedding_init = self.relation_emb_matrix.assign(self.action_embedding_placeholder)

        # [batch_size, dim]
        self.user_embeddings = tf.nn.embedding_lookup(self.user_emb_matrix, self.user_indices)

        entities, relations = self.get_neighbors(self.item_indices)
        # [batch_size, dim]
        self.item_embeddings, self.aggregators = self.aggregate(entities, relations)

        # [batch_size]
        self.scores = tf.reduce_sum(self.user_embeddings * self.item_embeddings, axis=1)
        self.scores_normalized = tf.sigmoid(self.scores)

    def get_neighbors(self, seeds):
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        relations = []
        for i in range(self.n_iter):
            neighbor_entities = tf.reshape(tf.gather(self.adj_entity, entities[i]), [self.batch_size, -1])
            neighbor_relations = tf.reshape(tf.gather(self.adj_relation, entities[i]), [self.batch_size, -1])
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    # feature propagation
    def aggregate(self, entities, relations):
        aggregators = []  # store all aggregators
        entity_vectors = [tf.nn.embedding_lookup(self.entity_emb_matrix, i) for i in entities]
        relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                aggregator = self.aggregator_class(self.save_model_name,self.batch_size, self.dim, act=tf.nn.tanh, name = i)
            else:
                aggregator = self.aggregator_class(self.save_model_name,self.batch_size, self.dim, name = i)
            aggregators.append(aggregator)


            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = [self.batch_size, -1, self.n_neighbor, self.dim]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vectors=tf.reshape(entity_vectors[hop + 1], shape),
                                    neighbor_relations=tf.reshape(relation_vectors[hop], shape),
                                    user_embeddings=self.user_embeddings,
                                    masks=None)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        res = tf.reshape(entity_vectors[0], [self.batch_size, self.dim])

        return res, aggregators

    def _build_train(self):
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.scores))

        self.l2_loss = tf.nn.l2_loss(self.user_emb_matrix) + tf.nn.l2_loss(
            self.entity_emb_matrix) + tf.nn.l2_loss(self.relation_emb_matrix)
        for aggregator in self.aggregators:
            self.l2_loss = self.l2_loss + tf.nn.l2_loss(aggregator.weights)
        self.loss = self.base_loss + self.l2_weight * self.l2_loss 

        self.l2_loss_final = self.l2_weight * self.l2_loss

        self.optimizer = tf.train.AdamOptimizer(self.lr_placeholder).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss, self.l2_loss_final], feed_dict)

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
