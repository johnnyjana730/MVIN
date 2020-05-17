import tensorflow as tf
from aggregators import SumAggregator_urh_matrix
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np

class MVIN(object):
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
        self.h_hop = args.h_hop
        self.batch_size = args.batch_size
        self.n_neighbor = args.neighbor_sample_size
        self.p_hop = args.p_hop
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.l2_agg_weight = args.l2_agg_weight
        self.kge_weight = args.kge_weight
        self.lr = args.lr
        self.save_model_name = args.save_model_name
        self.n_mix_hop = args.n_mix_hop
        self.n_memory = args.n_memory
        self.update_item_emb = args.update_item_emb
        self.h0_att = args.h0_att
        self.path = args.path
        self.User_orient_rela = args.User_orient_rela

        self.args = args
        self.act= tf.nn.relu

        self.aggregator_class = SumAggregator_urh_matrix

        if self.args.wide_deep == True: self.agg_fun = self.aggregate_delta_whole
        else: self.agg_fun = self.aggregate

    def _build_inputs(self):
        self.user_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='user_indices')
        self.item_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='item_indices')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')

        self.memories_h = []
        self.memories_r = []
        self.memories_t = []

        for hop in range(max(1,self.p_hop)):
            self.memories_h.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_h_" + str(hop)))
            self.memories_r.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_r_" + str(hop)))
            self.memories_t.append(
                tf.placeholder(dtype=tf.int32, shape=[None, self.n_memory], name="memories_t_" + str(hop)))

    def save_pretrain_emb_fuc(self,sess, saver):
        saver.save(sess, f"{self.args.path.emb}_sw_para_{self.save_model_name}" + '_parameter')

    def _build_model(self, n_user, n_entity, n_relation):
        self.n_entity = n_entity

        with tf.variable_scope("user_emb_matrix_STWS"):
            self.user_emb_matrix = tf.get_variable(
                shape=[n_user, self.dim], initializer=MVIN.get_initializer(), name='user_emb_matrix_STWS')

        with tf.variable_scope("entity_emb_matrix_STWS"):
            self.entity_emb_matrix = tf.get_variable(
                shape=[n_entity, self.dim], initializer=MVIN.get_initializer(), name='entity_emb_matrix_STWS')

        with tf.variable_scope("relation_emb_matrix_STWS"):
            self.relation_emb_matrix = tf.get_variable(
                shape=[n_relation,self.dim], initializer=MVIN.get_initializer(), name='relation_emb_matrix_STWS')

        with tf.variable_scope("relation_emb_KGE_matrix_STWS"):
            self.relation_emb_KGE_matrix = tf.get_variable(
                shape=[n_relation,self.dim, self.dim], initializer=MVIN.get_initializer(), name='relation_emb_KGE_matrix_STWS')

        self.enti_transfer_matrix_list = []
        self.enti_transfer_bias_list = []

        for n in range(self.n_mix_hop):
            with tf.variable_scope("enti_mlp_matrix"+str(n)):
                self.enti_transfer_matrix = tf.get_variable(
                    shape=[self.dim * (self.h_hop+1), self.dim], initializer=MVIN.get_initializer(), name='transfer_matrix'+str(n))
                self.enti_transfer_bias = tf.get_variable(
                    shape=[self.dim], initializer=MVIN.get_initializer(), name='transfer_bias'+str(n))
                self.enti_transfer_matrix_list.append(self.enti_transfer_matrix)
                self.enti_transfer_bias_list.append(self.enti_transfer_bias)

        with tf.variable_scope("user_mlp_matrix"):
            if self.args.PS_O_ft == True: user_mlp_shape = self.p_hop+1
            else: user_mlp_shape = self.p_hop
            self.user_mlp_matrix = tf.get_variable(
                shape=[self.dim * (user_mlp_shape), self.dim], initializer=MVIN.get_initializer(), name='user_mlp_matrix')
            self.user_mlp_bias = tf.get_variable(shape=[self.dim], initializer=MVIN.get_initializer()
                                                , name='user_mlp_bias')
        self.transfer_matrix_list = []
        self.transfer_matrix_bias = []
        for n in range(self.n_mix_hop*self.h_hop+1):
            with tf.variable_scope("transfer_agg_matrix"+str(n)):
                self.transform_matrix = tf.get_variable(name='transfer_agg_matrix'+str(n), shape=[self.dim, self.dim], dtype=tf.float32,
                                                initializer=MVIN.get_initializer())
                self.transform_bias = tf.get_variable(name='transfer_agg_bias'+str(n), shape=[self.dim], dtype=tf.float32,
                                                initializer=MVIN.get_initializer())
                self.transfer_matrix_bias.append(self.transform_bias)
                self.transfer_matrix_list.append(self.transform_matrix)

        with tf.variable_scope("h_emb_item_mlp_matrix"):
            self.h_emb_item_mlp_matrix = tf.get_variable(
                shape=[self.dim * 2, 1], initializer=MVIN.get_initializer(), name='h_emb_item_mlp_matrix')
            self.h_emb_item_mlp_bias = tf.get_variable(shape=[1], initializer=MVIN.get_initializer()
                                                , name='h_emb_item_mlp_bias')


        self.h_emb_list = []
        self.r_emb_list = []
        self.t_emb_list = []
        for i in range(max(1,self.p_hop)):
            # [batch size, n_memory, dim]
            self.h_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_h[i]))
            # [batch size, n_memory, dim, dim]
            self.r_emb_list.append(tf.nn.embedding_lookup(self.relation_emb_KGE_matrix, self.memories_r[i]))
            # [batch size, n_memory, dim]
            self.t_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_t[i]))

        # [batch_size, dim]
        entities, relations = self.get_neighbors(self.item_indices)

        if self.args.PS_only == True:
            user_o, transfer_o = self._key_addressing()
            item_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.item_indices)

        elif self.args.HO_only == True:
            user_o = tf.nn.embedding_lookup(self.user_emb_matrix, self.user_indices)
            if self.args.User_orient_kg_eh == True:  _, transfer_o = self._key_addressing()
            else: transfer_o = [user_o]
            item_embeddings, self.aggregators = self.agg_fun(entities, relations, transfer_o)

        else:
            print('MVIN PS and HO')
            user_o, transfer_o = self._key_addressing()
            if self.args.User_orient_kg_eh == False: transfer_o = [tf.nn.embedding_lookup(self.user_emb_matrix, self.user_indices)]
            item_embeddings, self.aggregators = self.agg_fun(entities, relations, transfer_o)

        self.scores = tf.reduce_sum(user_o * item_embeddings, axis=1)
        self.scores_normalized = tf.sigmoid(self.scores)

    def _key_addressing(self):
        def soft_attention_h_set():
            user_embedding_key = tf.nn.embedding_lookup(self.user_emb_matrix, self.user_indices)
            # [batch_size, 1, dim]
            item = tf.expand_dims(user_embedding_key, axis=1)
            # print('item = ', item.shape)
            # [batch_size, n_memory, dim]
            item = tf.tile(item, [1, self.h_emb_list[0].shape[1], 1])
            # print('item = ', item.shape)

            h_emb_item = [self.h_emb_list[0],item]
            # [batch_size, n_memory, 2 * dim]

            h_emb_item = tf.concat(h_emb_item, 2)
            # print('h_emb_item = ', h_emb_item.shape)
            # [batch_size, n_memory, 1]

            # [-1 , dim * 2]
            h_emb_item = tf.reshape(h_emb_item,[-1,self.dim * 2])
            # print('h_emb_item = ', h_emb_item.shape)
            # [-1]
            probs = tf.squeeze(tf.matmul(h_emb_item, self.h_emb_item_mlp_matrix), axis=-1) + self.h_emb_item_mlp_bias
            # print('probs = ', probs.shape)

            # [batch_size, n_memory]
            probs = tf.reshape(probs,[-1,self.h_emb_list[0].shape[1]])
            # print('probs = ', probs.shape)

            probs_normalized = tf.nn.softmax(probs)
            # [batch_size, n_memory,1]

            probs_expanded = tf.expand_dims(probs_normalized, axis=2)

            # [batch_size, 1, dim]
            user_h_set = tf.reduce_sum(self.h_emb_list[0] * probs_expanded, axis=1)

            return user_h_set

        item_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.item_indices)


        o_list = []

        if self.args.PS_O_ft == True:
            user_h_set = soft_attention_h_set()
            o_list.append(user_h_set)

        transfer_o = []

        for hop in range(self.p_hop):
            h_expanded = tf.expand_dims(self.h_emb_list[hop], axis=3)

            # [batch_size, n_memory, dim]
            Rh = tf.squeeze(tf.matmul(self.r_emb_list[hop], h_expanded), axis=3)

            # [batch_size, n_memory, dim]
            v = tf.expand_dims(item_embeddings, axis=2)

            # [batch_size, n_memory]
            probs = tf.squeeze(tf.matmul(Rh, v), axis=2)

            # [batch_size, n_memory]
            probs_normalized = tf.nn.softmax(probs) 

            # [batch_size, n_memory, 1]
            probs_expanded = tf.expand_dims(probs_normalized, axis=2)

            # [batch_size, dim]
            o = tf.reduce_sum(self.t_emb_list[hop] * probs_expanded, axis=1)
            o_list.append(o)

        o_list = tf.concat(o_list, -1)
        if self.args.PS_O_ft == True:
            user_o = tf.matmul(tf.reshape(o_list,[-1,self.dim * (self.p_hop+1)]), self.user_mlp_matrix) + self.user_mlp_bias
        else:
            user_o = tf.matmul(tf.reshape(o_list,[-1,self.dim * (self.p_hop)]), self.user_mlp_matrix) + self.user_mlp_bias

        transfer_o.append(user_o)

        return user_o, transfer_o


    def get_neighbors(self, seeds):

        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]

        relations = []
        n = self.n_neighbor
        for i in range(self.n_mix_hop*self.h_hop):
            neighbor_entities = tf.reshape(tf.gather(self.adj_entity, entities[i]), [self.batch_size, n])
            neighbor_relations = tf.reshape(tf.gather(self.adj_relation, entities[i]), [self.batch_size, n])
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
            n *= self.n_neighbor
        return entities, relations


    def aggregate_delta_whole(self, entities, relations, transfer_o):
        # print('aggregate_delta_whole ===')
        user_query = transfer_o[0]
        print('MVIN aggregate_delta_whole')
        aggregators = []  # store all aggregators
        mix_hop_res = []


        entity_vectors = [tf.nn.embedding_lookup(self.entity_emb_matrix, i) for i in entities]
        relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations]

        if self.args.User_orient == True:
            print('user_orient')
            for index in range(len(transfer_o)):
                transfer_o[index] = tf.expand_dims(transfer_o[index], axis=1)
            for index in range(len(transfer_o)):
                for e_i in range(len(entity_vectors)):
                    # [b,1,dim]
                    n_entities = entity_vectors[e_i] + transfer_o[index]
                    # [-1,dim]
                    n_entities = tf.matmul(tf.reshape(n_entities, [-1,self.dim]), self.transfer_matrix_list[e_i]) + self.transfer_matrix_bias[e_i]
                    # [b,n,dim]
                    entity_vectors[e_i] = tf.reshape(n_entities, [self.batch_size, entity_vectors[e_i].shape[1],self.dim])
                    # [b,?*n,dim]
                    transfer_o[index] = tf.tile(transfer_o[index],[1,self.n_neighbor,1])


        for n in range(self.n_mix_hop):
            mix_hop_tmp = []
            mix_hop_tmp.append(entity_vectors)
            for i in range(self.h_hop):
                aggregator = self.aggregator_class(self.save_model_name,self.batch_size, self.dim, name = str(i)+'_'+str(n), User_orient_rela = self.User_orient_rela)
                aggregators.append(aggregator)
                entity_vectors_next_iter = []

                if i == 0: self.importance_list = []
                for hop in range(self.h_hop*self.n_mix_hop-(self.h_hop*n+i)):
                    shape = [self.batch_size, entity_vectors[hop].shape[1], self.n_neighbor, self.dim]
                    shape_r = [self.batch_size, entity_vectors[hop].shape[1], self.n_neighbor, self.dim]
                    print('relation_vectors[hop = ', relation_vectors[hop].shape)
                    vector, probs_normalized = aggregator(self_vectors=entity_vectors[hop],
                                        neighbor_vectors=tf.reshape(entity_vectors[hop + 1], shape),
                                        neighbor_relations=tf.reshape(relation_vectors[hop], shape_r),
                                        user_embeddings=user_query,
                                        masks=None)
                    if i == 0: self.importance_list.append(probs_normalized)
                    entity_vectors_next_iter.append(vector)
                entity_vectors = entity_vectors_next_iter
                mix_hop_tmp.append(entity_vectors)

            entity_vectors = []
            for mip_hop in zip(*mix_hop_tmp):
                mip_hop = tf.concat(mip_hop, -1)
                mip_hop = tf.matmul(tf.reshape(mip_hop,[-1,self.dim * (self.h_hop+1)]), self.enti_transfer_matrix_list[n]) + self.enti_transfer_bias_list[n]
                mip_hop = tf.reshape(mip_hop,[self.batch_size,-1,self.dim]) 
                entity_vectors.append(mip_hop)
                if len(entity_vectors) == (self.n_mix_hop-(n+1))*self.h_hop+1:  break

        mix_hop_res = tf.reshape(entity_vectors[0], [self.batch_size, self.dim])

        self.importance_list_0 = self.importance_list[0]
        if len(self.importance_list) > 1:
            self.importance_list_1 = self.importance_list[1]
        else:
            self.importance_list_1 = 0
        return mix_hop_res, aggregators


    def aggregate(self, entities, relations, transfer_o):

        user_query = transfer_o[0]

        print('aggregate agg method')
        aggregators = []  # store all aggregators
        entity_vectors = [tf.nn.embedding_lookup(self.entity_emb_matrix, i) for i in entities]
        relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations]

        if self.args.User_orient == True:
            for index in range(len(transfer_o)):
                # [b,1,dim]
                transfer_o[index] = tf.expand_dims(transfer_o[index], axis=1)
                # print('transfer_o[index] = ', transfer_o[index].shape)
            for index in range(len(transfer_o)):
                for e_i in range(len(entity_vectors)):
                    # print('entity_vectors[e_i] = ', entity_vectors[e_i].shape)
                    # [b,1,dim]
                    n_entities = entity_vectors[e_i] + transfer_o[index]
                    # print('n_entities = ', n_entities.shape)
                    # [-1,dim]
                    n_entities = tf.matmul(tf.reshape(n_entities, [-1,self.dim]), self.transfer_matrix_list[e_i]) + self.transfer_matrix_bias[e_i]
                    # n_entities = self.act(n_entities)
                    # print('n_entities = ', n_entities.shape)
                    # [b,n,dim]
                    entity_vectors[e_i] = tf.reshape(n_entities, [self.batch_size, entity_vectors[e_i].shape[1],self.dim])
                    # print('entity_vectors[e_i] = ', entity_vectors[e_i].shape)
                    # [b,?*n,dim]
                    transfer_o[index] = tf.tile(transfer_o[index],[1,self.n_neighbor,1])

        for i in range(self.h_hop):

            aggregator = self.aggregator_class(self.save_model_name,self.batch_size, self.dim, name = i)
            aggregators.append(aggregator)

            entity_vectors_next_iter = []
            for hop in range(self.h_hop - i):
                shape = [self.batch_size, entity_vectors[hop].shape[1], self.n_neighbor, self.dim]
                shape_r = [self.batch_size, entity_vectors[hop].shape[1], self.n_neighbor, self.dim]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vectors=tf.reshape(entity_vectors[hop + 1], shape),
                                    neighbor_relations=tf.reshape(relation_vectors[hop], shape_r),
                                    user_embeddings=user_query,
                                    masks=None)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        res = tf.reshape(entity_vectors[0], [self.batch_size, self.dim])

        return res, aggregators

    def _build_train(self):
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.scores))

        self.l2_loss = 0
        for hop in range(self.p_hop):
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.h_emb_list[hop] * self.h_emb_list[hop]))
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.t_emb_list[hop] * self.t_emb_list[hop]))
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.r_emb_list[hop] * self.r_emb_list[hop]))

        self.l2_loss += tf.nn.l2_loss(self.relation_emb_matrix) 

        self.l2_agg_loss = 0
        
        self.l2_agg_loss += tf.nn.l2_loss(self.user_emb_matrix)
        if self.args.PS_only != True:
            for aggregator in self.aggregators:
                self.l2_agg_loss += tf.nn.l2_loss(aggregator.weights)
                self.l2_agg_loss += tf.nn.l2_loss(aggregator.urh_weights)
                # print('self.l2_agg_loss += tf.nn.l2_loss(aggregator.weights)')
                # print('self.l2_agg_loss += tf.nn.l2_loss(aggregator.urh_weights)')

        for n in range(self.n_mix_hop):
            self.l2_agg_loss += tf.nn.l2_loss(self.enti_transfer_matrix_list[n]) + tf.nn.l2_loss(self.enti_transfer_bias_list[n])
        
        if self.p_hop > 0:
            self.l2_loss += tf.nn.l2_loss(self.user_mlp_matrix) + tf.nn.l2_loss(self.user_mlp_bias)
            self.l2_loss += tf.nn.l2_loss(self.transform_matrix) + tf.nn.l2_loss(self.transform_bias)

            for n in range(self.h_hop+1):
                self.l2_loss += tf.nn.l2_loss(self.transfer_matrix_list[n]) + tf.nn.l2_loss(self.transfer_matrix_bias[n])

        self.l2_loss += tf.nn.l2_loss(self.h_emb_item_mlp_matrix) +  tf.nn.l2_loss(self.h_emb_item_mlp_bias)

        self.loss = self.base_loss + self.l2_weight * self.l2_loss + self.l2_agg_weight * self.l2_agg_loss 

        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        acc = np.mean(np.equal(scores, labels))
        return auc, acc, f1

    def eval_case_study(self, sess, feed_dict):
        user_indices, labels, item_indices, entities_data, relations_data, importance_list_0, importance_list_1 = sess.run([self.user_indices, self.labels, self.item_indices,  
            self.entities_data, self.relations_data, self.importance_list_0, self.importance_list_1], feed_dict)

        return user_indices, labels, item_indices, entities_data, relations_data, importance_list_0, importance_list_1

    def get_scores(self, sess, feed_dict):
        return sess.run([self.item_indices, self.scores_normalized], feed_dict)
