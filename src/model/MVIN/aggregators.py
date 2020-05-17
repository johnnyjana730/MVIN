import tensorflow as tf
from abc import abstractmethod

LAYER_IDS = {}

# tf.set_random_seed(1)

def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]


class Aggregator(object):
    def __init__(self, save_model_name, batch_size, dim, dropout, act, name):

        layer = self.__class__.__name__.lower()
        name = layer + '_'+save_model_name+'_' + str(name)
        # print('name = ',name)
        self.name = name
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim

    def __call__(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks):
        outputs = self._call(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings,masks)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks):
        pass

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        avg = False
        if not avg:
            # [batch_size, 1, 1, dim]
            user_embeddings = tf.reshape(user_embeddings, [neighbor_vectors.shape[0], 1, 1, self.dim])

            # [batch_size, -1, n_neighbor]
            user_relation_scores = tf.reduce_mean(user_embeddings * neighbor_relations, axis=-1)
            user_relation_scores_normalized = tf.nn.softmax(user_relation_scores, dim=-1)

            # [batch_size, -1, n_neighbor, 1] shape_n = [self.batch_size, -1, self.n_neighbor, self.dim]
            user_relation_scores_normalized = tf.expand_dims(user_relation_scores_normalized, axis=-1)

            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(user_relation_scores_normalized * neighbor_vectors, axis=2)
        else:
            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(neighbor_vectors, axis=2)

        return neighbors_aggregated

    def _mix_neighbor_vectors_urv(self, neighbor_vectors, neighbor_relations, user_embeddings):
        avg = False
        if not avg:
            # [batch_size, 1, 1, dim]
            user_embeddings = tf.reshape(user_embeddings, [neighbor_vectors.shape[0], 1, 1, self.dim])

            # [batch_size, -1, n_neighbor]
            user_relation_scores = tf.reduce_mean(user_embeddings * neighbor_relations, axis=-1)
            user_relation_scores_normalized = tf.nn.softmax(user_relation_scores, dim=-1)

            # [batch_size, -1, n_neighbor, 1] shape_n = [self.batch_size, -1, self.n_neighbor, self.dim]
            user_relation_scores_normalized = tf.expand_dims(user_relation_scores_normalized, axis=-1)

            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(user_relation_scores_normalized * neighbor_vectors, axis=2)
        else:
            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(neighbor_vectors, axis=2)

        return neighbors_aggregated        

class SumAggregator_urh_matrix(Aggregator):
    def __init__(self, save_model_name, batch_size, dim, dropout=0., act=tf.nn.relu, name=None, User_orient_rela = True):
        super(SumAggregator_urh_matrix, self).__init__(save_model_name,batch_size, dim, dropout, act, name)

        with tf.variable_scope(self.name+'_wights'):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(seed = 1), name='weights')
        with tf.variable_scope(self.name+'_bias'):
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

        with tf.variable_scope(self.name+'_urh_wights'):
            self.urh_weights = tf.get_variable(
                shape=[3 * self.dim, 1], initializer=tf.contrib.layers.xavier_initializer(seed = 1), name='weights')
        with tf.variable_scope(self.name+'_urh_bias'):
            self.urh_bias = tf.get_variable(shape=[1], initializer=tf.zeros_initializer(), name='bias')

        self.User_orient_rela = User_orient_rela
        self.act = tf.nn.relu

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks):
        # [batch_size, -1, dim]
        if self.User_orient_rela == True:
            print('self.User_orient_rela  = ', self.User_orient_rela, '_mix_neighbor_vectors_urh')
            neighbors_agg, probs_normalized = self._mix_neighbor_vectors_urh(self_vectors, user_embeddings, neighbor_vectors, neighbor_relations)
        else:
            print('self.User_orient_rela  = ', self.User_orient_rela, '_mix_neighbor_vectors_no_ur')
            neighbors_agg = self._mix_neighbor_vectors_no_ur(self_vectors, user_embeddings, neighbor_vectors, neighbor_relations)
            probs_normalized = None
        # [-1, dim]
        output = tf.reshape(self_vectors + neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        # return output
        return self.act(output), probs_normalized

    def _mix_neighbor_vectors_urh(self, self_vectors, user_embeddings, neighbor_vectors, neighbor_relations):

        # [batch_size, 1, 1, dim]
        user_embeddings = tf.reshape(user_embeddings, [self.batch_size, 1, 1, self.dim])
        user_embeddings = tf.tile(user_embeddings, multiples=[1, neighbor_relations.get_shape()[1], neighbor_relations.get_shape()[2], 1])


        # [batch_size, -1, 1, dim]
        self_vectors = tf.expand_dims(self_vectors, axis=2)
        self_vectors = tf.tile(self_vectors, multiples=[1, 1, neighbor_relations.get_shape()[2], 1])

        # [batch_size, -1, -1, dim * 4]
        urh_matrix = [user_embeddings, neighbor_relations, self_vectors]
        urh_matrix = tf.concat(urh_matrix, -1)
        # [-1, 1]
        urh_matrix = tf.matmul(tf.reshape(urh_matrix,[-1, 3 * self.dim]), self.urh_weights)


        probs = tf.reshape(urh_matrix,[neighbor_vectors.get_shape()[0],neighbor_vectors.get_shape()[1],neighbor_vectors.get_shape()[2]])

        # [batch_size, -1, n_memory]
        probs_normalized = tf.nn.softmax(probs)
        # [batch_size,-1, n_memory, 1]
        probs_expanded = tf.expand_dims(probs_normalized, axis= -1)

        # [batch_size, -1, n_memory]
        neighbors_aggregated = tf.reduce_mean(probs_expanded * neighbor_vectors, axis=2)

        return neighbors_aggregated, probs_normalized

    def _mix_neighbor_vectors_no_ur(self, self_vectors, user_embeddings, neighbor_vectors, neighbor_relations):

        neighbors_aggregated = tf.reduce_mean(neighbor_vectors, axis=2)

        return neighbors_aggregated

