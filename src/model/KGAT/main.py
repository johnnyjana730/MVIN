'''
Created on Dec 18, 2018
Tensorflow Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import tensorflow as tf
from utility.helper import *
from utility.batch_test import *
from time import time

from BPRMF import BPRMF
from CKE import CKE
from CFKG import CFKG
from NFM import NFM
from KGAT import KGAT
from metrics import ndcg_at_k, map_at_k, recall_at_k, hit_ratio_at_k, mrr_at_k, precision_at_k

import os
import sys

def topk_eval(sess, args, data_generator, model, user_list, train_record, eval_record, test_record, 
    item_set, k_list, batch_size, mode = 'test'):
    
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}
    MAP_list = {k: [] for k in k_list}
    hit_ratio_list = {k: [] for k in k_list}
    ndcg_list = {k: [] for k in k_list}

    
    for user in user_list:
        if mode == 'eval': ref_user = eval_record
        else: ref_user = test_record
        if user in ref_user:
            test_item_list = list(item_set - train_record[user])
            item_score_map = dict()
            start = 0

            while start + batch_size <= len(test_item_list):

                user_list_tmp = [user] * batch_size
                user_list_tmp = np.array(user_list_tmp)
                
                item_list = test_item_list[start:start + batch_size]
                item_list = np.array(item_list)

                labels_list = [1] * batch_size
                labels_list = np.array(labels_list)

                data = np.concatenate((np.expand_dims(user_list_tmp, axis=1), np.expand_dims(item_list, axis=1),
                                    np.expand_dims(labels_list, axis=1)), axis=1)

                batch_data = data_generator.generate_rating_batch(data, 0, args.batch_size)
                feed_dict = data_generator.generate_feed_rating_dict(model, batch_data)
                items, scores = model.get_scores(sess,feed_dict)

                for item, score in zip(items, scores):
                    item_score_map[item] = score
                start += batch_size

            # padding the last incomplete minibatch if exists
            if start < len(test_item_list):

                user_list_tmp = [user] * batch_size
                user_list_tmp = np.array(user_list_tmp)
                item_list = test_item_list[start:] + [test_item_list[-1]] * (batch_size - len(test_item_list) + start)
                item_list = np.array(item_list)
                labels_list = [1] * batch_size
                labels_list = np.array(labels_list)
                data = np.concatenate((np.expand_dims(user_list_tmp, axis=1), np.expand_dims(item_list, axis=1),
                                    np.expand_dims(labels_list, axis=1)), axis=1)

                batch_data = data_generator.generate_rating_batch(data, 0, args.batch_size)
                feed_dict = data_generator.generate_feed_rating_dict(model, batch_data)
                items, scores = model.get_scores(sess,feed_dict)

                for item, score in zip(items, scores):
                    item_score_map[item] = score

            item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
            item_sorted = [i[0] for i in item_score_pair_sorted]


            for k in k_list:
                precision_list[k].append(precision_at_k(item_sorted,ref_user[user],k))
                recall_list[k].append(recall_at_k(item_sorted,ref_user[user],k))

            # ndcg
            r_hit = []
            for i in item_sorted[:k]:
                if i in ref_user[user]:
                    r_hit.append(1)
                else:
                    r_hit.append(0)
            for k in k_list:
                ndcg_list[k].append(ndcg_at_k(r_hit,k))

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]
    ndcg = [np.mean(ndcg_list[k]) for k in k_list]

    return precision, recall, ndcg, None, None


def ctr_eval(args, sess, model, data, batch_size):
    start = 0
    auc_list = []
    acc_list = []
    f1_list = []

    while start + args.batch_size <= data.shape[0]:
        batch_data = data_generator.generate_rating_batch(data, start, start + args.batch_size)
        feed_dict = data_generator.generate_feed_rating_dict(model, batch_data)
        auc, acc,  f1 = model.eval(sess,feed_dict)
        auc_list.append(auc)
        acc_list.append(acc)
        f1_list.append(f1)
        start += batch_size

    return auc_list, acc_list, f1_list, float(np.mean(auc_list)), float(np.mean(acc_list)), float(np.mean(f1_list))


class Eval_score_info:
    def __init__(self):
        self.train_auc_acc_f1 = [0 for _ in range(3)]
        self.eval_auc_acc_f1 = [0 for _ in range(3)]
        self.test_auc_acc_f1 = [0 for _ in range(3)]

        self.train_ndcg_recall_pecision = [[0 for i in range(7)] for _ in range(3)]
        self.eval_ndcg_recall_pecision = [[0 for i in range(7)] for _ in range(3)]
        self.test_ndcg_recall_pecision = [[0 for i in range(7)] for _ in range(3)]

        self.eval_precision = 0
        self.eval_recall = 0
        self.eval_ndcg = 0
        self.test_precision = 0
        self.test_recall = 0
        self.test_ndcg = 0
    def eval_st_score(self):
        return self.eval_auc_acc_f1[0]
    def eval_st_top_k_score(self):
        return self.eval_ndcg_recall_pecision[1][2]


def load_pretrained_data(args):
    pre_model = 'mf'
    if args.pretrain == -2:
        pre_model = 'kgat'
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, pre_model)
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained bprmf model parameters.')
    except Exception:
        pretrain_data = None
    return pretrain_data


if __name__ == '__main__':
    args = parse_args()

    """
    *********************************************************
    Load Data from data_generator function.
    """
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['n_relations'] = data_generator.n_relations
    config['n_entities'] = data_generator.n_entities

    if args.model_type in ['kgat', 'cfkg']:
        "Load the laplacian matrix."
        config['A_in'] = sum(data_generator.lap_list)

        "Load the KG triplets."
        config['all_h_list'] = data_generator.all_h_list
        config['all_r_list'] = data_generator.all_r_list
        config['all_t_list'] = data_generator.all_t_list
        config['all_v_list'] = data_generator.all_v_list

    t0 = time()

    """
    *********************************************************
    Use the pretrained data to initialize the embeddings.
    """
    if args.pretrain in [-1, -2]:
        pretrain_data = load_pretrained_data(args)
    else:
        pretrain_data = None

    """
    *********************************************************
    Select one of the models.
    """
    if args.model_type == 'bprmf':
        print('args.model_type = ', args.model_type, 'bprmf')
        model = BPRMF(data_config=config, pretrain_data=pretrain_data, args=args)

    elif args.model_type == 'cke':
        print('args.model_type = ', args.model_type, 'cke')
        model = CKE(data_config=config, pretrain_data=pretrain_data, args=args)

    elif args.model_type in ['cfkg']:
        print('args.model_type = ', args.model_type, 'cfkg')
        model = CFKG(data_config=config, pretrain_data=pretrain_data, args=args)

    elif args.model_type in ['nfm', 'fm']:
        print('args.model_type = ', args.model_type, 'nfm, fm')
        model = NFM(data_config=config, pretrain_data=pretrain_data, args=args)

    elif args.model_type in ['kgat']:
        print('args.model_type = ', args.model_type, 'kgat')
        model = KGAT(data_config=config, pretrain_data=pretrain_data, args=args)

    saver = tf.train.Saver()

    """
    *********************************************************
    Save the model parameters.
    """
    if args.save_flag == 1:
        if args.model_type in ['bprmf', 'cke', 'fm', 'cfkg']:
            weights_save_path = '%sweights/%s/%s/l%s_r%s' % (args.save_path, args.dataset, model.model_type,
                                                             str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))

        elif args.model_type in ['ncf', 'nfm', 'kgat']:
            layer = '-'.join([str(l) for l in eval(args.layer_size)])
            weights_save_path = '%sweights/%s/%s/%s/l%s_r%s' % (
                args.save_path, args.dataset, model.model_type, layer, str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))

        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    """
    *********************************************************
    Reload the model parameters to fine tune.
    """
    if args.pretrain == 1:
        if args.model_type in ['bprmf', 'cke', 'fm', 'cfkg']:
            pretrain_path = '%sweights/%s/%s/l%s_r%s' % (args.save_path, args.dataset, model.model_type, str(args.lr),
                                                             '-'.join([str(r) for r in eval(args.regs)]))

        elif args.model_type in ['ncf', 'nfm', 'kgat']:
            layer = '-'.join([str(l) for l in eval(args.layer_size)])
            pretrain_path = '%sweights/%s/%s/%s/l%s_r%s' % (
                args.save_path, args.dataset, model.model_type, layer, str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('load the pretrained model parameters from: ', pretrain_path)

            # *********************************************************
            # get the performance from the model to fine tune.
            if args.report != 1:
                users_to_test = list(data_generator.test_user_dict.keys())

                ret = test(sess, model, users_to_test, drop_flag=False, batch_test_flag=batch_test_flag)
                cur_best_pre_0 = ret['recall'][0]

                pretrain_ret = 'pretrained model recall=[%.5f, %.5f], precision=[%.5f, %.5f], hit=[%.5f, %.5f],' \
                               'ndcg=[%.5f, %.5f], auc=[%.5f]' % \
                               (ret['recall'][0], ret['recall'][-1],
                                ret['precision'][0], ret['precision'][-1],
                                ret['hit_ratio'][0], ret['hit_ratio'][-1],
                                ret['ndcg'][0], ret['ndcg'][-1], ret['auc'])
                print(pretrain_ret)

                # *********************************************************
                # save the pretrained model parameters of mf (i.e., only user & item embeddings) for pretraining other models.
                if args.save_flag == -1:
                    user_embed, item_embed = sess.run(
                        [model.weights['user_embedding'], model.weights['item_embedding']],
                        feed_dict={})
                    # temp_save_path = '%spretrain/%s/%s/%s_%s.npz' % (args.proj_path, args.dataset, args.model_type, str(args.lr),
                    #                                                  '-'.join([str(r) for r in eval(args.regs)]))
                    temp_save_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, model.model_type)
                    ensureDir(temp_save_path)
                    np.savez(temp_save_path, user_embed=user_embed, item_embed=item_embed)
                    print('save the weights of fm in path: ', temp_save_path)
                    exit()

                # *********************************************************
                # save the pretrained model parameters of kgat (i.e., user & item & kg embeddings) for pretraining other models.
                if args.save_flag == -2:
                    user_embed, entity_embed, relation_embed = sess.run(
                        [model.weights['user_embed'], model.weights['entity_embed'], model.weights['relation_embed']],
                        feed_dict={})

                    temp_save_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, args.model_type)
                    ensureDir(temp_save_path)
                    np.savez(temp_save_path, user_embed=user_embed, entity_embed=entity_embed, relation_embed=relation_embed)
                    print('save the weights of kgat in path: ', temp_save_path)
                    exit()

        else:
            sess.run(tf.global_variables_initializer())
            cur_best_pre_0 = 0.
            print('without pretraining.')
    else:
        sess.run(tf.global_variables_initializer())
        cur_best_pre_0 = 0.
        print('without pretraining.')

    """
    *********************************************************
    Get the final performance w.r.t. different sparsity levels.
    """
    if args.report == 1:
        assert args.test_flag == 'full'
        users_to_test_list, split_state = data_generator.get_sparsity_split()

        users_to_test_list.append(list(data_generator.test_user_dict.keys()))
        split_state.append('all')

        save_path = '%sreport/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
        ensureDir(save_path)
        f = open(save_path, 'w')
        f.write('embed_size=%d, lr=%.6f, regs=%s, loss_type=%s, \n' % (args.embed_size, args.lr, args.regs,
                                                                       args.loss_type))

        for i, users_to_test in enumerate(users_to_test_list):
            ret = test(sess, model, users_to_test, drop_flag=False, batch_test_flag=batch_test_flag)

            final_perf = "recall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                         ('\t'.join(['%.5f' % r for r in ret['recall']]),
                          '\t'.join(['%.5f' % r for r in ret['precision']]),
                          '\t'.join(['%.5f' % r for r in ret['hit_ratio']]),
                          '\t'.join(['%.5f' % r for r in ret['ndcg']]))
            print(final_perf)

            f.write('\t%s\n\t%s\n' % (split_state[i], final_perf))
        f.close()
        exit()

    """
    *********************************************************
    Train.
    """

    loss_loger, train_auc_acc_f1, eval_auc_acc_f1, test_auc_acc_f1  = [], [[] for _ in range(3)], [[] for _ in range(3)], [[] for _ in range(3)]
    best_eval_auc_acc_f1, best_test_auc_acc_f1 = [0 for _ in range(3)], [0 for _ in range(3)]
    
    max_eval_recall = [0 for i in range(7)]
    max_test_recall = [0 for i in range(7)]

    max_eval_ndcg = [0 for i in range(7)]
    max_test_ndcg = [0 for i in range(7)]

    max_eval_precision = [0 for i in range(7)]
    max_test_precision = [0 for i in range(7)]

    stopping_step = 0
    should_stop = False

    eval_score_info = Eval_score_info()

    for epoch in range(args.epoch):
        t1 = time()
        loss, base_loss, kge_loss, reg_loss = 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1


        np.random.shuffle(data_generator.train_ratings)
        start = 0

        """
        *********************************************************
        Alternative Training for KGAT:
        ... phase 1: to train the recommender.
        """

        while start + args.batch_size <= data_generator.train_ratings.shape[0]:
            btime= time()

            batch_data = data_generator.generate_rating_batch(data_generator.train_ratings, start, start + args.batch_size)
            feed_dict = data_generator.generate_feed_rating_dict(model, batch_data)
            _, batch_loss, batch_base_loss, batch_kge_loss, batch_reg_loss = model.train(sess, feed_dict=feed_dict)

            loss += batch_loss
            base_loss += batch_base_loss
            kge_loss += batch_kge_loss
            reg_loss += batch_reg_loss
            start += args.batch_size
            
        if np.isnan(loss) == True:
            print('ERROR: loss@phase1 is nan.')
            sys.exit()

        """
        *********************************************************
        Alternative Training for KGAT:
        ... phase 2: to train the KGE method & update the attentive Laplacian matrix.
        """
        if args.model_type in ['kgat']:

            n_A_batch = len(data_generator.all_h_list) // args.batch_size_kg + 1

            if args.use_kge is True:
                # using KGE method (knowledge graph embedding).
                for idx in range(n_A_batch):
                    btime = time()

                    A_batch_data = data_generator.generate_train_A_batch()
                    feed_dict = data_generator.generate_train_A_feed_dict(model, A_batch_data)

                    _, batch_loss, batch_kge_loss, batch_reg_loss = model.train_A(sess, feed_dict=feed_dict)

                    loss += batch_loss
                    kge_loss += batch_kge_loss
                    reg_loss += batch_reg_loss

            if args.use_att is True:
                # updating attentive laplacian matrix.
                model.update_attentive_A(sess)

        if np.isnan(loss) == True:
            print('ERROR: loss@phase2 is nan.')
            sys.exit()

        show_step = 10
        if (epoch + 1) % show_step != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
                    epoch, time() - t1, loss, base_loss, kge_loss, reg_loss)
                print(perf_str, end = '\r')
            continue

        t2 = time()

        _, _, _, auc, acc, f1 = ctr_eval(args, sess, model, data_generator.train_ratings, args.batch_size)
        eval_score_info.train_auc_acc_f1 = auc, acc, f1
        _, _, _, auc, acc, f1 = ctr_eval(args, sess, model, data_generator.eval_ratings, args.batch_size)
        eval_score_info.eval_auc_acc_f1 = auc, acc, f1
        test_auc_list, test_acc_list, test_f1_list, auc, acc, f1 = ctr_eval(args, sess, model, data_generator.test_ratings, args.batch_size)
        eval_score_info.test_auc_acc_f1 = auc, acc, f1

        satistic_list = [test_auc_list, test_acc_list, test_f1_list]

        if True:
            precision, recall, ndcg, MAP, hit_ratio = topk_eval(
                sess, args, data_generator, model, data_generator.user_list, data_generator.train_record, data_generator.eval_record, 
                data_generator.test_record, data_generator.item_set_most_pop, data_generator.k_list, args.batch_size, mode = 'eval')

            n_precision_eval = [round(i, 6) for i in precision]
            n_recall_eval = [round(i, 6) for i in recall]
            n_ndcg_eval = [round(i, 6) for i in ndcg]

            precision, recall, ndcg, MAP, hit_ratio = topk_eval(
                sess, args, data_generator, model, data_generator.user_list, data_generator.train_record, data_generator.eval_record, 
                data_generator.test_record, data_generator.item_set_most_pop, data_generator.k_list, args.batch_size, mode = 'test')

            n_precision_test = [round(i, 6) for i in precision]
            n_recall_test = [round(i, 6) for i in recall]
            n_ndcg_test = [round(i, 6) for i in ndcg]


            eval_score_info.eval_ndcg_recall_pecision = [n_ndcg_eval, n_recall_eval, n_precision_eval]
            eval_score_info.test_ndcg_recall_pecision = [n_ndcg_test, n_recall_test, n_precision_test]

        """
        *********************************************************
        Performance logging.
        """
        t3 = time()

        loss_loger.append(loss)
        
        for i in range(len(train_auc_acc_f1)):
            train_auc_acc_f1[i].append(eval_score_info.train_auc_acc_f1[i])
            eval_auc_acc_f1[i].append(eval_score_info.eval_auc_acc_f1[i])
            test_auc_acc_f1[i].append(eval_score_info.test_auc_acc_f1[i])

        if args.verbose > 0:
            train_auc, train_acc, train_f1 = eval_score_info.train_auc_acc_f1[0], eval_score_info.train_auc_acc_f1[1], eval_score_info.train_auc_acc_f1[2]
            eval_auc, eval_acc, eval_f1 = eval_score_info.eval_auc_acc_f1[0], eval_score_info.eval_auc_acc_f1[1], eval_score_info.eval_auc_acc_f1[2]
            test_auc, test_acc, test_f1 =  eval_score_info.test_auc_acc_f1[0], eval_score_info.test_auc_acc_f1[1], eval_score_info.test_auc_acc_f1[2]

            print('epoch %d  train auc: %.6f acc: %.6f f1: %.6f eval auc: %.6f acc: %.6f f1: %.6f test auc: %.6f acc: %.6f f1: %.6f'
                      % (epoch, train_auc, train_acc, train_f1, eval_auc, eval_acc, eval_f1, test_auc, test_acc, test_f1))

            if True:
                train_ndcg, train_recall, train_precision = eval_score_info.train_ndcg_recall_pecision[0], eval_score_info.train_ndcg_recall_pecision[1], eval_score_info.train_ndcg_recall_pecision[2]
                eval_ndcg, eval_recall, eval_precision = eval_score_info.eval_ndcg_recall_pecision[0], eval_score_info.eval_ndcg_recall_pecision[1], eval_score_info.eval_ndcg_recall_pecision[2]
                test_ndcg, test_recall, test_precision =  eval_score_info.test_ndcg_recall_pecision[0], eval_score_info.test_ndcg_recall_pecision[1], eval_score_info.test_ndcg_recall_pecision[2]

                print(f"epoch = {epoch}, eval ndcg = {eval_ndcg} ")
                print(f"{epoch}, test ndcg = {test_ndcg} \n")
                print(f"epoch = {epoch}, eval recall = {eval_recall} ")
                print(f"{epoch}, test recall = {test_recall} \n")



        if True:
            cur_best_pre_0, stopping_step, should_stop = early_stopping(eval_score_info.eval_st_score(), cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=2)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.

        if eval_score_info.eval_st_score() == cur_best_pre_0 and args.save_flag == 1:
            best_eval_auc_acc_f1 = eval_score_info.eval_auc_acc_f1[0], eval_score_info.eval_auc_acc_f1[1], eval_score_info.eval_auc_acc_f1[2]
            best_test_auc_acc_f1 = eval_score_info.test_auc_acc_f1[0], eval_score_info.test_auc_acc_f1[1], eval_score_info.test_auc_acc_f1[2]
            
            max_eval_ndcg, max_eval_recall, max_eval_precision = eval_score_info.eval_ndcg_recall_pecision[0], eval_score_info.eval_ndcg_recall_pecision[1], eval_score_info.eval_ndcg_recall_pecision[2]
            max_test_ndcg, max_test_recall, max_test_precision =  eval_score_info.test_ndcg_recall_pecision[0], eval_score_info.test_ndcg_recall_pecision[1], eval_score_info.test_ndcg_recall_pecision[2]

            save_path = '%sconfig/%s/%s_batch_%s_emb_%s_lr_%s/' % (args.save_path, args.dataset, model.model_type, args.batch_size, args.embed_size, args.lr)

            try: os.makedirs(save_path)
            except: pass

            with open(save_path + 'config_auc.txt', 'w') as output:
                for row in satistic_list[0]:
                    output.write(str(row) + '\n')
            with open(save_path + 'config_acc.txt', 'w') as output:
                for row in satistic_list[1]:
                    output.write(str(row) + '\n')
            with open(save_path + 'config_f1.txt', 'w') as output:
                for row in satistic_list[2]:
                    output.write(str(row) + '\n')

            save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
            print('save the weights in path: ', weights_save_path)

    final_perf = 'epoch %d  eval auc: %.6f acc: %.6f f1: %.6f test auc: %.6f acc: %.6f f1: %.6f' \
              % (epoch, best_eval_auc_acc_f1[0], best_eval_auc_acc_f1[1], best_eval_auc_acc_f1[2], best_test_auc_acc_f1[0], best_test_auc_acc_f1[1], best_test_auc_acc_f1[2])
    print(final_perf)


    final_top_k_perf = f"epoch {epoch}, eval ndcg: {max_eval_ndcg} \nrecall: {max_eval_recall} pre: {max_eval_precision} \ntest ndcg: {max_test_ndcg} recall: {max_test_recall} \n pre: {max_test_precision}"

    save_path = '%soutput/%s/%s_batch_%s.result_news_top_k_auc_b_500_i_250_u' % (args.proj_path, args.dataset, model.model_type, args.batch_size)
    ensureDir(save_path)
    f = open(save_path, 'a')
    f.write(f"embed_size={args.embed_size}, lr={args.lr}, layer_size={args.layer_size}, node_dropout={args.node_dropout},mess_dropout={args.mess_dropout}, regs={args.regs},")
    f.write(f"batch_size={args.batch_size}, loss_type={args.loss_type},using_all_hops={args.item_update_mode}, adj_type={args.adj_type}, use_att={args.use_att}, use_kge={args.use_kge}\n{final_perf}\n{final_top_k_perf}\n")
    f.write(f"{'*'*100} \n")
    f.close()
