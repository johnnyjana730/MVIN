import os
import time
import numpy as np
import csv

def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

class Early_stop_info:
    def __init__(self, args):
        self.best_score = -float('inf')
        self.training_start_time = time.time()
        self.earlystop_counter = 0
        self.early_stop = args.early_stop
        self.tolerance = args.tolerance

    def update_score(self, epoch, score):
        if epoch + 1 > self.tolerance:
            if score >= self.best_score:
                self.best_score = score
                self.earlystop_counter = 0
            else: self.earlystop_counter += 1

        return self.earlystop_counter >= self.early_stop


class Train_info_record:
    def __init__(self, args, tags=[], eval_methods=['auc', 'acc', 'f1'], topk_methods=['p', 'r', 'ndcg']):
        cur_tim = time.strftime("%Y%m%d")

        self.folder_path = args.path.output + f"{str(cur_tim)}_{args.log_name}.log"
        self.folder_path_best = args.path.output + f"{str(cur_tim)}_{args.log_name}_best.log"

        self.topk_eval = args.topk_eval
        self.k_list = args.k_list
        self.tags = tags
        self.eval_methods = eval_methods
        self.topk_methods = topk_methods
        self.counter = 0
        self.time_format = '%m/%d'

        self.static_info = [
            'dim',
            'dataset', 'lr', 'l2_weight', 'tolerance', 'early_stop',
            'batch_size',
            'n_memory',
        ]
        self.dynamic_info = [
            'round',
            'n_hop',
            'load_emb',
        ]

        self.scores = dict()
        self.scores_best = dict()
        self.scores_best_tmp = dict()

        self.is_early_stop = False
        self.is_refrsh = False
        self.early_stop_score_best = 0.
        self.early_stop_flag = 0

        self.avg_user_entity_interaction = dict()
        self.avg_user_entity_interaction_tmp = 0
        self.user_ere_interaction_dict = dict()

        self.user_ere_interaction = dict()
        self.user_ere_interaction_tmp = 0
        self.user_entity_interaction = dict()
        self.user_entity_interaction_tmp = 0

        self.all_user_entity_count = 1
        self.explored_rate = {}
        self.explored_rate_tmp = 0.

        self.init_train_info(args)
        self.init_scores(tags)

    def get_eval_methods(self):
        return self.get_eval_methods
  
    def check_refresh_state(self):
        if self.is_refrsh:
            self.early_stop_flag = 0
        else:
            self.early_stop_flag += 1

    def start_early_stop(self):
        self.is_early_stop = True
        self.early_stop_flag = 0

    def check_early_stop(self, number):
        return self.is_early_stop and self.early_stop_flag >= number

    def init_train_info(self, args):
        log_list = [f'{c}: {getattr(args, c)}' for c in self.static_info]
        log_str = f'Time: {time.strftime(self.time_format)}\n'
        for i, log in enumerate(log_list, 1):
            log_str += log + ('\t' if i % 6 != 0 else '\n')
        print(log_str)

        with open(self.folder_path, 'a') as f:
            f.write(log_str + '\n')
        with open(self.folder_path_best, 'a') as f:
            f.write(log_str + '\n')

    def init_scores(self, tags):
        for tag in tags:
            self.scores[tag] = {m: 0. for m in self.eval_methods}
            for m in self.topk_methods:
                self.scores[tag][m] = [0.] * len(self.k_list)
            self.scores[tag]['ere'] = 0

            self.scores_best[tag] = {method: 0. for method in self.eval_methods}
            for m in self.topk_methods:
                self.scores_best[tag][m] = [0.] * len(self.k_list)
            self.scores_best[tag]['ere'] = 0

            self.avg_user_entity_interaction[tag] = 0
            self.user_entity_interaction[tag] = 0
            self.user_ere_interaction[tag] = 0
            self.explored_rate[tag] = 0.
        self.tags = tags
            
    
    def update_cur_train_info(self, args, refresh_score=True, refresh_interaction=True, user_ere_interaction_dict={}, all_user_entity_count=1):
        self.is_refrsh = False
        
        self.all_user_entity_count = all_user_entity_count
        if refresh_score:
            for t in ['train', 'eval', 'test']:
                self.scores_best_tmp[t] = {m: 0. for m in self.eval_methods}
                for m in self.topk_methods:
                    self.scores_best_tmp[t][m] = [0.] * len(self.k_list)
            self.scores_best_tmp['ere'] = 0
            
        if refresh_interaction:
            self.user_ere_interaction_dict = user_ere_interaction_dict
        else:
            self.user_ere_interaction_dict = {u: list(set(e_d + self.user_ere_interaction_dict[u]))  for u, e_d in user_ere_interaction_dict.items()}
        
        user_e_count = 0
        # ere is entity relation entity
        user_ere_count = 0
        user_ere_set = set()
        
        for ere_list in self.user_ere_interaction_dict.values():
            user_e_list = np.array([[h, t] for h, _, t in ere_list]).flatten()
            user_e_count += np.unique(user_e_list).shape[0]
            user_ere_count += len(set(ere_list))
            user_ere_set.update(ere_list)
        
        self.user_entity_interaction_tmp = user_e_count
        self.user_ere_interaction_tmp = user_ere_count
        self.avg_user_entity_interaction_tmp = user_e_count / len(self.user_ere_interaction_dict.keys())
        self.explored_rate_tmp = len(user_ere_set) / all_user_entity_count * 100

        log_list = [f'{c}: {getattr(args, c)}' for c in self.dynamic_info]
        log_str = f'Time: {time.strftime(self.time_format)}'
        for i, log in enumerate(log_list, 1):
            log_str += ', ' + log + ('\n' if i % 6 == 0 else '')
        log_str += '\n'
        # log_str += ('avg i.e.: %.1f' % (self.avg_user_entity_interaction_tmp))
        # log_str += (' | total i.e.: %d' % (self.user_entity_interaction_tmp))
        log_str += ('ere: %d' % (self.user_ere_interaction_tmp))
        log_str += (' | e.r.: %.8f%%' % (self.explored_rate_tmp))
        print(log_str)
        
        with open(self.folder_path, 'a') as f:
            f.write(log_str + '\n')
        with open(self.folder_path_best, 'a') as f:
            f.write(log_str + '\n')

    def update_score(self, step, scores):
        log = 'Ep%2d' % (step)
        for t in ['train', 'eval', 'test']:
            log += f'|{t}|'
            for m in self.eval_methods:
                log += ('%s|%.8f|' % (m, scores[t][m])).replace('0.', '.')

        if self.topk_eval:
            for t in ['eval', 'test']:
                log += f'\n    |{t}|'
                for m in self.topk_methods:
                    s = self.topk_score_transform(scores[t][m])
                    log += f'{m}|{s}|'
                
        print(log)

        with open(self.folder_path, 'a') as f:
            f.write(log + '\n')
        
        if scores['train']['auc'] > self.scores_best_tmp['train']['auc']:
            self.scores_best_tmp['train'] = scores['train']

        if self.topk_eval and scores['eval']['r'][-1] > self.scores_best_tmp['eval']['r'][-1]:
            self.is_refrsh = True
            for m in self.topk_methods:
                self.scores_best_tmp['eval'][m] = scores['eval'][m]
                self.scores_best_tmp['test'][m] = scores['test'][m]

        if scores['eval']['auc'] > self.scores_best_tmp['eval']['auc']:
            if not self.topk_eval:
                self.is_refrsh = True
            for m in self.eval_methods:
                self.scores_best_tmp['eval'][m] = scores['eval'][m]
                self.scores_best_tmp['test'][m] = scores['test'][m]
            self.scores_best_tmp['ere'] = self.user_ere_interaction_tmp

    def train_over(self, tag):
        log = 'Best'
        for t in ['train', 'eval', 'test']:
            log += f'|{t}|'
            for m in self.eval_methods:
                log += ('%s|%.8f|' % (m, self.scores_best_tmp[t][m])).replace('0.', '.')
            
        if self.topk_eval:
            for t in ['eval', 'test']:
                log += f'\n    |{t}|'
                for m in self.topk_methods:
                    s = self.topk_score_transform(self.scores_best_tmp[t][m])
                    log += f'{m}|{s}|'

        print(log)

        with open(self.folder_path, 'a') as f:
            f.write('*'*100 + '\n')
            f.write(log + '\n')
            f.write('*'*100 + '\n')
        with open(self.folder_path_best, 'a') as f:
            f.write(log + '\n')
        
        for m in self.eval_methods:
            self.scores_best[tag][m] += self.scores_best_tmp['test'][m]
        for m in self.topk_methods:
            for k in range(len(self.k_list)):
                self.scores_best[tag][m][k] += self.scores_best_tmp['test'][m][k]
        self.scores_best[tag]['ere'] += self.scores_best_tmp['ere']
        self.avg_user_entity_interaction[tag] += self.avg_user_entity_interaction_tmp
        self.user_entity_interaction[tag] += self.user_entity_interaction_tmp
        self.user_ere_interaction[tag] += self.user_ere_interaction_tmp
        self.explored_rate[tag] += self.explored_rate_tmp
        

    def record_final_score(self):
        self.is_early_stop = False
        self.counter += 1

        log = ''
        max_tag_len = max([len(t) for t in self.tags])
        for tag in self.tags:
            log += tag + ' '*(max_tag_len + 1 - len(tag)) + '|'
            
            for m in self.eval_methods:
                s = round(self.scores_best[tag][m]/self.counter, 8)
                log += ('%s|%.8f|' % (m, s)).replace('0.', '.')
            if self.topk_eval:
                for m in self.topk_methods:
                    s = self.topk_score_transform([self.scores_best[tag][m][k]/self.counter for k in range(len(self.k_list))])
                    log += '%s|%s|' % (m, s)
            log += ('|ere|%d|' % (self.scores_best[tag]['ere']/self.counter))
            log += '\n'
        print('*' * 120)
        print(log)
        print('*' * 120)

        with open(self.folder_path_best, 'a') as f:
            f.write(log)
            f.write('*'*120 + '\n')
    
    def topk_score_transform(self, scores):
        return ' '.join([('%.8f' % (round(scores[k], 8))).replace('0.', '.') for k in range(len(self.k_list))])
