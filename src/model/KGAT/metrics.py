import numpy as np

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


def cal_precision_at_k(pred, answer, k):
    # print(k)
    # print(pred)
    return len(set(pred[:k]) & set(answer)) / k

def precision_at_k(preds, answers, top_k, weights=None):
    """ R@k metric

    Args:
        preds: batch of prediction indices. shape: (batch, k)
        answers: batch of answer indices. shape: (batch, random)
        k: list of top k. e.g. [10, 25, 50, 100]
    """
    # r_at_k = np.zeros(len(top_k))

    prec_at_k = cal_precision_at_k(preds, answers, top_k)
    return prec_at_k


def p_at_k(preds, answers, k):
    """ P@k metric

    Args:
        preds: prediction indices. shape: (k)
        answers: answer indices. shape: (random)
        k: topk, integer value
    """
    correct_num = 0.
    for pred in preds[:k]:
        if pred in answers:
            correct_num += 1
    return correct_num / k

def ap_at_k(preds, answers, k):
    """ AP@k metric
        AP@k = (1/k) * sum_i=0^k P@K  * rel(k)
    Args:
        preds: prediction indices. shape: (k)
        answers: answer indices. shape: (random)
        k: topk, integer value
    """
    AP_at_k = 0.
    for i in range(1, k+1): 
        #if preds[i-1] in answers:
        AP_at_k += p_at_k(preds, answers, i)
    
    return AP_at_k / k

def map_at_k(preds, answers, top_k, weights=None):
    """ MAP@k metric

    Args:
        preds: batch of prediction indices. shape: (batch, k)
        answers: batch of answer indices. shape: (batch, random)
        weights: occurance of each article in log
        top_k: list of top k. e.g. [10, 25, 50, 100]
    """
    # map_at_k = np.zeros(len(top_k))
    

    map_at_k = ap_at_k(preds, answers, top_k) # drop first prediction, since it is query itself
    return map_at_k

def cal_recall_at_k(pred, answer, k):
    # print(k)
    # print(pred)
    return len(set(pred[:k]) & set(answer)) / len(answer)

def recall_at_k(preds, answers, top_k, weights=None):
    """ R@k metric

    Args:
        preds: batch of prediction indices. shape: (batch, k)
        answers: batch of answer indices. shape: (batch, random)
        k: list of top k. e.g. [10, 25, 50, 100]
    """
    # r_at_k = np.zeros(len(top_k))

    r_at_k = cal_recall_at_k(preds, answers, top_k)
    return r_at_k

def cal_hr_at_k(pred, answer, k):
    return 1 if len(set(pred[:k]) & set(answer)) > 0 else 0

def hit_ratio_at_k(preds, answers, top_k, weights=None):
    """ HR@k metric

    Args:
        preds: batch of prediction indices. shape: (batch, k)
        answers: batch of answer indices. shape: (batch, random)
        k: list of top k. e.g. [10, 25, 50, 100]
    """
    # predict_record_matrix = []
    # hr_at_k = np.zeros(len(top_k))
    
    hr_at_k = cal_hr_at_k(preds, answers, top_k)
    return hr_at_k

def cal_mrr_at_k(pred, answer, k):
    for pos, p in enumerate(pred[:k], 1):
        if p in answer:
            #print("position", pos)
            return 1. / pos
    return 0

def mrr_at_k(preds, answers, top_k, weights=None):
    """ MRR@k metric

    Args:
        preds: batch of prediction indices. shape: (batch, k)
        answers: batch of answer indices. shape: (batch, random)
        k: list of top k. e.g. [10, 25, 50, 100]
    """

    mrr_at_k = cal_mrr_at_k(preds, answers, top_k)
    return mrr_at_k