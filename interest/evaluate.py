import numpy as np

def precision_at_k(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(k)
    return result

def recall_at_k(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(len(act_set))
    return result

def ndcg_at_k(actual, predicted, k):
    act_set = set(actual)
    pred_list = predicted[:k]
    dcg = 0.0
    for i, pred in enumerate(pred_list):
        if pred in act_set:
            dcg += 1.0 / np.log2(i + 2)  # log2(i+2) because i+1 is 1-based index and +1 for log
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(act_set), k))])
    return dcg / idcg if idcg > 0 else 0.0

def hit_rate_at_k(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    return 1.0 if len(act_set & pred_set) > 0 else 0.0