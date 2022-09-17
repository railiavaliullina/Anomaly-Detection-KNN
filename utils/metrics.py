import numpy as np


def precision_recall_curve(labels, scores, p=None, r=None):
    if np.any([p, r]) is None:
        p, r = precision_recall(labels)
    t = [np.mean([scores[i], scores[i + 1]]) for i in range(len(scores) - 1)]
    return p, r, t


def f1_score(precision, recall):
    return 2 * precision * recall / np.maximum(precision + recall, np.finfo(np.float64).eps)


def average_precision_score(precision, recall):
    recall = np.concatenate(([0.], recall, [1.]))
    precision = np.concatenate(([0.], precision, [0.]))
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])
    idx_with_recall_changes = np.where(recall[1:] != recall[:-1])[0]
    auc = (recall[idx_with_recall_changes + 1] -
           recall[idx_with_recall_changes]) * precision[idx_with_recall_changes + 1]
    ap = np.sum(auc)
    return ap


def confusion_matrix(labels, predictions):
    true_pred = (predictions == labels)
    false_pred = predictions != labels
    tp = np.sum(true_pred[predictions == 1])
    tn = np.sum(true_pred[predictions == 0])
    fp = np.sum(false_pred[predictions == 1])
    fn = np.sum(false_pred[predictions == 0])
    conf_matrix = [[tn, fp], [fn, tp]]
    return conf_matrix


def precision_recall(labels):
    tp = np.cumsum(labels)
    precision = tp / (np.arange(len(tp)) + 1)
    recall = tp / np.maximum(sum(labels == 1), np.finfo(np.float64).eps)
    return precision, recall
