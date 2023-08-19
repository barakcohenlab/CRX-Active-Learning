"""
Wrapper functions for sklearn performance metrics for compabitility with Selene.
"""
import numpy as np
from scipy import stats
import sklearn.metrics as skmetrics
from sklearn.preprocessing import label_binarize


# Helper function to turn log probabilities into linear probabilities
def _decode_logs(y_score):
    y_score = np.exp(y_score)
    return y_score


def _auroc(y_true, y_score, average):
    y_true = label_binarize(y_true, classes=np.arange(y_score.shape[1]))
    return skmetrics.roc_auc_score(y_true, y_score, average=average, multi_class="ovr",
                                   labels=np.arange(y_score.shape[1]))


def _aupr(y_true, y_score, average):
    y_true = label_binarize(y_true, classes=np.arange(y_score.shape[1]))
    return skmetrics.roc_auc_score(y_true, y_score, average=average, multi_class="ovr",
                                   labels=np.arange(y_score.shape[1]))


def _aupr(y_true, y_score, average):
    y_true = label_binarize(y_true, classes=np.arange(y_score.shape[1]))
    return skmetrics.average_precision_score(y_true, y_score, average=average)


def _f1(y_true, y_score, average):
    # If average is None we end up getting the F1 for each class. Otherwise, we need to get the argmax.
    if average:
        y_score = y_score.argmax(axis=1)

    return skmetrics.f1_score(y_true, y_score, average=average)


def micro_auroc(y_true, y_score):
    y_score = _decode_logs(y_score)
    return _auroc(y_true, y_score, "micro")


def macro_auroc(y_true, y_score):
    y_score = _decode_logs(y_score)
    return _auroc(y_true, y_score, "macro")


def micro_aupr(y_true, y_score):
    y_score = _decode_logs(y_score)
    return _aupr(y_true, y_score, "micro")


def macro_aupr(y_true, y_score):
    y_score = _decode_logs(y_score)
    return _aupr(y_true, y_score, "macro")


def micro_auroc_resnet(y_true, y_score):
    return _auroc(y_true, y_score, "micro")


def macro_auroc_resnet(y_true, y_score):
    return _auroc(y_true, y_score, "macro")


def micro_aupr_resnet(y_true, y_score):
    return _aupr(y_true, y_score, "micro")


def macro_aupr_resnet(y_true, y_score):
    return _aupr(y_true, y_score, "macro")


def micro_f1(y_true, y_score):
    return _f1(y_true, y_score, "micro")


def macro_f1(y_true, y_score):
    return _f1(y_true, y_score, "macro")


def confusion_matrix(y_true, y_score):
    # If y_true is a column vector, flatten it
    if y_true.shape[1] == 1:
        y_true = y_true.flatten()
    # If y_true is a matrix, it's one-hot encoded data so we need to decode the labels
    else:
        y_true = y_true.argmax(axis=1)

    # Take argmax on probabilities
    y_preds = y_score.argmax(axis=1)
    return skmetrics.confusion_matrix(y_true, y_preds)


def pearson(y_true, y_score):
    return stats.pearsonr(y_true, y_score)[0]


def spearman(y_true, y_score):
    return stats.spearmanr(y_true, y_score)[0]
