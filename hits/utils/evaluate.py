# Based on implementations by https://github.com/pan-webis-de/pan-code/blob/master/clef22/authorship-verification/pan22_verif_evaluator.py

import json
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import (roc_auc_score,
                             accuracy_score,
                             f1_score,
                             brier_score_loss,
                             precision_score,
                             recall_score)

def binarize(y, threshold=0.5, triple_valued=False):
    y = np.array(y)
    y = np.ma.fix_invalid(y, fill_value=threshold)
    if triple_valued:
        y[y > threshold] = 1
    else:
        y[y >= threshold] = 1
    y[y < threshold] = 0
    return y

def auc(true_y, pred_y):
    """
    Calculates the AUC score (Area Under the Curve), a well-known
    scalar evaluation score for binary classifiers. This score
    also considers "unanswered" problem, where score = 0.5.
    Parameters
    ----------
    prediction_scores : array [n_problems]
        The predictions outputted by a verification system.
        Assumes `0 >= prediction <=1`.
    ground_truth_scores : array [n_problems]
        The gold annotations provided for each problem.
        Will typically be `0` or `1`.
    Returns
    ----------
    auc = the Area Under the Curve.
    References
    ----------
        E. Stamatatos, et al. Overview of the Author Identification
        Task at PAN 2014. CLEF (Working Notes) 2014: 877-897.
    """
    try:
        return roc_auc_score(true_y, pred_y)
    except ValueError:
        return 0.0


def c_at_1(true_y, pred_y, threshold=0.5):
    """
    Calculates the c@1 score, an evaluation method specific to the
    PAN competition. This method rewards predictions which leave
    some problems unanswered (score = 0.5). See:
        A. Peñas and A. Rodrigo. A Simple Measure to Assess Nonresponse.
        In Proc. of the 49th Annual Meeting of the Association for
        Computational Linguistics, Vol. 1, pages 1415-1424, 2011.
    Parameters
    ----------
    prediction_scores : array [n_problems]
        The predictions outputted by a verification system.
        Assumes `0 >= prediction <=1`.
    ground_truth_scores : array [n_problems]
        The gold annotations provided for each problem.
        Will always be `0` or `1`.
    Returns
    ----------
    c@1 = the c@1 measure (which accounts for unanswered
        problems.)
    References
    ----------
        - E. Stamatatos, et al. Overview of the Author Identification
        Task at PAN 2014. CLEF (Working Notes) 2014: 877-897.
        - A. Peñas and A. Rodrigo. A Simple Measure to Assess Nonresponse.
        In Proc. of the 49th Annual Meeting of the Association for
        Computational Linguistics, Vol. 1, pages 1415-1424, 2011.
    """

    n = float(len(pred_y))
    nc, nu = 0.0, 0.0

    for gt_score, pred_score in zip(true_y, pred_y):
        if pred_score == 0.5:
            nu += 1
        elif (pred_score > 0.5) == (gt_score > 0.5):
            nc += 1.0
    
    return (1 / n) * (nc + (nu * nc / n))


def accuracy(true_y, pred_y):
    """
    Assesses verification performance, assuming that every
    `score > 0.5` represents a same-author pair decision.
    Note that all non-decisions (scores == 0.5) are ignored
    by this metric.
    Parameters
    ----------
    prediction_scores : array [n_problems]
        The predictions outputted by a verification system.
        Assumes `0 >= prediction <=1`.
    ground_truth_scores : array [n_problems]
        The gold annotations provided for each problem.
        Will typically be `0` or `1`.
    Returns
    ----------
    acc = The number of correct attributions.
    References
    ----------
        E. Stamatatos, et al. Overview of the Author Identification
        Task at PAN 2014. CLEF (Working Notes) 2014: 877-897.
    """
    true_y_filtered, pred_y_filtered = [], []

    for true, pred in zip(true_y, pred_y):
        if pred != 0.5:
            true_y_filtered.append(true)
            pred_y_filtered.append(pred)
    
    pred_y_filtered = binarize(pred_y_filtered)

    return accuracy_score(true_y_filtered, pred_y_filtered)

def f1(true_y, pred_y):
    """
    Assesses verification performance, assuming that every
    `score > 0.5` represents a same-author pair decision.
    Note that all non-decisions (scores == 0.5) are ignored
    by this metric.
    Parameters
    ----------
    prediction_scores : array [n_problems]
        The predictions outputted by a verification system.
        Assumes `0 >= prediction <=1`.
    ground_truth_scores : array [n_problems]
        The gold annotations provided for each problem.
        Will typically be `0` or `1`.
    Returns
    ----------
    acc = The number of correct attributions.
    References
    ----------
        E. Stamatatos, et al. Overview of the Author Identification
        Task at PAN 2014. CLEF (Working Notes) 2014: 877-897.
    """
    true_y_filtered, pred_y_filtered = [], []

    for true, pred in zip(true_y, pred_y):
        if pred != 0.5:
            true_y_filtered.append(true)
            pred_y_filtered.append(pred)
    
    pred_y_filtered = binarize(pred_y_filtered)

    return f1_score(true_y_filtered, pred_y_filtered)

def precision(true_y, pred_y):
    true_y_filtered, pred_y_filtered = [], []

    for true, pred in zip(true_y, pred_y):
        if pred != 0.5:
            true_y_filtered.append(true)
            pred_y_filtered.append(pred)
    
    pred_y_filtered = binarize(pred_y_filtered)
    return precision_score(true_y_filtered, pred_y_filtered)

def recall(true_y, pred_y):
    true_y_filtered, pred_y_filtered = [], []

    for true, pred in zip(true_y, pred_y):
        if pred != 0.5:
            true_y_filtered.append(true)
            pred_y_filtered.append(pred)
    
    pred_y_filtered = binarize(pred_y_filtered)

    return recall_score(true_y_filtered, pred_y_filtered)

def f_05_u_score(true_y, pred_y, pos_label=1, threshold=0.5):
    """
    Return F0.5u score of prediction.
    :param true_y: true labels
    :param pred_y: predicted labels
    :param threshold: indication for non-decisions (default = 0.5)
    :param pos_label: positive class label (default = 1)
    :return: F0.5u score
    """

    pred_y = binarize(pred_y, triple_valued=True)

    n_tp = 0
    n_fn = 0
    n_fp = 0
    n_u = 0

    for i, pred in enumerate(pred_y):
        if pred == threshold:
            n_u += 1
        elif pred == pos_label and pred == true_y[i]:
            n_tp += 1
        elif pred == pos_label and pred != true_y[i]:
            n_fp += 1
        elif true_y[i] == pos_label and pred != true_y[i]:
            n_fn += 1

    return (1.25 * n_tp) / (1.25 * n_tp + 0.25 * (n_fn + n_u) + n_fp)

def evaluate_all(true_y, pred_y):
    results = {'auc': auc(true_y, pred_y),
               'c@1': c_at_1(true_y, pred_y),
               'f_05_u': f_05_u_score(true_y, pred_y),
               'F1': f1(true_y, pred_y),
              }
    results['overall'] = np.mean(list(results.values()))
    return results

def evaluate_all2(true_y, pred_y):
    results = {'Accuracy': accuracy(true_y, pred_y),
               'Precision': precision(true_y, pred_y),
               'Recall': recall(true_y, pred_y),
               'F1': f1(true_y, pred_y),
              }
    return results

def run_av_eval(pred_path, test_path, output_path):
    pred = pd.read_csv(pred_path)
    test = pd.read_csv(test_path)
    pred_col = 'pred'
    test_col = test.columns[0]
    true_y, pred_y = list(test[test_col]), list(pred[pred_col])
    results = evaluate_all(true_y, pred_y)
    print(results)
    with open(f"{output_path}/results.csv", 'w') as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', type=str)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--output_path', type=str)
    
    args = parser.parse_args()
    run_av_eval(args.pred_path, args.test_path, args.output_path)
    