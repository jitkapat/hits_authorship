import pandas as pd
import numpy as np
import os
import tqdm
import logging
import argparse
from hits.av_methods.ngram_dist import correct_scores
from hits.preprocessing.dset_utils import sample_pairs
from hits.preprocessing.evaluate import evaluate_all
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
#Selects optimal threshold from dev set and computes predictions on test set.

def compute_similarity(df, model):
    sim = []
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        text1, text2 = row['text1'], row['text2']
        vec1 = model.encode([text1])
        vec2 = model.encode([text2])
        sim.append(cosine_similarity(vec1, vec2)[0])
    return sim

def select_threshold(similarities, labels):   
    logging.info('Grid search p1/p2:')
    step_size = 0.01
    thresholds = np.arange(0.01, 0.99, step_size)
    combs = [(p1, p2) for (p1, p2) in combinations(thresholds, 2) if p1 < p2]

    # calculate scores for all combinations
    params = {}
    for p1, p2 in combs:
        corrected_scores = np.array(list(correct_scores(similarities, p1=p1, p2=p2)))
        score = evaluate_all(labels, corrected_scores)['overall']
        params[(p1, p2)] = score
    opt_p1, opt_p2 = max(params, key=params.get)
    logging.info('optimal p1/p2:', opt_p1, opt_p2)
    
    corrected_scores = np.array(list(correct_scores(similarities, p1=opt_p1, p2=opt_p2)))
    logging.info('optimal score:', roc_auc_score(labels, corrected_scores))

    logging.info('-> determining optimal threshold')
    scores = []
    for th in np.linspace(0.25, 0.75, 1000):
        adjusted = (corrected_scores >= th) * 1
        scores.append((th,
                    roc_auc_score(labels, adjusted)))
    thresholds, auc = zip(*scores)

    max_idx = np.array(auc).argmax()
    max_auc = auc[max_idx]
    max_th = thresholds[max_idx]
    logging.info(f'Dev results -> AUC={max_auc} at th={max_th}')
    return opt_p1, opt_p2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--output_path', type=str)
    
    args = parser.parse_args()
    
    model = SentenceTransformer(args.model_path)
    
    train_df = pd.read_csv(args.train_path)
    train_pairs = sample_pairs(train_df)
    train_sim = compute_similarity(train_pairs, model)
    p1, p2 = select_threshold(train_sim, train_pairs['label'])
    
    test_df = pd.read_csv(args.test_path)
    test_sim = compute_similarity(test_df, model)
    test_sim = np.array(list(correct_scores(test_sim, p1=p1, p2=p2)))
    
    pred = pd.DataFrame(test_sim, columns=['pred'])
    pred['pred'] = [sc for sc in correct_scores(pred['pred'], p1, p2)]
    pred.to_csv(os.path.join(args.output_path, 'pred.csv'))
    
    
    