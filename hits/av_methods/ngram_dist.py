# Modified from https://github.com/pan-webis-de/pan-code/blob/master/clef23/authorship-verification/pan23-verif-baseline-cngdist.py

import logging
import argparse
import random
import pickle
import os
import pandas as pd
import numpy as np
from tqdm import trange
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score
from hits.preprocessing.dset_utils import sample_pairs
from itertools import combinations
from hits.preprocessing.evaluate import evaluate_all, f1_score

def rescale(value, orig_min, orig_max, new_min, new_max):
    """
    Rescales a `value` in the old range defined by
    `orig_min` and `orig_max`, to the new range
    `new_min` and `new_max`. Assumes that
    `orig_min` <= value <= `orig_max`.
    Parameters
    ----------
    value: float, default=None
        The value to be rescaled.
    orig_min: float, default=None
        The minimum of the original range.
    orig_max: float, default=None
        The minimum of the original range.
    new_min: float, default=None
        The minimum of the new range.
    new_max: float, default=None
        The minimum of the new range.
    Returns
    ----------
    new_value: float
        The rescaled value.
    """

    orig_span = orig_max - orig_min
    new_span = new_max - new_min

    try:
        scaled_value = float(value - orig_min) / float(orig_span)
    except ZeroDivisionError:
        orig_span += 1e-6
        scaled_value = float(value - orig_min) / float(orig_span)

    return new_min + (scaled_value * new_span)

def correct_scores(scores, p1, p2):
    for sc in scores:
        if sc <= p1:
            yield rescale(sc, 0, p1, 0, 0.49)
        elif p1 < sc < p2:
            yield 0.5
        else:
            yield rescale(sc, p2, 1, 0.51, 1)  # np.array(list

class NGramDist():
    def __init__(self, vocab_size, ngram_size, analyzer='char') -> None:
        self.vocab_size = vocab_size
        self.ngram_size = ngram_size
        self.analyzer = analyzer
        
    
    def fit_vectorizer(self, train_df, vocab_size, ngram_size, analyzer):
        logging.info('Constructing vectorizer')
        
        vectorizer = TfidfVectorizer(max_features=vocab_size,
                                    analyzer=analyzer,
                                    ngram_range=(ngram_size, ngram_size))
        print('fitting vocabulary')
        vectorizer.fit(train_df['text'])
        self.vectorizer = vectorizer

    def compute_similarity(self, text_pairs):
        logging.info('Vectorizing data')
        print('transforming data')
        vectors1 = self.vectorizer.transform(text_pairs['text1'])
        vectors2 = self.vectorizer.transform(text_pairs['text2'])
        
        logging.info('Calculating pairwise similarities')
        similarities_pairwise = []
        for i in trange(vectors1.shape[0]):
            sim = cosine_similarity(vectors1[i], vectors2[i])[0][0]
            similarities_pairwise.append(sim)

        return similarities_pairwise
    
    def select_threshold(self, train_pairs):
        similarities = self.compute_similarity(train_pairs)
        labels = np.array(train_pairs['label'])
        
        logging.info('Grid search p1/p2:')
        step_size = 0.01
        thresholds = np.arange(0.01, 0.99, step_size)
        combs = [(p1, p2) for (p1, p2) in combinations(thresholds, 2) if p1 < p2]

        # calculate scores for all combinations
        params = {}
        for p1, p2 in combs:
            corrected_scores = np.array(list(correct_scores(similarities, p1=p1, p2=p2)))
            score = evaluate_all(labels, corrected_scores)
            params[(p1, p2)] = score['overall']
        opt_p1, opt_p2 = max(params, key=params.get)
        logging.info('optimal p1/p2:', opt_p1, opt_p2)

        self.opt_p1 = opt_p1
        self.opt_p2 = opt_p2
        
        corrected_scores = np.array(list(correct_scores(similarities, p1=opt_p1, p2=opt_p2)))
        logging.info('optimal score:', evaluate_all(labels, corrected_scores))

        #logging.info('-> determining optimal threshold')
        #scores = []
        #for th in np.linspace(0.25, 0.75, 1000):
        #    adjusted = (corrected_scores >= th) * 1
        #    scores.append((th,
        #                f1_score(labels, adjusted)))
        #thresholds, f1 = zip(*scores)

        #max_idx = np.array(f1).argmax()
        #max_f1 = f1[max_idx]
        #max_th = thresholds[max_idx]
        #self.threshold = max_th
        #with open('best_t.txt', 'w') as f:
        #    f.write(str(max_th))
        #logging.info(f'Dev results -> F1={max_f1} at th={max_th}')
    
    def fit(self, train_df):
        self.fit_vectorizer(train_df, self.vocab_size, self.ngram_size, self.analyzer)
        print("sampling training pairs")
        train_pairs = sample_pairs(train_df)
        self.select_threshold(train_pairs)
    
    def inference(self, df, pred_path):
        similarities = self.compute_similarity(df)
        pred = pd.DataFrame(similarities, columns=['pred'])
        pred['pred'] = [sc for sc in correct_scores(pred['pred'], self.opt_p1, self.opt_p2)]
        pred.to_csv(os.path.join(pred_path, 'pred.csv'))
    
def main():
    print('getting arguments')
    parser = argparse.ArgumentParser()
    # data settings:
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str, default='.')
    parser.add_argument('--test_path', type=str)
    # algorithmic settings:
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--vocab_size', default=3000, type=int,
                        help='Maximum number of vocabulary items in feature space')
    parser.add_argument('--ngram_size', default=4, type=int, help='Size of the ngrams')
    parser.add_argument('--analyzer', default='char', type=str, help='word or characterngram')


    args = parser.parse_args()
    print('setting seed')
    np.random.seed(args.seed)
    random.seed(args.seed)

    # training 
    print('initializing model')
    model = NGramDist(args.vocab_size, args.ngram_size, args.analyzer)
    print("Loading data")
    train_df = pd.read_csv(args.input_path)#.sample(1000)
    model.fit(train_df)
        
    # save model
    output_path = os.path.join(args.output_path, 'model.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
            
    # evaluate on test set
    test_df = pd.read_csv(args.test_path)
    model.inference(test_df, args.output_path)
    
if __name__ == '__main__':
    main()
