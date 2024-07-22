# Given a corpus of documents each in different topic, pick a subset of n topics to constitute a new dataset
# Dataset schema: .csv with columns: text, author, topic
# output dataset schema: 
# AA: author_id, text
# AV: same/diff auth, text1, text2

from argparse import ArgumentParser
from datetime import datetime
from src.preprocessing.train_test_split import *
from src.preprocessing.dset_utils import sample_pairs
from src.preprocessing.run_sampling import *
import numpy as np
import pandas as pd
import random
import os

def run_sampling(df,
                 min_doc,
                 topic_vec_mode,
                 sampler,
                 n_topic,
                 split_method,
                 threshold,
                 k_fold,
                 n_doc_per_topic,
                 is_group=False):
    # preprocess
    df = filter_df_by_count(df, min_doc)
    
    if is_group == True:
        sampled_df = topic_cosine_similarity_grouping(df,
                                              vector_mode=topic_vec_mode,
                                              n=n_topic)
    else:
        # sample
        if sampler == "random":
            sampled_df = random_sample(df, n=n_topic)
        elif sampler == "cosine_iter":
            sampled_df = topic_cosine_similarity_sample_iterative(df,
                                                                  vector_mode=topic_vec_mode,
                                                                    n=n_topic,)
        else:
            raise NotImplementedError

    folds = cross_topic_kfold_cv(sampled_df, k=k_fold)
    for fold in folds:
        train_df, test_df = fold
        train_df = equalize_samples(train_df, n_doc_per_topic)
        test_df = equalize_samples(test_df, n_doc_per_topic)
        yield train_df, test_df
        

if __name__ == "__main__":
    startTime = datetime.now()
    
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, default="/ist/ist-share/scads/nook/authorship/journal/datasets/processed/pan_20_21/pan20-aa-training-large.csv")
    parser.add_argument("--output", type=str, default="test")
    parser.add_argument("--min_doc", type=int, default=1000)
    parser.add_argument("--topic_vec_mode", type=str, default="nmf")
    parser.add_argument("--sampler", type=str, default="cosine")
    parser.add_argument("--n_sampled_topic", type=int, default=50)
    parser.add_argument("--split_method", type=str, default="random")
    parser.add_argument("--n_test_topic", type=int, default=10)
    parser.add_argument("--n_doc_per_topic", type=int, default=1000)
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--k_fold", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.81)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group", action='store_true')
    args = parser.parse_args()
    
    # fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # read data
    df = pd.read_csv(args.input)
    
    # sample data
    print("sampling...")
    folds = run_sampling(df,
                        args.min_doc,
                        args.topic_vec_mode,
                        args.sampler,
                        args.n_sampled_topic,
                        args.split_method,
                        args.threshold,
                        args.k_fold,
                        args.n_doc_per_topic,
                        args.group)
    for k, data in enumerate(folds):
        train_df, test_df = data
        # remove vector column to save space
        try:
            train_df.drop(columns=['vector'], inplace=True)
            test_df.drop(columns=['vector'], inplace=True)
        except:
            pass
        
        # sample test pairs
        train_pairs = sample_pairs(train_df)
        test_pairs = sample_pairs(test_df)
        
        # convert to appropriate format
        train_df = convert_to_aa_format(train_df)
        test_df = convert_to_aa_format(test_df)
        #test_pairs = convert_to_av_format(test_pairs)
        
        # save sampled data to disk
        output_path = f"{args.output}_fold{k}"
        try:
            os.mkdir(output_path)
        except:
            pass
        
        train_df.to_csv(f"{output_path}/train.csv", index=False)
        test_df.to_csv(f"{output_path}/test_AA.csv", index=False)
        train_pairs.to_csv(f"{output_path}/train_AV.csv", index=False)
        test_pairs.to_csv(f"{output_path}/test_AV.csv", index=False)
        
    print(datetime.now() - startTime)