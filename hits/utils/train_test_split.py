# Given a corpus of documents each in different topic, select subset of topics for test data and the rest for training data
# Dataset schema: .csv with columns: text, author, topic

import random
import numpy as np
from src.preprocessing.topic_vector import create_topic_vector

def random_split(df, n_test_topic=5):
    # get a random sample of n topics
    topics = set(df['topic'])
    topics = random.sample(topics, n_test_topic)
    # get the documents in those topics
    test_df = df[df['topic'].isin(topics)]
    train_df = df[~df['topic'].isin(topics)]
    return train_df, test_df

def random_search_split(df, n_test_topics, n_iter):
    """
    perform train-test split to maximize cosine similarity between topics in training data and test data
    """
    # create topic vector
    topic_vectors = create_topic_vector(df)
    
    # sample and pick the subset with the highest cosine similarity
    best_score = -999
    for _ in range(n_iter):
        test_topics = random.sample(topic_vectors.keys(), n_test_topics)
        test_topic_vectors = np.stack([topic_vectors[topic]
                                       for topic in test_topics],1)
        train_topic_vectors = np.stack([topic_vectors[topic]
                                        for topic in set(df.topic) if topic not in test_topics], 1)
        sim = np.dot(test_topic_vectors.T, train_topic_vectors)
        score = sim.mean(1).mean(0)
        if score > best_score:
            final_test_topic = test_topics
            best_score = score
    train_df = df[~df.topic.isin(final_test_topic)]
    test_df = df[df.topic.isin(final_test_topic)]
    return train_df, test_df

def cross_topic_kfold_cv(df, k=10):
    topics = list(set(df.topic))
    assert len(topics) % k == 0
    n_topics_per_fold = int(len(topics) / k)
    folds = []
    for i in range(0, len(topics), n_topics_per_fold):
        fold = []
        for j in range(n_topics_per_fold):
            fold.append([topics[i+j]])
        folds.append(fold)
        
    # create train-test split for each fold
    fold_splits = []
    for i, fold in enumerate(folds):
        test_topics = [topic for sublist in fold for topic in sublist]
        train_topics = [topic for topic in topics if topic not in test_topics]
        train_df = df[df.topic.isin(train_topics)]
        test_df = df[df.topic.isin(test_topics)]
        fold_splits.append((train_df, test_df))
    return fold_splits