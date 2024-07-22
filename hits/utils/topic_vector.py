# Given a corpus of documents each in different topic, create a vector representing each topic's documents
# Dataset schema: .csv with columns: text, author, topic

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# function for loading dataset
def load_csv(path):
    return pd.read_csv(path)

# transform texts into vectors using LDA
def lda_transform(texts,
                  max_df=0.5,
                  min_df=0.1,
                  max_features=5000,
                  n_components=50,
                  random_state=0):
    
    # transform into bag of words
    cv = CountVectorizer(max_df=max_df,
                         min_df=min_df,
                         max_features=max_features,
                         binary=True)
    bow_vector = cv.fit_transform(texts)
    # transform bow vector into LDA vector
    lda = LatentDirichletAllocation(n_components=n_components,
                                    random_state=random_state)
    lda_vector = lda.fit_transform(bow_vector)
    return lda_vector

def nmf_transform(texts,
                  max_df=0.5,
                  min_df=0.1,
                  max_features=5000,
                  n_components=50,
                  random_state=0):
    #transform into tfidf vector
    tfidf = TfidfVectorizer(max_df=max_df,
                            min_df=min_df,
                            max_features=max_features)
    tfidf_vector = tfidf.fit_transform(texts)
    # transform tfidf vector into NMF vector
    nmf = NMF(n_components=n_components,
              init='random',
              max_iter=2000,
              random_state=random_state)
    nmf_vector = nmf.fit_transform(tfidf_vector)
    return nmf_vector

def bert_transform(texts, model_path="/ist-project/scads/nook/journal/pretrained_models/all-MiniLM-L6-v2"):
    model_path = "/ist-project/scads/nook/journal/pretrained_models/all-MiniLM-L6-v2"
    # load model
    model = SentenceTransformer(model_path)
    # transform texts
    bert_vector = model.encode(texts)
    # transform into numpy array
    #bert_vector = bert_vector.numpy()
    return bert_vector

def create_topic_vector(df, mode='nmf', aggregate='mean', **kwargs):
    texts = list(df.text)
    # transform texts
    if mode == "lda":
        vectors = lda_transform(texts, **kwargs)
    elif mode == 'nmf':
        vectors = nmf_transform(texts, **kwargs)
    elif mode == 'bert':
        vectors = bert_transform(texts, **kwargs)
    else:
        raise NotImplementedError
    vectors = normalize(vectors)
    
    # add to dataframe
    df['vector'] = vectors.tolist()
    
    # create topic vector
    topic_vectors = {}
    for topic in sorted(list(set(df.topic))):
        topic_df = df[df.topic==topic]
        topic_doc_vectors = np.stack(list(topic_df.vector))
        if aggregate == "mean":
            topic_vector = np.mean(topic_doc_vectors, 0)
        elif aggregate == "median":
            topic_vector = np.median(topic_doc_vectors, 0)
        topic_vectors[topic] = topic_vector
    return topic_vectors
        