# Given a corpus, sample pairs of documents with label of whether the pair is written by the same author
# Dataset schema: .csv with columns: text, author, topic
# 50-50 (unless specified) ratio of same author and different author pairs

import random
import pandas as pd

random.seed(0)

def df_to_dict(df):
    """
    given a dataframe with columns: text, author, topic, return a dictionary of authors and a dictionary of topics
    """
    authors = {}
    for row in df.itertuples(index=False, name=None):
        author, topic, text = row
        #author, text = row
        if author not in authors:
            authors[author] = []
        authors[author].append(text)
    return authors

def sample_pairs(df, txt_size_lim=None):
    try:
        df = df[['author', 'topic', 'text']]
    except:
        pass
    data = df_to_dict(df)
    sampled_dset = []
    for auth, texts in data.items():
        for text in texts:
            # get a same_author sample - if not enough texts just skip
            if len(texts) > 1:
                same_auth_txts = [text_ for text_ in texts if text_ != text]
                if len(same_auth_txts) == 0:
                    continue
                same_auth_txt = random.choice(same_auth_txts)
                #while text == same_auth_txt:
                #    same_auth_txt = random.choice(data[auth])
                #    print(auth, same_auth_txt[:100])
                if txt_size_lim is None:
                    sampled_dset.append([1, text, same_auth_txt])
                else:
                    sampled_dset.append([1, text[:txt_size_lim], same_auth_txt[:txt_size_lim]])

            # get a different_author sample
            diff_auths = [auth_ for auth_ in data.keys() if auth_ != auth]
            diff_auth = random.choice(diff_auths)
            #while auth == diff_auth:
            #    diff_auth = random.choice(list(data.keys()))
            diff_auth_txt = random.choice(data[diff_auth])
            if txt_size_lim is None:
                sampled_dset.append([0, text, diff_auth_txt])
            else:
                sampled_dset.append([0, text[:txt_size_lim], same_auth_txt[:txt_size_lim]])
    return pd.DataFrame(sampled_dset, columns=['label', 'text1', 'text2'])

def sample_av_triplet(data, txt_size_lim=None):
    sampled_dset = []
    for auth, texts in data.items():
        for text in texts:
            # get a same_author sample - if not enough texts just skip
            if len(texts) < 2:
                continue
            same_auth_txt = random.choice(data[auth])
            while text == same_auth_txt:
                same_auth_txt = random.choice(data[auth])
                
            diff_auth = random.choice(list(data.keys()))
            while auth == diff_auth:
                diff_auth = random.choice(list(data.keys()))
            diff_auth_txt = random.choice(data[diff_auth])
                
            if txt_size_lim is None:
                sampled_dset.append([1, text, same_auth_txt, diff_auth_txt])
            else:
                sampled_dset.append([1, text[:txt_size_lim], same_auth_txt[:txt_size_lim], diff_auth_txt[:txt_size_lim]])

            # get a different_author sample
            if txt_size_lim is None:
                sampled_dset.append([0, text, diff_auth_txt])
            else:
                sampled_dset.append([0, text[:txt_size_lim], same_auth_txt[:txt_size_lim]])
    return sampled_dset

def read_train_data(dset_path):
    return pd.read_csv(f"{dset_path}/train.csv")

def read_test_data(dset_path):
    with open(f"{dset_path}/test_pairs.json") as f:
        return json.load(f)
