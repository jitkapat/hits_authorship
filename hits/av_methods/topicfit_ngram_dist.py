from hits.av_methods.ngram_dist import NGramDist
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import argparse
import requests, re
import pandas as pd
import numpy as np
import random
import pickle
import os

# modified from https://github.com/AuthorshipVerifier/TextDistortion (Stamatatos, 2017)

class TextDistorter:
    def __init__(self, word_list=None):
        """
        Text Distortion provides a technique to mask topic-related text units in a given text.
        Args:
            word_list: The list of text units (ordered by decreasing frequency) to be distorted. 
            If None, a list is loaded based on 'https://www.kilgarriff.co.uk/BNClists/variances'
        """
        self.word_list = word_list
        if not word_list:
            self.word_list = requests.get("https://www.kilgarriff.co.uk/BNClists/variances").text # load word list file
            self.word_list = [line.split()[0] for line in self.word_list.split("\n")[:-2]]             # parse words
            self.word_list = list(dict.fromkeys(self.word_list).keys())                                # remove duplicates

    def distort(self, text, k, multiple_asterisk=False, distort_char="*", digit_char="#"):
        """
        Distort topic-related text units in a given text.
        Args:
            k: The k-frequent words in the internal wordlist (these will be preserved in the masked text representation).
            multiple_asterisk: Flags which text distortion variant should be used (True = Multiple Asterisks, False = Single Asterisks).
            distort_char: The character used to mask NOT topic-related text units.
            digit_char: The character used to mask text units comprising digits.
        """
        word_set = set(self.word_list[:k])
        for match in reversed(list(re.finditer(r"\b\w+\b", text))):
            match_string = match.group()
            if match_string.isdigit():
                text = text[:match.start()] + digit_char * (len(match_string) if multiple_asterisk else 1) + text[match.end():]
            elif match_string.lower() in word_set:
                text = text[:match.start()] + distort_char * (len(match_string) if multiple_asterisk else 1) + text[match.end():]
        return text

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
    parser.add_argument('--k', default=10000, type=int, help='number of top k words to mask')

    args = parser.parse_args()
    print('setting seed')
    np.random.seed(args.seed)
    random.seed(args.seed)

    # mask training data
    print("Loading data")
    train_df = pd.read_csv(args.input_path)
    tokenizer = CountVectorizer().build_tokenizer()
    print("Tokenizing data")
    train_df['tokens'] = train_df['text'].apply(tokenizer)
    vocabs = Counter([word for sent in train_df['tokens'] for word in sent])
    word_list = [word for word, _ in vocabs.most_common(len(vocabs))]
    distorter = TextDistorter(word_list=word_list)
    train_df['text'] = train_df['text'].apply(lambda x: distorter.distort(x, args.k))
    train_df.drop(columns=['tokens'], inplace=True)
    
    # training 
    print('initializing model')
    model = NGramDist(args.vocab_size, args.ngram_size, args.analyzer)
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