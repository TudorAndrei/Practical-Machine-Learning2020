import random
import unicodedata
from multiprocessing import Pool

import autocorrect
import numpy as np
import pandas as pd
import regex as re
import spacy
from nltk.stem.snowball import SnowballStemmer
from pandarallel import pandarallel
from tqdm import tqdm

# !python -m spacy download en


nlp = spacy.load('en')


path = r'data/content_dataset.csv'
data = pd.read_csv(path, encoding='utf-8')
data_full = pd.read_csv(r'data/content_dataset_full.csv', encoding='utf-8')


spl = autocorrect.Speller(lang='en')


def remove_xa0(text):
    # Remove the \xa0 symbol from the text, this appears due to some encoding error
    return unicodedata.normalize("NFKD",  text)


def remove_symbols(text):
    # remove anything that's not alphanumeric and whitespace
    pattern = '[^\w\s]|-|_'
    return re.sub(pattern, "", text)


def stemmer(text):
    stem = SnowballStemmer(language='english')
    stems = []
    for word in text.split():
        stems.append(stem.stem(word))

    return " ".join(stems)


def preprocess(dataframe, sentences=1000, sample=True):

    if sample:

        df = dataframe.head(sentences).copy()
    else:
        df = dataframe.copy()
    methods = {'remove "\\xa0"': remove_xa0,
               'remove symbols': remove_symbols,
               'stemming': stemmer}

    for method in methods.keys():
        print(f"Performing method {method}")
        df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: methods[method](x))

    return df


processed_df = preprocess(data_full, sample=False)

processed_df.to_csv("./data/processed_full.csv", index=False)
