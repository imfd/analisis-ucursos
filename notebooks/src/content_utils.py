from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, Counter
import spacy
import threading
import requests
import string
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from gensim.models.phrases import Phrases, Phraser
from nltk.corpus import stopwords
from textacy import preprocessing
import logging

from typing import List, Any

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

nlp = spacy.load('es_core_news_sm')  # default
# nlp_es = spacy.load('/tmp/sbwc')  # con spanish word embeddings

sw = stopwords.words('spanish')
sw2 = spacy.lang.es.stop_words.STOP_WORDS

punct = string.punctuation + '“' + '”' + '¿' + '⋆' + '�'
URL_REGEX = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
DECIMAL_REGEX = '[0-9]+,[0-9]+'

def tokenize_pipe(docs, nlp, include_stopwords=False,
                  include_punctuation=False,
                  include_quotes=False,
                  lemmatize=False):
    for tokens in nlp.pipe(docs):
        processed = []
        for token in tokens:
            word = None
            if token.is_digit or re.match(DECIMAL_REGEX, token.lower_):
                word = "NUMBER"
            elif re.match(URL_REGEX, token.lower_):
                word = "URL"
            elif token.is_stop or token.lower_ in sw:
                word = "STOPWORD" if include_stopwords else None
            elif (token.is_punct or token.lower_ in punct) and not token.is_quote:
                word = "PUNCT" if include_punctuation else None
            elif token.is_quote:
                word = "QUOTE" if include_quotes else None
            else:
                if lemmatize:
                    word = token.lemma_
                else:
                    word = token.text
                word = word.lower()
                word = word.strip()
                word = preprocessing.remove_accents(word)
                word = word if word != '' else None

            if word:
                processed.append(word)
        yield processed

def tokenize(doc,
             nlp,
             include_stopwords=False,
             include_punctuation=False,
             include_quotes=False,
             lemmatize=False):
    tokens = nlp(" ".join(doc.split()))
    processed = []
    for token in tokens:
        word = None
        if token.is_digit or re.match(DECIMAL_REGEX, token.lower_):
            word = "NUMBER"
        elif re.match(URL_REGEX, token.lower_):
            word = "URL"
        elif token.is_stop or token.lower_ in sw:
            word = "STOPWORD" if include_stopwords else None
        elif (token.is_punct or token.lower_ in punct) and not token.is_quote:
            word = "PUNCT" if include_punctuation else None
        elif token.is_quote:
            word = "QUOTE" if include_quotes else None
        else:
            if lemmatize:
                word = token.lemma_
            else:
                word = token.text
            word = word.lower()
            word = word.strip()
            word = preprocessing.remove_accents(word)
            word = word if word != '' else None

        if word:
            processed.append(word)
    return processed


def extract_phrases(corpus, min_count=5, threshold=100):
    import gensim
    bigrams = gensim.models.Phrases(
        corpus, min_count=min_count, threshold=threshold)
    bigram_mod = gensim.models.phrases.Phraser(bigrams)

    trigrams = gensim.models.Phrases(
        bigrams[corpus], min_count=min_count, threshold=threshold)
    trigram_mod = gensim.models.phrases.Phraser(trigrams)

    return bigram_mod, trigram_mod


def make_ngrams(corpus, min_count=5, threshold=100):
    bmod, tmod = extract_phrases(corpus, min_count, threshold)
    return [tmod[bmod[doc]] for doc in corpus]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# PMI
def fit_pmi(X: List[List[Any]], y: List[str], label_pos: str, label_neg: str):
    p_y = {
        0: Counter(y)[label_neg] / len(y),
        1: Counter(y)[label_pos] / len(y)
    }

    p_x = Counter()
    p_xy = defaultdict(lambda: {0: 1, 1: 1})
    for x, lbl in zip(X, y):
        for elem in x:
            p_x[elem] += 1
            p_xy[elem][1 if lbl == label_pos else 0] += 1

    def pmi(token, label):
        return np.log(p_xy[token][1 if label == label_pos else 0] / (p_x[token] * p_y[1 if label == label_pos else 0]))

    return pmi


def top_discriminative(X, pmi, label_pos, label_neg):
    vocab = set([e for x in X for e in x])
    scores = []
    for e in vocab:
        scores.append((e, abs(pmi(e, label_pos) - pmi(e, label_neg))))
    return sorted(scores, reverse=True, key=lambda s: s[1])


def load_lexicon():
    from pathlib import Path
    from tqdm import tqdm
    LEX_PATH = Path("/Users/mquezada/IMFD/projects/fb-ss1/NRC-Sentiment-Emotion-Lexicons/NRC-Sentiment-Emotion-Lexicons/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-v0.92-In105Languages-Nov2017Translations.xlsx")
    lex = pd.read_excel(LEX_PATH)
    lex = lex[['English (en)', 'Spanish (es)', 'Positive', 'Negative', 'Anger',
               'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust']]

    key = 'Spanish (es)'
    lexicon = {}
    for _, row in tqdm(lex.iterrows(), total=len(lex.index)):
        lexicon[row[key]] = (row['Positive'], row['Negative'])

    return lexicon


def sent_words(doc, lexicon):
    pos = []
    neg = []
    neutral = []
    # conteo de palabras pos y neg
    for token in doc:
        sent = lexicon.get(token)
        if not sent:
            continue
        if sent[0] > 0:
            pos.append(token)
        elif sent[1] > 0:
            neg.append(token)
        else:
            neutral.append(token)
    return pos, neg, neutral


def count_sent_words(corpus, lexicon):
    total_pos, total_neg, total_neutral = 0, 0, 0
    for doc in corpus:
        pos, neg, neutral = sent_words(doc, lexicon)
        total_pos += len(pos)
        total_neg += len(neg)
        total_neutral += len(neutral)
    return total_pos, total_neg, total_neutral
