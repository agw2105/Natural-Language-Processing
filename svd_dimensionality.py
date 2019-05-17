import numpy
import nltk
import sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import tokenizer as t

corpus = pd.read_csv("data.csv")

totalvocab_stemmed = []
totalvocab_tokenized = []

for i in corpus:
    allwords_stemmed = t.tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    allwords_tokenized = t.tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)    
tfidf_vectorizer = TfidfVectorizer(min_df=0.02, max_df=0.90, stop_words='english',
                                  use_idf=True, tokenizer=t.tokenize_and_stem, ngram_range=(1,3))
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
terms = tfidf_vectorizer.get_feature_names()

print(tfidf_matrix.shape)

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=50, random_state=0) #reduce data to 50 dimensions
svd_tfidf = svd.fit_transform(tfidf_matrix)

print(svd_tfidf.shape)

from sklearn.manifold import TSNE

tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
tsne_tfidf = tsne_model.fit_transform(svd_tfidf)
tfidf_df = pd.DataFrame(tsne_tfidf, columns=['x', 'y'])