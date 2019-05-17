import pandas as pd
import lda
from sklearn.feature_extraction.text import CountVectorizer
import logging
logging.getLogger("lda").setLevel(logging.WARNING)
import tokenizer as t

corpus = pd.read_csv("data.csv")

cvectorizer = CountVectorizer(min_df=0.02, max_features=10000, tokenizer=t.tokenize_and_stem, stop_words="english", ngram_range=(1,3))
cvz = cvectorizer.fit_transform(corpus)

n_topics = 20
n_iter = 2000
lda_model = lda.LDA(n_topics=n_topics, n_iter=n_iter)
X_topics = lda_model.fit_transform(cvz)

import numpy as np
n_top_words = 8
topic_summaries = []

topic_word = lda_model.topic_word_  # get the topic words
vocab = cvectorizer.get_feature_names()
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))
    print('Topic {}: {}'.format(i, ' '.join(topic_words).encode('utf-8')))

print(topic_words)

from svd_dimensionality import tsne_model
tsne_lda = tsne_model.fit_transform(X_topics) #import from svd model

doc_topic = lda_model.doc_topic_
lda_keys = []
for i, tweet in enumerate(corpus):
    lda_keys += [doc_topic[i].argmax()]
