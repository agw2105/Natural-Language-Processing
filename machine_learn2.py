import pandas as pd
import nltk
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as numpy
from sklearn.pipeline import Pipeline
import string
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

#import training and test data
df = pd.read_csv("training_set.csv")
data_frame = pd.DataFrame(df, columns = ["Title", "Class"], index=None)
#data_frame = data_frame.reindex(numpy.random.permutation(data_frame.index))

def stemming_tokenizer(text):
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in word_tokenize(text)]

data_train, data_test, y_train, y_true = \
    train_test_split(data_frame['Title'], data_frame['Class'], stratify=data_frame['Class'], test_size=0.2, random_state=42)

trial = Pipeline([('vectorizer', TfidfVectorizer(tokenizer = stemming_tokenizer)),('classifier', MultinomialNB(alpha=0.05))])
model = trial.fit(data_train, y_train)
y_test = model.predict(data_test)
print(sklearn.metrics.accuracy_score(y_true, y_test))
labels = [#list of labels]
cm = sklearn.metrics.confusion_matrix(y_true, y_test, labels=labels)
df_cm = pd.DataFrame(cm, index = [i for i in labels], columns = [i for i in labels])
sns.heatmap(df_cm, annot=True)

cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
scores = cross_val_score(trial, data_frame.Title, data_frame.Class, cv=10)
print(scores.mean())