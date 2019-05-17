from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

with open("list.csv") as csvfile:
    text = csvfile.read()
    
from scipy.misc import imread
mask = imread("mask.png", flatten=True)


stopwords = set(STOPWORDS)
wc = WordCloud(font_path="arial.ttf", width=1200, height=800, scale=5, max_font_size=100, background_color=None, mode="RGBA", max_words=1000, stopwords=stopwords, colormap="ocean_r")
wc.generate(text)
fig = plt.figure(figsize=(20,10), dpi=300)
plt.axis("off")
plt.show()
