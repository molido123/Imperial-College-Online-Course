import nltk
import numpy as np
import pandas as pd
from nltk import SnowballStemmer, RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

noise = stopwords.words("english")


def stemming(token):
    stemming = SnowballStemmer("english")
    stemmed = [stemming.stem(each) for each in token]
    return stemmed


def tokenize(text):
    tokenizer = RegexpTokenizer(r'^[a-zA-Z]+$')  # 设置正则表达式规则
    tokens = tokenizer.tokenize(text)
    stems = stemming(tokens)
    return stems


def main():
    df = pd.read_csv("data/chatgpt-reddit-comments.csv")
    comment = df["comment_body"]
    comment = comment.dropna()
    comment = np.array(comment)
    CV = CountVectorizer(stop_words=noise, tokenizer=tokenize, lowercase=False)
    words = CV.fit_transform(comment)
    words_frequency = words.todense()  # convert into mertic
    words_list = CV.get_feature_names_out()  # all words
    print(words_frequency)
    print(words_list[5:100])


if __name__ == '__main__':
    main()
