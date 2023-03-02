import nltk
import nltk.classify.util
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews


def extract_features(word_list):
    return dict([(word.lower(), True) for word in word_list])


def main():
    df = pd.read_csv("data/chatgpt-reddit-comments.csv")
    df.dropna(inplace=True)
    comment = df["comment_body"]
    comment = comment.dropna()
    comment = np.array(comment)
    # we use the moive reviews data to train my model
    # get the negative and postive data from nltk's corpus
    positive_fileids = movie_reviews.fileids('pos')
    negative_fileids = movie_reviews.fileids('neg')
    # extract the features of the negative and the postive data
    features_positive = [(extract_features(movie_reviews.words(fileids=[f])), 'Positive') for f in positive_fileids]
    features_negative = [(extract_features(movie_reviews.words(fileids=[f])), 'Negative') for f in negative_fileids]
    # define the features
    # we use 80% of the data to train the model and 20% to test it.
    threshold_factor = 0.8
    threshold_positive = int(threshold_factor * len(features_positive))
    threshold_negative = int(threshold_factor * len(features_negative))
    feature_train = features_positive[:threshold_positive] + features_negative[:threshold_negative]
    feature_test = features_positive[threshold_positive:] + features_negative[threshold_negative:]
    classifier = NaiveBayesClassifier.train(feature_train)
    print('Number of training datapoints:', len(feature_train))
    print('Number of test datapoints:', len(feature_test))
    print('Accuracy of the classifier:', nltk.classify.util.accuracy(classifier, feature_test))
    #
    # we use the trained model to do our work
    # tokenize the text
    input_reviews = comment.tolist()
    input_reviews = [x.lower() for x in input_reviews]
    attitude = []
    Neutral_point = 0.51
    NG = 0
    PS = 0
    NT = 0
    for review in input_reviews:
        # run the data in the classification
        probdist = classifier.prob_classify(extract_features(review.split()))
        # get the most probable sentiment
        pred_sentiment = probdist.max()
        prob = probdist.prob(pred_sentiment)
        if '[deleted]' == review or '[removed]' == review:
            continue
        if prob <= Neutral_point:
            pred_sentiment = "Neutral"
            NT = NT + 1
        if pred_sentiment == "Negative":
            NG = NG + 1
        else:
            PS = PS + 1
        attitude.append((review, pred_sentiment))
    # depict the sentiment
    x_values = ["Neutral", "Negative", "Positive"]
    y_values = [NT, NG, PS]
    plt.bar(x_values, y_values)
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.title("Sentiment Analysis")
    plt.savefig("SentimentAnalysis.png")
    plt.show()


if __name__ == '__main__':
    main()
