import re
from collections import Counter

import nltk
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nltk.corpus import stopwords

noise = stopwords.words("english")


# data  [("word", frequency)]
def depict(data: list[tuple[str, int]], name: str):
    # depict the picture of most common adj
    x_values = [word[0] for word in data]
    y_values = [word[1] for word in data]
    plt.bar(x_values, y_values)
    plt.xlabel('Word')
    plt.xticks(rotation=45)
    plt.ylabel('Frequency')
    plt.title('Most Commonly Used ' + name + ' Words')
    plt.savefig("Most Commonly Used " + name + " Words.png")
    plt.show()


def part_of_speech(words):
    # Define the classes and their corresponding keywords
    data = []
    # set the sample size
    count = 1000000
    for sent in words:
        if count < 0:
            break
        count -= 1
        data.append(nltk.pos_tag(nltk.word_tokenize(sent)))
    # collect all the words with label JJ
    adj = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j][1] == "JJ":
                adj.append(data[i][j][0])
    words_count = Counter(adj)
    # depict the picture of most common adj
    top_words = words_count.most_common(10)
    # depict it
    depict(top_words, "ADJ")
    #
    #
    # depict the picture of most common JJS
    JJS = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j][1] == "JJS":
                JJS.append(data[i][j][0])
    words_count = Counter(JJS)
    # depict the picture of most common noun
    depict(words_count.most_common(10), "JJS")
    #
    #
    # depict the picture of most common verb
    verb = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j][1] == "VB":
                verb.append(data[i][j][0])
    words_count = Counter(verb)
    # depict the picture of most common verb
    depict(words_count.most_common(10), "VB")
    #
    #
    # depict the picture of most common noun
    noun = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j][1] == "NN":
                noun.append(data[i][j][0])
    words_count = Counter(noun)
    # depict the picture of most common noun
    depict(words_count.most_common(10), "NN")
    #


def general_counter(words):
    # Count the frequency of each word
    # this Counter function is from Collections library
    word_counts = Counter(words)
    # Create a bar chart of the most common words
    top_words = word_counts.most_common(10)  # Change the number to show more/less words
    x_values = [word[0] for word in top_words]
    y_values = [word[1] for word in top_words]
    plt.bar(x_values, y_values)
    plt.xlabel('Word')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.title('Most Commonly Used Words in general')
    plt.savefig("Most Commonly Used Words in general.png")
    plt.show()


def main():
    df = pd.read_csv("data/chatgpt-reddit-comments.csv")
    comment = df["comment_body"]
    comment = comment.dropna()
    comment = np.array(comment)
    # Concatenate all texts into a single string
    text = " ".join(comment)
    # Remove URLs, mentions, and hashtags from the text
    # make all characters be lowercase
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'#\S+', '', text)
    words = text.split()
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if not word in stop_words]
    #
    general_counter(words)
    part_of_speech(words)


if __name__ == '__main__':
    main()
