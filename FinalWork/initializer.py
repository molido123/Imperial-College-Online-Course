import re
from collections import Counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nltk.corpus import stopwords

noise = stopwords.words("english")


# detect meaningless words
def is_meaningless_str(s):
    # meaningfully word won't be too long
    if len(s) > 10 or not set(s).issubset(set('0123456789abcdef')):
        return True
    # after removing duplicated words, if the length reduce too much, we can assume it may be meaningless.
    return len(set(s)) < len(s)


# only can handle one word at a time


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
    # Count the frequency of each word
    # this Conter function is from Collections library
    word_counts = Counter(words)
    # Create a bar chart of the most common words
    top_words = word_counts.most_common(10)  # Change the number to show more/less words
    x_values = [word[0] for word in top_words]
    y_values = [word[1] for word in top_words]
    plt.bar(x_values, y_values)
    plt.xlabel('Word')
    plt.ylabel('Frequency')
    plt.title('Most Commonly Used Words in general')
    plt.savefig("Most Commonly Used Words in general.png")


if __name__ == '__main__':
    main()
