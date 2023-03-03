import regex as re
import warnings

import gensim.corpora as corpora
import pandas as pd
import regex as re
import spacy
from gensim import models
from gensim.models import LdaModel
from matplotlib import pyplot as plt
from nltk.corpus import stopwords

# Gensim

stop_words = stopwords.words('english')
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Creating nlp object
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
# decide the number of models you want to get
NUM_ROUND = 16


def clean_text(text):
    # Remove URLs, mentions, and hashtags from the text
    # make all characters be lowercase
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r"\d", "", text)
    text = re.sub(r"\p{Emoji}", "", text)
    return text


# main function
def LDA_implementation():
    df = pd.read_csv("data/chatgpt-reddit-comments.csv")
    df.dropna(inplace=True)
    df['comment_body'] = df['comment_body'].apply(lambda x: clean_text(x))
    df.dropna(inplace=True)
    comments = df['comment_body']
    # get the comments list
    comment_list = comments.values
    # remove the stop words
    text_lists = [[word for word in d.lower().split() if word not in stop_words] for d in comment_list]
    # build the dictionary
    dictionary = corpora.Dictionary(text_lists)
    # remove the most frequent words
    dictionary.filter_extremes()
    dictionary.filter_n_most_frequent(15)
    print(dictionary.most_common(20))
    # trun the text lists into word bag lists
    corpus = [dictionary.doc2bow(text) for text in text_lists]
    # # train the model again
    # train_model(corpus, dictionary)
    # plot_perplexity(corpus)
    # compute_coherence(corpus,dictionary)
    #
    #
    # estimate the best model visually
    import pyLDAvis.gensim_models
    from gensim import models
    model_name = 'model/lda_3.model'  #
    pos_model = models.ldamodel.LdaModel.load(model_name)
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim_models.prepare(pos_model, corpus, dictionary)
    pyLDAvis.save_html(vis, 'general.html')


def train_model(corpus, dic):
    for i in range(1, NUM_ROUND):
        print('\n')
        print('nums of topic:{}'.format(i))
        temp = 'lda_{}'.format(i)
        tmp = LdaModel(corpus=corpus, num_topics=i, id2word=dic, passes=20)
        file_path = 'model/{}.model'.format(temp)
        tmp.save(file_path)
        print('------------------')


def plot_perplexity(corpus):
    x_list = []
    y_list = []
    for i in range(1, NUM_ROUND):
        temp_model = 'model/lda_{}.model'.format(i)
        try:
            lda = models.ldamodel.LdaModel.load(temp_model)
            perplexity = lda.log_perplexity(corpus)  # compute perplexity the lower the better
            x_list.append(i)
            y_list.append(perplexity)
        except Exception as e:
            print(e)
    plt.plot(x_list, y_list)
    plt.xlabel('num topics')
    plt.ylabel('perplexity score')
    plt.legend('perplexity_values', loc='best')
    plt.savefig('perplexity_values.png')
    plt.show()


from gensim.models import CoherenceModel


def compute_coherence(corpus, dictionary):
    x_list = []
    y_list = []
    for i in range(1, NUM_ROUND):
        temp_model = 'model/lda_{}.model'.format(i)
        try:
            # load the model
            lda = models.ldamodel.LdaModel.load(temp_model)
            # compute the model's coherence score, the higher the better
            cv_tmp = CoherenceModel(model=lda, corpus=corpus, dictionary=dictionary, coherence='u_mass')
            # compute the coherence
            x_list.append(i)
            y_list.append(cv_tmp.get_coherence())
        except:
            print('not found this model:{}'.format(temp_model))
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(x_list, y_list)
    plt.xlabel('num topics')
    plt.ylabel('coherence score')
    plt.legend('coherence_values', loc='best')
    plt.savefig('coherence_values.png')
    plt.show()


LDA_implementation()
