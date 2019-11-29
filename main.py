import pandas as pd
import numpy as np
import math
import datetime
from datetime import timedelta
import sqlite3
import re
import os
from os import listdir
from os.path import isfile, join
import warnings
import pytz
import pickle

from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GRU
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

from nltk.corpus import stopwords
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import preprocess_documents
from gensim.utils import simple_preprocess
from gensim.test.utils import get_tmpfile
from gensim.models import Phrases
from gensim.models import Word2Vec



# python array of files by pycruft, stackoverflow
# this function concats the stock data from different sources within a path directory
def gather_stocks_from_dir(path):
    dataframe_stocks_complete = pd.DataFrame()

    # gathers a list of names of all files in the dir
    stock_files = [f for f in listdir(path) if isfile(join(path, f))]

    for file in stock_files:
        if file.endswith('.p'):
            temp_stocks_dataframe = pd.read_pickle(path + '/' + file)
            dataframe_stocks_complete = dataframe_stocks_complete.append(temp_stocks_dataframe)

    return dataframe_stocks_complete


def gather_data_from_stocks():
    stocks_dataframe = gather_stocks_from_dir('./Stocks')

    # gather the values that will be used for regression
    y_class_stocks = pd.DataFrame([stocks_dataframe['close']]).transpose()

    # gather a list of close such that a second col updown can be created
    close_list = y_class_stocks['close']

    close_last = close_list[0]
    close_updown = [None]

    for ind in range(1, len(close_list)):
        if close_last <= close_list[ind]:
            close_updown.append('up')
        elif close_last > close_list[ind]:
            close_updown.append('down')
        close_last = close_list[ind]

    # add classification list to set of stocks
    # ERROR - with all stocks that are in dataframe there exists an index where there is a symbol change
    #   This will make a single wrong value where the change of symbol is
    y_class_stocks['updown'] = close_updown

    return y_class_stocks.reset_index()


# gather data from the sqlite3 database into a list for easier usage
def gather_news_content(database_path):
    connect = sqlite3.connect(database_path)
    cursor = connect.cursor()

    cursor.execute('''
    SELECT query, title, description, content, published_at FROM documents;
    ''')
    news_data = cursor.fetchall()
    # data is returned in a list of tuples
    # [(title, description, content, published_at)]

    # replacer will take away the special chars, and end of document [+number chars]
    news_data = replacer(news_data)

    # close connection and cursor
    cursor.close()
    connect.close()

    return news_data


def replacer(list_data):
    # for each piece of content within the list_data
    query_list = []
    dates_list = []
    content_list = []

    for content in list_data:
        content = list(content)

        # remove query from data so that all of the
        query = content[0]
        del content[0]

        # remove the date from each of the pieces of content
        date = content[-1]
        del content[-1]

        # for each piece of information within the content of each document
        temp_collect = []
        temp_last = ''
        for index in range(len(content)):
            if content[index] is not None:
                # replaces certain parts of the content of the data for better readability
                content[index] = re.sub(r'\s+', ' ', content[index])
                content[index] = re.sub(r'<[/]*\w>', '', content[index])
                content[index] = re.sub(r'[?]|[!]|[,]', '', content[index])
                end_string_list = content[index].split('[')

                # checks to see if the description is the same as the content of the file.
                temp_collect.append(end_string_list[0].lower())

        # remove excess date information
        if len(date) >= 10:
            date = date[0:10]

        query_list.append(query)
        dates_list.append(date)
        content_list.append(' '.join(temp_collect))

    # convert string to datetime
    for index_in_date_list in range(len(dates_list)):
        dates_list[index_in_date_list] = datetime.datetime.strptime(dates_list[index_in_date_list]
                                                                    , '%Y-%m-%d').replace(tzinfo=pytz.utc)

    return query_list, dates_list, content_list


def preprocess_content_for_gensim(content_list):
    for ind in range(len(content_list)):
        temp_store = remove_stopwords(content_list[ind])
        temp_store = temp_store.replace('-', '').split(' ')
        content_list[ind] = temp_store
    return content_list


def tsne_plot(model, word, perplexity, quit_words):

    # Gather the closest words a few times to have some data to look at
    close_words = model.similar_by_word(word)
    close_words_extra = []
    for word_item in close_words:
        close_words_extra.append(model.similar_by_word(word_item[0]))
    close_words_final = []
    for word_item in close_words_extra:
        close_words_final.append(model.similar_by_word(word_item[0]))
    close_words_final.append(close_words)

    X_data = np.empty((0, model.vector_size))

    word_labels = [word]

    # takes from Aneesha Bakharia python notebook
    # https://medium.com/@aneesha/using-tsne-to-plot-a-subset-of-similar-words-from-word2vec-bb8eeaea6229
    X_data = np.append(X_data, np.array([model[word]]), axis=0)
    for word_and_score in close_words_final:
        for words in word_and_score:
            word_vector = model[words[0]]
            word_labels.append(words[0])
            X_data = np.append(X_data, np.array([word_vector]), axis=0)

    tsne = TSNE(n_components=2, n_iter=10000, perplexity=perplexity, n_iter_without_progress=500)
    y = tsne.fit_transform(X_data)

    x_coordinates = []
    y_coordinates = []

    for x in y:
        x_coordinates.append(x[0])
        y_coordinates.append(x[1])

    plt.scatter(x_coordinates, y_coordinates)

    i = 0
    for label, x_co, y_co in zip(word_labels, x_coordinates, y_coordinates):
        plt.annotate(label, xy=(x_co, y_co), xytext=(0, 0), textcoords='offset points')
        if i > quit_words:
            break
        i += 1
    plt.xlim(min(x_coordinates) + 0.5, max(x_coordinates) + 0.5)
    plt.ylim(min(y_coordinates) + 0.5, max(y_coordinates) + 0.5)
    plt.show()


# return both a data after and data before a date
def gather_data_before_and_after(dataframe, date):
    beforedataframe = pd.DataFrame()
    afterdataframe = pd.DataFrame()
    for index, row in dataframe.iterrows():
        if row['date'] < date:
            beforedataframe = beforedataframe.append(row)
        else:
            afterdataframe = afterdataframe.append(row)
        if index % 500 == 0:
            print(index)
    return beforedataframe, afterdataframe


# return dataframe of all rows with date less than the param date
def data_before_date(dataframe, date):
    beforedataframe = pd.DataFrame()
    for index, row in dataframe.iterrows():
        if row['date'] < date:
            beforedataframe = beforedataframe.append(row)
    return beforedataframe, date


# return dataframe of all rows with date more than or equal to the param date
def data_after_date(dataframe, date):
    afterdataframe = pd.DataFrame()
    for index, row in dataframe.iterrows():
        if row['date'] >= date:
            afterdataframe = afterdataframe.append(row)
    return afterdataframe, date


# Splits the dataset for training and testing by the input date.
def tfidf_data_before_date(data_to_be_tfidf, date):
    tfidf_training_set = []
    tfidf_test_set = []
    for index, row in data_to_be_tfidf.iterrows():
        if row['date'] < date:
            tfidf_training_set.append(row['content'])
        else:
            tfidf_test_set.append(row['content'])
    return tfidf_training_set, tfidf_test_set


def create_gensim_word_2_vec_model(content_list):
    gensim_content_list = preprocess_content_for_gensim(content_list)
    bigrams = Phrases(gensim_content_list)
    word_to_vec_model = Word2Vec(bigrams[gensim_content_list], min_count=1, window=3, size=300)
    return word_to_vec_model


def load_gensim_word_2_vec_model(path):
    file = get_tmpfile(path)
    return Word2Vec.load(file)


# training and testing data is assumed to be in a list of values
# df['content'] is the data
def keras_word_embedding(training_data, testing_data, training_class, testing_class,
                         embedding_dimension=None, model_ex=None, updown=True,
                         save_path='Models'):
    # create tokenizer to generate training and testing tokens for later use
    tokens = Tokenizer()
    total_text = training_data + testing_data
    tokens.fit_on_texts(total_text)

    # get the max len of any of the string such that they can be padded with zeros
    max_token_length = max([len(strings.split()) for strings in total_text])

    # num words in the vocab of the corpus
    vocab_size = len(tokens.word_index) + 1

    # convert training and testing strings to tokens
    training_data_tokens = tokens.texts_to_sequences(training_data)
    testing_data_tokens = tokens.texts_to_sequences(testing_data)

    # pads the training and testing data with zeros to make all the same length
    # pads with zeros at the end of the data
    training_data_tokens_pad = pad_sequences(training_data_tokens, maxlen=max_token_length, padding='post')
    testing_data_tokens_pad = pad_sequences(testing_data_tokens, maxlen=max_token_length, padding='post')

    if embedding_dimension is None:
        embedding_dimension = 100
    else:
        embedding_dimension = embedding_dimension

    # if it is a classification of a binary, which updown is
    if updown:

        training_class2 = list()
        testing_class2 = list()

        for t_class in training_class:
            if t_class == 'up':
                training_class2.append(1)
            else:
                training_class2.append(0)

        for t_class in testing_class:
            if t_class == 'up':
                testing_class2.append(1)
            else:
                testing_class2.append(0)
        training_class = np.asarray(training_class2).astype('int8')
        testing_class = np.asarray(testing_class2).astype('int8')

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dimension, input_length=max_token_length))

    if model_ex == 'simple':
        # create a word embedding model
        model.add(GRU(units=100, dropout=0, recurrent_dropout=0))
        model.add(Dense(1, activation='sigmoid'))
        # Learning function for that model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # That it is a 100% accuracy, something broke
    elif model_ex == 'relu':
        # create word embedding model with close
        model.add(GRU(units=100))
        model.add(Dense(units=100, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        # Learning function for that model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif model_ex == 'lstm':
        # create word embedding model with close
        model.add(GRU(units=100))
        model.add(Dense(units=100, activation='lstm'))
        model.add(Dense(1, activation='sigmoid'))
        # Learning function for that model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    callbacks = ModelCheckpoint(save_path,
                                save_best_only=True,
                                verbose=1)

    model.fit(training_data_tokens_pad, training_class, batch_size=64,
              epochs=15, verbose=2, validation_data=(testing_data_tokens_pad, testing_class),
              callbacks=[callbacks])

    return model


def date_and_content_class_gatherer(stocks_data, content_data):
    amzn_updown = []
    amzn_close = []
    amd_updown = []
    amd_close = []
    aapl_updown = []
    aapl_close = []
    jpm_updown = []
    jpm_close = []
    gme_updown = []
    gme_close = []
    googl_updown = []
    googl_close = []
    hpq_updown = []
    hpq_close = []
    lyft_updown = []
    lyft_close = []
    msft_updown = []
    msft_close = []
    ntdoy_updown = []
    ntdoy_close = []
    nvda_updown = []
    nvda_close = []
    sne_updown = []
    sne_close = []
    td_updown = []
    td_close = []
    uber_updown = []
    uber_close = []

    for index, content in content_data.iterrows():
        date = content['date']
        stocks_date = stocks_data.loc[stocks_data['date'] == date]

        # This section of code is for selecting the monday after a weekend if the content was posted on a weekend
        # this also accounts for days that are considered holidays
        # placing the close and updown at the end of the holiday
        # this breaks if the date of content is beyond the date of stocks
        while stocks_date.empty:
            date = date + timedelta(days=1)
            stocks_date = stocks_data.loc[stocks_data['date'] == date]

        for index, stock in stocks_date.iterrows():
            if stock['symbol'] == 'AMZN':
                amzn_updown.append(stock['updown'])
                amzn_close.append(stock['close'])
            elif stock['symbol'] == 'AMD':
                amd_updown.append(stock['updown'])
                amd_close.append(stock['close'])
            elif stock['symbol'] == 'AAPL':
                aapl_updown.append(stock['updown'])
                aapl_close.append(stock['close'])
            elif stock['symbol'] == 'JPM':
                jpm_updown.append(stock['updown'])
                jpm_close.append(stock['close'])
            elif stock['symbol'] == 'GME':
                gme_updown.append(stock['updown'])
                gme_close.append(stock['close'])
            elif stock['symbol'] == 'GOOGL':
                googl_updown.append(stock['updown'])
                googl_close.append(stock['close'])
            elif stock['symbol'] == 'HPQ':
                hpq_updown.append(stock['updown'])
                hpq_close.append(stock['close'])
            elif stock['symbol'] == 'LYFT':
                lyft_updown.append(stock['updown'])
                lyft_close.append(stock['close'])
            elif stock['symbol'] == 'MSFT':
                msft_updown.append(stock['updown'])
                msft_close.append(stock['close'])
            elif stock['symbol'] == 'NTDOY':
                ntdoy_updown.append(stock['updown'])
                ntdoy_close.append(stock['close'])
            elif stock['symbol'] == 'NVDA':
                nvda_updown.append(stock['updown'])
                nvda_close.append(stock['close'])
            elif stock['symbol'] == 'SNE':
                sne_updown.append(stock['updown'])
                sne_close.append(stock['close'])
            elif stock['symbol'] == 'TD':
                td_updown.append(stock['updown'])
                td_close.append(stock['close'])
            elif stock['symbol'] == 'UBER':
                uber_updown.append(stock['updown'])
                uber_close.append(stock['close'])

    content_data['AMZN_updown'] = amzn_updown
    content_data['AMZN_close'] = amzn_close
    content_data['AMD_updown'] = amd_updown
    content_data['AMD_close'] = amd_close
    content_data['APPL_updown'] = aapl_updown
    content_data['APPL_close'] = aapl_close
    content_data['JPM_updown'] = jpm_updown
    content_data['JPM_close'] = jpm_close
    content_data['GME_updown'] = gme_updown
    content_data['GME_close'] = gme_close
    content_data['GOOGL_updown'] = googl_updown
    content_data['GOOGL_close'] = googl_close
    content_data['HPQ_updown'] = hpq_updown
    content_data['HPQ_close'] = hpq_close
    content_data['LYFT_updown'] = lyft_updown
    content_data['LYFT_close'] = lyft_close
    content_data['MSFT_updown'] = msft_updown
    content_data['MSFT_close'] = msft_close
    content_data['NTDOY_updown'] = ntdoy_updown
    content_data['NTDOY_close'] = ntdoy_close
    content_data['NVDA_updown'] = nvda_updown
    content_data['NVDA_close'] = nvda_close
    content_data['SNE_updown'] = sne_updown
    content_data['SNE_close'] = sne_close
    content_data['TD_updown'] = td_updown
    content_data['TD_close'] = td_close
    content_data['UBER_updown'] = uber_updown
    content_data['UBER_close'] = uber_close

    return content_data


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # news data is gathered in the format [query, content, date_published],
    # where content is title, description, content.
    # There exist data which the query is None, this data was collected with the use of an old version of searchthenews
    # This can be used for another Y_test set for determining which class of news it was pulled from
    #query_list, dates_list, content_list = gather_news_content('news.db')
    #content_dataframe = pd.DataFrame([query_list, dates_list, content_list]).transpose()
    #content_dataframe.columns = ['query', 'date', 'content']

    # Stocks information
    #stocks = gather_data_from_stocks()

    # gather all of the information into a single dataframe such that gathering training and testing sets becomes easier
    # This increases the size of time file but that is a fair tradeoff that i am willing to make
    # The dataframe is then saved such that the preprocessing
    # of the content and stocks information only happens a single time
    #total_data = date_and_content_class_gatherer(stocks, content_dataframe)
    #total_data.to_pickle('total_data.p')

    # all lines above this can be commented out if the total_data.p file exists
    #total_data = pd.read_pickle('total_data.p')
    #working_date = datetime.datetime.strptime('2019-11-11', '%Y-%m-%d').replace(tzinfo=pytz.UTC)

    #print('Dataframe split')
    #total_before, total_after = gather_data_before_and_after(total_data, working_date)
    #total_before.to_pickle('total_before.p')
    #total_after.to_pickle('total_after.p')

    total_before = pd.read_pickle('total_before.p')
    total_after = pd.read_pickle('total_after.p')

    print('NN Training')

    model = keras_word_embedding(total_before['content'].tolist(), total_after['content'].tolist(),
                                 np.asarray(total_before['AMZN_updown'].tolist()),
                                 np.asarray(total_after['AMZN_updown'].tolist()),
                                 embedding_dimension=100, updown=True, model_ex='simple', save_path='Simple')

    # create a tfidf of the content_list
    # tfidf_training_set, tfidf_test_set = tfidf_data_before_date(content_dataframe, datetime.datetime(2019, 11, 1))

    # tfidf_vector = TfidfVectorizer()
    # tfidf_vector.fit(content_list)
    # tfidf_content = tfidf_vector.transform(content_list)

    # TSNE PLOT OF WORD2VEC similar words
    # word_to_vec_model = create_gensim_word_2_vec_model(content_list)
    # word_to_vec_model = load_gensim_word_2_vec_model('content_word2vec.p')
    # tsne_plot(word_to_vec_model, 'sony', 50, 20)


