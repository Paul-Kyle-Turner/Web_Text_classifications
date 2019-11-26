import pandas as pd
import numpy as np
import math
import datetime
import sqlite3
import re
from os import listdir
from os.path import isfile, join
import warnings

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import preprocess_documents
from gensim.utils import simple_preprocess
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

    # add classification list to set of stocks
    # ERROR - with all stocks that are in dataframe there exists an index where there is a symbol change
    #   This will make a single wrong value where the change of symbol is
    y_class_stocks['updown'] = close_updown

    return y_class_stocks


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
                end_string_list = content[index].split('[')

                # checks to see if the description is the same as the content of the file.
                temp_collect.append(end_string_list[0].lower())

        # remove excess date information
        if len(date) >= 20:
            date = date[0:19]

        query_list.append(query)
        dates_list.append(date)
        content_list.append(' '.join(temp_collect))

    # convert string to datetime
    for index_in_date_list in range(len(dates_list)):
        dates_list[index_in_date_list] = datetime.datetime.strptime(dates_list[index_in_date_list], '%Y-%m-%dT%H:%M:%S')

    return query_list, dates_list, content_list


def preprocess_content_for_gensim(content_list):
    for ind in range(len(content_list)):
        temp_store = remove_stopwords(content_list[ind])
        temp_store = temp_store.replace('-', '').split(' ')
        content_list[ind] = temp_store

    return content_list


# return dataframe of all rows with date less then the @param date
def data_before_date(dataframe, date):
    beforedataframe = pd.DataFrame()
    for index, row in dataframe.iterrows():
        if row['date'] < date:
            beforedataframe = beforedataframe.append(row)
    return beforedataframe, date


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


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # news data is gathered in the format [query, content, date_published],
    # where content is title, description, content.
    # There exist data which the query is None, this data was collected with the use of an old version of searchthenews
    # This can be used for another Y_test set for determining which class of news it was pulled from
    query_list, dates_list, content_list = gather_news_content('news.db')

    content_dataframe = pd.DataFrame([query_list, dates_list, content_list]).transpose()
    content_dataframe.columns = ['query', 'date', 'content']

    # Stocks information
    stocks = gather_data_from_stocks()

    # create a tfidf of the content_list
    #tfidf_vector = TfidfVectorizer()
    #tfidf_vector.fit(content_list)
    #tfidf_content = tfidf_vector.transform(content_list)

    (test, yeet) = tfidf_data_before_date(content_dataframe, datetime.datetime(2019, 11, 1))
    print(test)
    print(yeet)
    #print(stocks)

    gensim_content_list = preprocess_content_for_gensim(content_list)
    word_to_vec_model = Word2Vec(gensim_content_list, min_count=1, window=3)






