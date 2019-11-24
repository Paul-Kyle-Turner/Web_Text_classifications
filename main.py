import pandas as pd
import numpy as np
import math
import sqlite3
import re


# stock data will remain within a dataframe of col stocks, end of day, end of day increase decrease
# news information will be gathered from the news.db
#                   NEW FUNCTION GATHER CONTENT INFORMATION
#                   NEW FUNCTION COMBINE DESCRIPTION
def gather_data():
    print('working')


def gather_data_from_stocks():
    # Alex please make this function
    # It should take in a pandas dataframe of the stock information and return a dataframe that includes the stock
    #   names the end of date price of the stock, the date, and if the stock increased or decreased
    print('ALEX')


# gather data from the sqlite3 database into a list for easier useage
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

    return news_data


def replacer(list_data):
    # for each piece of content within the list_data
    total_list = []
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
        for index in range(len(content) - 1):
            if content[index] is not None:
                content[index] = re.sub(r'\s+', ' ', content[index])
                content[index] = re.sub(r'<[/]*\w>', '', content[index])
                end_string_list = content[index].split('[')
                if index == 1 and index != 0:
                    temp_last = content[index]
                else:
                    if temp_last != end_string_list[0]:
                        temp_collect.append(end_string_list[0])
        content = [query, ' '.join(temp_collect), date]
        total_list.append(content)
    return total_list


if __name__ == '__main__':
    # news data is gathered in the format [query, content, date_published],
    # where content is title, description, content.
    # There exist data which the query is None, this data was collected with the use of an old version of searchthenews
    # This can be used for another Y_test set for determining which class of news it was pulled from
    news_data = gather_news_content('news.db')

