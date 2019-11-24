import pandas as pd
import numpy as np
import sqlite3


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


def gather_news_content(database_path):
    connect = sqlite3.connect(database_path)
    cursor = connect.cursor()

    cursor.execute('''
    SELECT title, description, content, published_at FROM documents;
    ''')
    news_data = cursor.fetchall()
    print(type(news_data))
    print(news_data)



if __name__ == '__main__':
    gather_news_content('news.db')