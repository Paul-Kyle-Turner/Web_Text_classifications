'''
This creates the bar graphs to compare the percentages
'''

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


def updown_to_1_0(testing):
    # if it is a classification of a binary, which updown is
    testing_class2 = list()

    for t_class in testing:
        if t_class == 'up':
            testing_class2.append(1)
        else:
            testing_class2.append(0)

    testing_class = np.asarray(testing_class2).astype('int8')

    return testing_class

'''
models = ('BNB', 'MNB', 'RF', 'LOGR', 'KNN')
stocks = ('AAPL', 'AMD', 'AMZN', 'GME', 'GOOGL', 'HPQ', 'JPM', 'LYFT', 'MSFT', 'NTDOY', 'NVDA', 'SNE', 'TD', 'UBER')

for stock in stocks:
    y_pos = np.arange(len(models))
    performance = []
    print(stock)
    for model in models:
        with open('SKLEARN_MODELS/' + str(stock) + '/' + str(model) + '/' + str(model.lower()) + 'output.txt') as file:
            if model == 'RF':
                performance.append(float(file.readlines()[7]) * 100)
            elif model == 'LOGR':
                performance.append(float(file.readlines()[5]) * 100)
            elif model == 'KNN':
                performance.append(float(file.readlines()[3]) * 100)
            else:
                performance.append(float(file.readlines()[1]) * 100)
    print(performance)

    plt.bar(y_pos, performance, align='center', alpha=1, width=0.3)
    axes = plt.gca()
    axes.set_ylim([0, 100])
    plt.xticks(y_pos, models)
    plt.ylabel('Percent Accuracy')
    plt.title('Tested models for ' + stock)

    plt.show()
    plt.clf()
'''

stocks = ('AAPL', 'AMD', 'AMZN', 'GME', 'GOOGL', 'HPQ', 'JPM', 'LYFT', 'MSFT', 'NTDOY', 'NVDA', 'SNE', 'TD', 'UBER')
models = ('RELU', 'SIMPLE')
performance = []

for modeltype in models:
    y_pos = np.arange(len(stocks))
    for stock in stocks:
        model = tf.keras.models.load_model('NN_STOCKS_UPDOWN_EMBEDDED/' + str(stock) + '/' + str(modeltype))
        #model.summary()
        tokens = Tokenizer()
        total_after = pd.read_pickle('total_after.p')
        testing_data = total_after['content'].tolist()
        total_text = testing_data
        tokens.fit_on_texts(total_text)
        max_token_length = max([len(strings.split()) for strings in total_text])
        testing_data_tokens = tokens.texts_to_sequences(testing_data)
        testing_data_tokens_pad = pad_sequences(testing_data_tokens, maxlen=max_token_length, padding='post')
        testing_class = np.asarray(total_after[stock + '_updown'].tolist())
        testing_class = updown_to_1_0(testing_class)
        loss, acc = model.evaluate(testing_data_tokens_pad,  testing_class, verbose=2)
        performance.append(100*acc)

    plt.figure(figsize=(10, 5))
    plt.bar(y_pos, performance, align='center', alpha=1)
    axes = plt.gca()
    axes.set_ylim([0, 100])
    plt.xticks(y_pos, stocks)
    plt.ylabel('Percent Accuracy')
    plt.title('Tested models for ' + modeltype)
    plt.savefig(str(modeltype) + '.png')
    plt.clf()
