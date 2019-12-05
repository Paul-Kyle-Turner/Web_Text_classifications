'''
This creates the bar graphs to compare the percentages
'''
import numpy as np
import matplotlib.pyplot as plt

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

    plt.bar(y_pos, performance, align='center', alpha=1)
    axes = plt.gca()
    axes.set_ylim([0, 100])
    plt.xticks(y_pos, models)
    plt.ylabel('Percent Accuracy')
    plt.title('Tested models for ' + stock)

    plt.savefig(str(stock) + '.png')
    plt.clf()
