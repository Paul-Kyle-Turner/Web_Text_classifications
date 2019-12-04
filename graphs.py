'''
This creates the bar graphs to compare the percentages
'''
import numpy as np
import matplotlib.pyplot as plt

models = ('bnb', 'mnb', 'rf', 'linr', 'logr', 'knn', 'sc')
y_pos = np.arange(len(models))
performance = [0.6078831587064, 0, 0, 0, 0, 0, 0] # Just hardcoded the percentages in this array

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, models)
plt.ylabel('Percentage')
plt.title('Tested models')

plt.show()