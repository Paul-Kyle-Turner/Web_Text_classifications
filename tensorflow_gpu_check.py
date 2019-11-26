from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

if __name__ == '__main__':

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
