from __future__ import absolute_import, division, print_function, unicode_literals
import gensim
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn

# raw_data = pd.read_csv('语料/processed_dataset.csv')

# def Tokenizer(raw_data, padding_size):
#     x_text, y = raw_data['title'].tolist(), raw_data['target'].tolist()
#
#     x = []
#     for sentence in x_text:
#         x.append(str(sentence))
#
#     # Tokenizer
#     text_preprocessor = tf.keras.preprocessing.text.Tokenizer()
#     text_preprocessor.fit_on_texts(x)
#     x = text_preprocessor.texts_to_sequences(x)
#     word_dict = text_preprocessor.word_index

# title_loss = [1.9403515,0.7153949,0.5231651,0.46700397,0.42678812,0.39176574,0.36315435,0.33613452,0.31365982,0.29334834,0.27331147,0.25754505,0.2420983,0.24136634,0.23797826
#               ,0.23244196,0.22302864,0.21275958,0.19530908,0.17974681]
#
# title_summary_loss = [1.7508756,0.53678894,0.45307535,0.39249998,0.35932344,0.30986175,0.24903852,0.21220294,0.19646785,0.1719655,0.1510769,0.11643,0.112765804,0.085359655,0.07388549,0.06299865,0.05694753
#                       ,0.038489304,0.034509994,0.025359817]
#
# epoch = np.arange(0,20)
#
# plt.xlim(-1,21)
# plt.plot(epoch, title_loss,'g',label='title')
# plt.plot(epoch, title_summary_loss, 'r', label='title+summary')
# plt.xticks(np.arange(0,20))
# plt.xlabel('epoch')
# plt.ylabel('avg_loss')
# plt.title('loss value')
# plt.legend(loc='upper right')
# plt.show()

# title_accuracy = [0.2445344,0.349433,0.5445223,0.573432,0.683094,0.774504,0.79405,0.814032,0.824314,0.834312,0.83345,0.833221,0.83432,0.826443,0.82405,0.83331,0.840843,0.8407443,0.840531,0.840091]
#
# title_summary_acc = [0.3021596, 0.862179, 0.86131656,0.88426006,0.88344765,0.88542365,0.8867093, 0.8908376, 0.8906272, 0.89071935, 0.88794583, 0.88848275, 0.8906966, 0.8966542, 0.8908843, 0.8934843, 0.8932843, 0.8957543, 0.8958643, 0.8968843]
#
# epoch = np.arange(0,20)
# plt.plot(epoch, title_accuracy, 'g', label='title')
# plt.plot(epoch, title_summary_acc, 'r', label='title+summary')
# plt.xticks(np.arange(0,20))
# plt.xlabel('epoch')
# plt.ylabel('avg_acc')
# plt.title('accuracy')
# plt.legend(loc='upper right')
# plt.show()

confusion = np.array([[0.84559387, 0.0164751, 0.05440613, 0.01455939,0.06168582,0.00727969],
 [0.00761636,0.92440056,0.00789845,0.00338505,0.0203103, 0.03638928],
 [0.02592152,0.00713436,0.87633769,0.03043995,0.05731272,0.00285375],
 [0.00787815,0.0049895, 0.03256303,0.91596639,0.02232143,0.01628151],
 [0.03275362,0.02434783,0.07971014,0.02811594,0.82666667,0.0084058 ],
 [0.00157853,0.03393844,0.00499868,0.01473297,0.0047356, 0.94001579]], dtype=np.float32)


sklearn.metrics.confusion_matrix()