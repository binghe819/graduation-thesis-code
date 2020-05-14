import pandas as pd
import gensim

#  http://doc.mindscale.kr/km/unstructured/11.html
#  https://devtimes.com/bigdata/2019/07/19/text-classification-word2vec/

# raw_data = pd.read_csv("语料/processed_dataset_title_summary.csv")
#
# data = raw_data['title_summary']
#
# featuring_data = []
#
# for sentence in data:
#     sentence = str(sentence)
#     featuring_data.append(sentence.split())
#
# # hyperparameter
# num_features = 300
# min_word_count = 5
# num_workers = 10
# window = 4
# downsampling = 1e-3
#
# # logging
# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
#    level=logging.INFO)
#
# print("Training Word2vec Model..")
#
# model = gensim.models.Word2Vec(featuring_data,
#                                size=num_features,
#                                workers=num_workers,
#                                min_count=min_word_count,
#                                window=window,
#                                sample=downsampling,
#                                iter=50,
#                                sg=1)
#
# model.save("word2vec_f300_w4_m5_i50_skipgram(title_summary)")

word2vec_model = gensim.models.Word2Vec.load('word2vec_f300_w4_m5_i50_skipgram(title_summary)')
#
# print(word2vec_model.wv.word_vec("信息"))
print(word2vec_model.wv.most_similar("Information"))

# word2vec_test_model = gensim.models.Word2Vec.load('word2vec_f300_w10_m3_i50')
# print(word2vec_test_model.wv.most_similar("游戏"))