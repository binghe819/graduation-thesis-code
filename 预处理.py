from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import jieba
import re

# 分词랑 去除停用词하는 코드
raw_data = pd.read_csv("/Users/kimbyeonghwa/Desktop/语料/zhexue_zongjiao.csv")

def Tokenizing_and_stop_word(dataset):
    result = []
    stopwords = open("stop_word.txt").read().split()
    for sentence in dataset:
        tmp = []
        sentence = " ".join(sentence.split())
        sentence = sentence.strip()\
            .replace(u'\u3000', u'')\
            .replace(u'\n', u'')\
            .replace(u'\r',u'')
        fenci_sentence = jieba.lcut(sentence, cut_all=False)
        fenci_sentence = [x for x in fenci_sentence if x != " "]
        for word in fenci_sentence:
            if word not in stopwords:
                tmp.append(word)
        result.append(" ".join(tmp))
    return result

title = Tokenizing_and_stop_word(raw_data['title'])
summary = Tokenizing_and_stop_word(raw_data['summary'])

# print(summary)

new_data = pd.DataFrame({'title':title, 'summary':summary})
new_data.to_csv('processed_zhexue_zongjiao.csv')


# data = Tokenizing_and_stop_word(raw_data['data'])
#
# data = pd.DataFrame({'data':data, 'target':raw_data['target']})
# data.to_csv('processed_news_data_14000.csv',index=False,sep=',')


# # title랑 summary 붙이기
#
# raw_data = pd.read_csv('语料/processed_dataset.csv')
#
# title_summary = []
#
# for i in range(raw_data.shape[0]):
#     title_summary.append(str(raw_data['title'][i])+" "+str(raw_data['summary'][i]))
#
# new_data = pd.DataFrame({'title_summary':title_summary,'target':raw_data['target']})
#
# new_data.to_csv('processed_dataset_title_summary.csv')