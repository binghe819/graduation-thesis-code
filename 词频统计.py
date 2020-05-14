from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import jieba
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# raw_data = pd.read_csv('语料/processed_dataset_title_summary.csv')
#
# # print(len(raw_data['title_summary'][3].split(" ")))
#
# print(len(raw_data['title_summary'][15053].split(" ")))

# -*- coding:utf-8 -*-
def read_csv(file):
    try:
        df = pd.read_csv(file)
        return df
    except:
        print('read csv error!')


if __name__ == '__main__':
    file = '语料/processed_dataset.csv'
    df = read_csv(file)
    cnt = Counter()

    seg_sentence = ""
    for sentence in df['title']:
        seg_sentence = seg_sentence + " " + str(sentence)

    # sentence_total = "".join(str(seg_sentence))

    for word in seg_sentence.split(" "):
        cnt[word]+=1

    result = ""
    for word, count in cnt.items():
        if(count <= 5):
            for i in range(count):
                result = result + " " + word

    my_wordclud = WordCloud(font_path="msyh.ttf", background_color='white', max_words=3000, width=1200,
                            height=800).generate(result)
    plt.imshow(my_wordclud)
    plt.axis("off")
    plt.show()
    my_wordclud.to_file('diyu5.png')
    # print(result)

    # print(sorted(cnt.items(), key=lambda i: i[1]))

    # for word in seg_sentence:
    #     cnt[word]+=1
    # print(sorted(cnt.items(), key=lambda i: i[1]))


    # cnt = Counter()
    # lrc_total = []
    # for lrc in df.lyrics:
    #     lrc_total.append(lrc)
    # lrc_total = "".join(lrc_total)
    # seg_list = jieba.lcut(lrc_total, cut_all=False)
    # for word in seg_list:
    #     cnt[word] += 1
    # c = cnt.most_common()
    # print(c)

