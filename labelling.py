from __future__ import absolute_import, division, print_function, unicode_literals
import os
import pandas as pd
import csv

raw_data = pd.read_csv("/Users/kimbyeonghwa/Desktop/语料/전처리된파일(테스트1)/processed_zhexue_zongjiao.csv")

target = ["B"]*raw_data.shape[0]

df = pd.DataFrame({'title':raw_data['title'],'summary':raw_data['summary'],'target':target})

df.to_csv("processed_zhexue_zongjiao.csv")
