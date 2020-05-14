from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import gensim
import matplotlib.pyplot as plt
import seaborn as sn

## Data Pipeline
raw_data = pd.read_csv("语料/processed_dataset_title_summary.csv")

data = []
for sentence in raw_data['title_summary']:
    sentence = str(sentence)
    data.append(sentence.split())
target = raw_data['target'].replace(to_replace=['B','F','I','J','K','T'],value=[0,1,2,3,4,5])

# split train data and test data
def split_data(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target,test_size=0.2,random_state=123)
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = split_data(data, target)


# target data convert to one - hot
def processing_target(train_target, test_target, num_classes):
    train_y = tf.one_hot(train_target, depth = num_classes)
    test_y = tf.one_hot(test_target, depth = num_classes)
    return train_y, test_y

num_classes = 6
Y_train, Y_test = processing_target(y_train, y_test, num_classes)

# load word2vec_model
word2vec_model = gensim.models.Word2Vec.load('word2vec_f300_w4_m5_i50_skipgram(title_summary)')

## Set hyperparameter
max_input_length = 100
num_words, emb_dim = word2vec_model.wv.vectors.shape # num_words : 임베딩된 단어 수, emb_dim : 임베딩 벡터사이즈
batch_size = 64

def word_to_idx(sentences):
    idx_senteces = []
    for sentence in sentences:
        tmp = []
        for word in sentence:
            if word in word2vec_model.wv.vocab:
                tmp.append(word2vec_model.wv.vocab[word].index)
        idx_senteces.append(tmp)
    return idx_senteces

X_test = tf.keras.preprocessing.sequence.pad_sequences(word_to_idx(x_test), maxlen=max_input_length, padding='post')
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).shuffle(buffer_size=100000).batch(batch_size)

# Model
model = tf.keras.models.load_model('model(title_summary)/title_summary_modeltf.Tensor(0.025359817, shape=(), dtype=float32).h5')
model.summary()

## Evaluate model accuracy
def evaluate(model, texts, labels, confusion_mat):
    hypothesis = model(texts, training=False)
    # print(tf.argmax(hypothesis,axis=1)) # hypothesis
    # print(tf.argmax(labels,axis=1)) # labels
    confusion = tf.math.confusion_matrix(tf.argmax(labels,axis=1),tf.argmax(hypothesis, axis=1),num_classes=6).numpy()
    confusion_mat+=confusion
    correct_prediction = tf.equal(tf.argmax(hypothesis,axis=1), tf.argmax(labels,axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
    return accuracy

# Test
avg_accuracy = 0
step = 0
confusion_mat = np.zeros((6,6),dtype=int)
for batch_images, batch_labels in test_dataset:
    accuracy = evaluate(model, batch_images, batch_labels, confusion_mat)
    avg_accuracy+=accuracy
    step+=1
    print("accuracy :",accuracy)

print("Average accuracy :",avg_accuracy/step)

# print(confusion_mat)

# confusion_mat plot
total = np.sum(confusion_mat, axis = 1)
array = confusion_mat / total[:,None]

print(array)



df_cm = pd.DataFrame(array, index = [i for i in "BFIJKT"],
                  columns = [i for i in "BFIJKT"])
# plt.figure(figsize = (7,7))
plt.title('confusion matrix')
sn.heatmap(df_cm, annot=True)
plt.xlabel('predict')
plt.ylabel('true')
plt.xlim(-0.5, 6)
plt.ylim(6, -0.5)
plt.show()
