from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import gensim
import os

## Text-CNN

## Data Pipeline
raw_data = pd.read_csv("语料/processed_dataset.csv")

data = []
for sentence in raw_data['title']:
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
word2vec_model = gensim.models.Word2Vec.load('word2vec_f300_w4_m5_i50_skipgram(title)')

## Set hyperparameter
max_input_length = 15
# num_words, emb_dim = word2vec_model.wv.vectors.shape # num_words : 임베딩된 단어 수, emb_dim : 임베딩 벡터사이즈
lr = 0.0001
batch_size = 64
epochs = 20
emb_dim = 300

def word_to_idx(sentences):
    idx_senteces = []
    for sentence in sentences:
        tmp = []
        for word in sentence:
            if word in word2vec_model.wv.vocab:
                tmp.append(word2vec_model.wv.vocab[word].index)
        idx_senteces.append(tmp)
    return idx_senteces
    # for sentence in sentences:
    #     length = range(len(sentence))
    #     for i in length:
    #         if sentence[i] in word2vec_model.wv.vocab:
    #             sentence[i] = word2vec_model.wv.vocab[sentence[i]].index
    # return sentences


# print(word_to_idx(x_train))

X_train = tf.keras.preprocessing.sequence.pad_sequences(word_to_idx(x_train), maxlen=max_input_length, padding='post')
X_test = tf.keras.preprocessing.sequence.pad_sequences(word_to_idx(x_test), maxlen=max_input_length, padding='post')

# # dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(buffer_size=100000).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).shuffle(buffer_size=100000).batch(batch_size)

inputs = tf.keras.Input(shape=(max_input_length,), name='input')
embed_initial = tf.keras.initializers.RandomUniform(minval=-1, maxval=1)
embed = tf.keras.layers.Embedding(max_input_length, emb_dim,
                                   embeddings_initializer=embed_initial,
                                   input_length=feature_size,
                                   name='embedding')(inputs)
# embedding = tf.keras.layers.Embedding(input_dim=num_words,output_dim=emb_dim,trainable=False, weights=[word2vec_model.wv.vectors],mask_zero=True ,input_length=max_input_length)(inputs)
# embedding_reshape = tf.keras.layers.Reshape((max_input_length, emb_dim, 1), name='add_channel')(embedding)

pool_outputs = []

filter_shape = 3
conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(filter_shape, emb_dim), strides=(1,1),padding="valid",activation=tf.nn.relu,kernel_initializer='glorot_normal', bias_initializer=tf.keras.initializers.constant(0.1) ,input_shape=(max_input_length,emb_dim,1))(embedding_reshape)
pool3 = tf.keras.layers.MaxPool2D(pool_size=(max_input_length-filter_shape+1,1), strides=(1,1), padding="valid")(conv3)
pool_outputs.append(pool3)

filter_shape = 4
conv4 = tf.keras.layers.Conv2D(filters=128, kernel_size=(filter_shape, emb_dim), strides=(1,1),padding="valid",activation=tf.nn.relu,kernel_initializer='glorot_normal', bias_initializer=tf.keras.initializers.constant(0.1) ,input_shape=(max_input_length,emb_dim,1))(embedding_reshape)
pool4 = tf.keras.layers.MaxPool2D(pool_size=(max_input_length-filter_shape+1,1), strides=(1,1), padding="valid")(conv4)
pool_outputs.append(pool4)

filter_shape = 5
conv5 = tf.keras.layers.Conv2D(filters=128, kernel_size=(filter_shape, emb_dim), strides=(1,1),padding="valid",activation=tf.nn.relu,kernel_initializer='glorot_normal', bias_initializer=tf.keras.initializers.constant(0.1) ,input_shape=(max_input_length,emb_dim,1))(embedding_reshape)
pool5 = tf.keras.layers.MaxPool2D(pool_size=(max_input_length-filter_shape+1,1), strides=(1,1), padding="valid")(conv5)
pool_outputs.append(pool5)

pool_outputs = tf.keras.layers.concatenate(pool_outputs, axis=-1, name='concatenate')
pool_outputs = tf.keras.layers.Flatten(data_format='channels_last', name='flatten')(pool_outputs)
pool_outputs = tf.keras.layers.Dropout(0.5, name='dropout')(pool_outputs)
outputs = tf.keras.layers.Dense(units=num_classes ,kernel_initializer='glorot_normal', bias_initializer=tf.keras.initializers.constant(0.1),kernel_regularizer=tf.keras.regularizers.l2(0.01),bias_regularizer=tf.keras.regularizers.l2(0.01))(pool_outputs)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary()

## loss function
def loss_fn(model, texts, labels):
    hypothesis = model(texts, training=True)  # training으로 인해서 실제 training때처럼 모델을 지나오게 됨 (dropout이 실행됨)
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=labels, y_pred=hypothesis, from_logits=True))  # from_logits : https://www.tensorflow.org/api_docs/python/tf/keras/losses/categorical_crossentropy ( softmax를 통과했냐 안했냐인가? )
    return loss

## Gradient Descent
def gradient(model, texts, labels):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, texts, labels)
    return tape.gradient(loss, sources=model.variables)

## Evaluate model accuracy
def evaluate(model, texts, labels):
    hypothesis = model(texts, training=False)
    correct_prediction = tf.equal(tf.argmax(hypothesis,axis=1), tf.argmax(labels,axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
    return accuracy

## optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

## Train function
def train(model, texts, labels):
    grads = gradient(model, texts, labels)
    return optimizer.apply_gradients(zip(grads, model.variables))

## Train
print("Train Started")
loss_backup = []
for epoch in range(epochs):
    avg_loss = 0.
    avg_train_accuracy = 0.
    avg_test_accuracy = 0.
    train_step = 0
    test_step = 0

    for batch_texts, batch_labels in train_dataset:
        train(model, batch_texts, batch_labels)
        loss = loss_fn(model, batch_texts, batch_labels)
        accuracy = evaluate(model, batch_texts, batch_labels)
        avg_loss += loss
        avg_train_accuracy += accuracy
        train_step += 1;
        print("Epoch : {}, Step : {}, loss : {:.4f}, accuracy : {:.4f}".format(epoch, train_step, loss, accuracy))
        if loss < 0.023:
            model.save("model(title)final/title_model" + str(loss) + ".h5")
        # if(train_step % 100 == 0):
        #     checkpoint.save(file_prefix=checkpoint_prefix)

    avg_loss /= train_step
    avg_train_accuracy /= train_step

    loss_backup.append(avg_loss)

    # Test
    for batch_images, batch_labels in test_dataset:
        accuracy = evaluate(model, batch_images, batch_labels)
        avg_test_accuracy += accuracy
        test_step += 1

    avg_test_accuracy /= test_step

    print("Epoch : {}, loss = {:.8f}, train accuracy = {:.4f}, test accuracy = {:.4f}".format(epoch, avg_loss,
                                                                                              avg_train_accuracy,
                                                                                              avg_test_accuracy))
    # checkpoint.save(file_prefix=checkpoint_prefix)
model.save("first_test_model.h5")
print("Train Finished")

loss_values = pd.DataFrame({'loss':loss_backup})
loss_values.to_csv("loss_values.csv")