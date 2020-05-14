from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


raw_data = pd.read_csv('语料/processed_dataset.csv')

data = raw_data['title']
target = raw_data['target'].replace(to_replace=['B','F','I','J','K','T'],value=[0,1,2,3,4,5])

# split train data and test data
def processing_dataset(data, target):
    tmp_data = []
    for sentence in data:
        tmp_data.append(str(sentence))
    x_train, x_test, y_train, y_test = train_test_split(tmp_data, target,test_size=0.2,random_state=123)
    return x_train, x_test, y_train, y_test

def processing_target(train_target, test_target, num_classes):
    train_y = tf.one_hot(train_target, depth = num_classes)
    test_y = tf.one_hot(test_target, depth = num_classes)
    return train_y, test_y

x_train, x_test, y_train, y_test = processing_dataset(data,target)


num_classes = 6

vect = TfidfVectorizer(min_df=5)
vect.fit(x_train)
# pickle.dump(vect.vocabulary_,open("tfidf_feature.pkl","wb"))
X_train = vect.transform(x_train)
X_test = vect.transform(x_test)

num_features = X_train.shape[1]

# SVM Model
SVM = SVC(C=1.0, kernel='linear', gamma='auto',verbose=True)
SVM.fit(X_train, y_train)

# predict the labels on validation dataset
predictions_SVM = SVM.predict(X_test)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, y_test)*100)
print("SVM Confusion Matrics -> ", confusion_matrix(y_test, predictions_SVM))
