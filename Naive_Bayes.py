import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC

# flattening the dataset
x_f = x.flatten().reshape((x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]))
x_f.shape

# splitting the dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(x_f, y, test_size=0.2, random_state=42)

# one hot encoding the categorical dataset
y_train_encode = to_categorical(y_train)
y_test_encode = to_categorical(y_test)

# creating model, training it, testing it, and printing the accuracy
mnb = MultinomialNB()
mnb.fit(x_train, y_train)
y_pred2 = mnb.predict(x_test)
print(mnb.score(x_test, y_test))

# creating model, training it, testing it, and printing the accuracy
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred2 = gnb.predict(x_test)
print(gnb.score(x_test, y_test))

# creating model, training it, testing it, and printing the accuracy
svc = SVC()
svc.fit(x_train, y_train)
y_pred2 = svc.predict(x_test)
print(svc.score(x_test, y_test))
