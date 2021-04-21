# loading the requiered models
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
import os 
import tensorflow as tf 
from tensorflow import keras 
from keras.models import Sequential 
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPool2D, Dropout, BatchNormalization
from tensorflow.keras import layers 
from keras.utils import to_categorical 
from keras.layers import LeakyReLU
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.models import Model

# reading another dataset of CT Scan images
d = '/content/drive/MyDrive/Covid19/COVID'
path = []
data = []
classes = []
for l in os.listdir(d): 
    path.append(os.path.join(d, l))
for img in path:
  img_arr = cv2.imread(img)
  resized_arr = cv2.resize(img_arr, (32, 32)) # Reshaping images to preferred size
  image = resized_arr.astype(np.float32)/255.0
  data.append(image)
  classes.append(2)
  
# converting the datset to numpy array
data = np.array(data)

data.shape # checing the shape of dataset

# rouding off the classes
y_pred2 = np.argmax(model.predict(data), axis=-1)
print(y_pred2)

# printing the results
print("Accuracy:",metrics.accuracy_score(classes, y_pred2))
print("Precision:", metrics.precision_score(classes, y_pred2,pos_label='positive', average='weighted'))
print("Recall:", metrics.recall_score(classes, y_pred2,pos_label='positive', average='weighted'))
def specificity_score(y_true, y_pred):
    p, r, f, s = metrics.precision_recall_fscore_support(y_true, y_pred,pos_label='positive', average='weighted')
    return r
print("sensitivity:", metrics.recall_score(classes, y_pred2,pos_label='positive', average='weighted'))
print("specificity:", specificity_score(classes, y_pred2))
print("f1 score:", metrics.f1_score(classes, y_pred2,pos_label='positive', average='weighted'))
print(metrics.classification_report(classes, classes, target_names = class_names))

# rouding off the classes
y_pred3 = np.argmax(model1.predict(data), axis=-1)
print(y_pred3)

# printing the results
print("Accuracy:",metrics.accuracy_score(classes, y_pred3))
print("Precision:", metrics.precision_score(classes, y_pred3,pos_label='positive', average='weighted'))
print("Recall:", metrics.recall_score(classes, y_pred3,pos_label='positive', average='weighted'))
def specificity_score(y_true, y_pred):
    p, r, f, s = metrics.precision_recall_fscore_support(y_true, y_pred,pos_label='positive', average='weighted')
    return r
print("sensitivity:", metrics.recall_score(classes, y_pred3,pos_label='positive', average='weighted'))
print("specificity:", specificity_score(classes, y_pred3))
print("f1 score:", metrics.f1_score(classes, y_pred3,pos_label='positive', average='weighted'))
print(metrics.classification_report(classes, classes, target_names = class_names))
