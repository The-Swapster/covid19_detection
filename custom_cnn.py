# loading the libraries
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

# creating the CNN
model = Sequential()
model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (32,32,3)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Flatten())
model.add(Dense(units = 128 , activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 3 , activation = 'softmax'))
model.compile(optimizer = "adam" , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
model.summary()

# one hot encoding the label
y_train_encode = to_categorical(y_train)
y_test_encode = to_categorical(y_test)

# visualising the train and test labels
sns.countplot(y_train)
sns.countplot(y_test)

# training the model
hist = model.fit(x_train, y_train_encode, batch_size = 256, epochs = 20, validation_split=0.1)

# evaluating the model
print("Loss of the model is - " , model.evaluate(x_test,y_test_encode)[0])
print("Accuracy of the model is - " , model.evaluate(x_test,y_test_encode)[1]*100 , "%")

# getting average accuracies and losses
train_acc = hist.history['accuracy']
train_loss = hist.history['loss']
val_acc = hist.history['val_accuracy']
val_loss = hist.history['val_loss']
print('Average training accuracy: ', np.mean(train_acc))
print('Average training loss: ', np.mean(train_loss))
print('Average validation accuracy: ', np.mean(val_acc))
print('Average validation loss: ', np.mean(val_acc))

# plotting the training and testing accuracy and loss
epochs = [i for i in range(20)]
fig , ax = plt.subplots(1,2)
fig.set_size_inches(20,10)

ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
ax[0].plot(epochs , val_acc , 'ro-' , label = 'Validation Accuracy')
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
ax[1].plot(epochs , val_loss , 'r-o' , label = 'Validation Loss')
ax[1].set_title('Testing Accuracy & Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Training & Validation Loss")
plt.show()

# predicting the images
y_pred = model.predict_classes(x_test)
print(y_pred)
rounded_labels=np.argmax(y_test_encode, axis=1)
print(rounded_labels)

# evaluating model
print("Accuracy:",metrics.accuracy_score(rounded_labels, y_pred))
print("Precision:", metrics.precision_score(rounded_labels, y_pred,pos_label='positive', average='weighted'))
print("Recall:", metrics.recall_score(rounded_labels, y_pred,pos_label='positive', average='weighted'))
def specificity_score(y_true, y_pred):
    p, r, f, s = metrics.precision_recall_fscore_support(y_true, y_pred,pos_label='positive', average='weighted')
    return r
print("sensitivity:", metrics.recall_score(rounded_labels, y_pred,pos_label='positive', average='weighted'))
print("specificity:", specificity_score(rounded_labels, y_pred))
print("f1 score:", metrics.f1_score(rounded_labels, y_pred,pos_label='positive', average='weighted'))
print(metrics.classification_report(rounded_labels, y_pred, target_names = class_names))

# creating a heat map
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(metrics.confusion_matrix(rounded_labels, y_pred), annot=True, ax=ax)

# reshape the predicted labels
yr_pred = y_pred.reshape(38985,1)
yr_pred.shape

# getting the roc auc score and plotting the ROC curve
fpr = {}
tpr = {}
thresh ={}
metrics.roc_auc_score(rounded_labels.reshape(38985,1), model.predict_proba(y_test), multi_class='ovr')
for i in range(3):    
    fpr[i], tpr[i], thresh[i] = metrics.roc_curve(rounded_labels, y_pred, pos_label=i)
fig, ax = plt.subplots(figsize=(20,20))
for i in range(3):
  sns.lineplot(fpr[i], tpr[i], linestyle='--', label=f'{class_names[i]} vs Rest', ax=ax)
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')   
