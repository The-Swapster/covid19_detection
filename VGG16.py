# loading the required libraries
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

# importing VGG16 and then modifying it for better results, adding a dense layer of 1024 and an output layer of 3. The input shape is also chnged to (32,32,3)
model1 = VGG16(include_top=False, input_shape=(32, 32, 3))
# not using top layer and also changing the shape of the input image
flat1=Flatten()(model1.layers[-1].output) # adding a flatten layer
class1=Dense(1024, activation='relu')(flat1) # adding a dense layer with 1024 neurons
output=Dense(3, activation='softmax')(class1) # adding an output layer with 3 neurons
model1=Model(inputs=model1.inputs, outputs=output) # adding the above layers to model_2
model1.summary()

# compiling and fitting the model
model1.compile(optimizer = "adam" , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
hist1 = model1.fit(x_train, y_train_encode, batch_size = 256, epochs = 20, validation_split=0.1)

# calculating the loss and accuracy of the model after testing
print("Loss of the model is - " , model1.evaluate(x_test,y_test_encode)[0])
print("Accuracy of the model is - " , model1.evaluate(x_test,y_test_encode)[1]*100 , "%")

# calculating the average testing and training accuracy
train_acc1 = hist1.history['accuracy']
train_loss1 = hist1.history['loss']
val_acc1 = hist1.history['val_accuracy']
val_loss1 = hist1.history['val_loss']
print('Average training accuracy: ', np.mean(train_acc1))
print('Average training loss: ', np.mean(train_loss1))
print('Average validation accuracy: ', np.mean(val_acc1))
print('Average validation loss: ', np.mean(val_loss1))

# plotting the accuracy and loss for validation and testing
epochs = [i for i in range(20)]
fig , ax = plt.subplots(1,2)
fig.set_size_inches(20,10)

ax[0].plot(epochs , train_acc1 , 'go-' , label = 'Training Accuracy')
ax[0].plot(epochs , val_acc1 , 'ro-' , label = 'Validation Accuracy')
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs , train_loss1 , 'g-o' , label = 'Training Loss')
ax[1].plot(epochs , val_loss1 , 'r-o' , label = 'Validation Loss')
ax[1].set_title('Testing Accuracy & Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Training & Validation Loss")
plt.show()

# saving and loading the model
model1.save('vgg16.h5')
model1 = models.load_model('vgg16.h5')

# predicting the class of the image and then getting the classe number
y_pred1 = model1.predict(x_test)
ry_pred1 = np.argmax(y_pred1,axis=1)
print(ry_pred1)
rounded_labels1=np.argmax(y_test_encode, axis=1)
print(rounded_labels1)

# calculating the accuracy, precision, recall, f1 score, specificity, sensitivity, and classification report
print("Accuracy:",metrics.accuracy_score(rounded_labels1, ry_pred1))
print("Precision:", metrics.precision_score(rounded_labels1, ry_pred1,pos_label='positive', average='weighted'))
print("Recall:", metrics.recall_score(rounded_labels1, ry_pred1,pos_label='positive', average='weighted'))
def specificity_score(y_true, y_pred):
    p, r, f, s = metrics.precision_recall_fscore_support(y_true, y_pred,pos_label='positive', average='weighted')
    return r
print("sensitivity:", metrics.recall_score(rounded_labels1, ry_pred1,pos_label='positive', average='weighted'))
print("specificity:", specificity_score(rounded_labels1, ry_pred1))
print("f1 score:", metrics.f1_score(rounded_labels1, ry_pred1,pos_label='positive', average='weighted'))
print(metrics.classification_report(rounded_labels1, ry_pred1, target_names = class_names))

# plotting the heat map for the model trained
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(metrics.confusion_matrix(rounded_labels1, ry_pred1), annot=True, ax=ax)

# reshaping the predicted class 
ryr_pred1 = ry_pred1.reshape(-1,1)
ryr_pred1.shape

# plotting the ROC curve
fpr1 = {}
tpr1 = {}
thresh1 ={}
for i in range(3):    
    fpr1[i], tpr1[i], thresh1[i] = metrics.roc_curve(rounded_labels1, ry_pred1, pos_label=i)
fig, ax = plt.subplots(figsize=(10,10))
for i in range(3):
  sns.lineplot(fpr1[i], tpr1[i], linestyle='--', label=f'{class_names[i]} vs Rest', ax=ax)
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
#plt.savefig('Multiclass ROC',dpi=300);    
