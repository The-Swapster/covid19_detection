# importing the required datasets
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
import os 
import tensorflow as tf 
from tensorflow import keras 
from keras.models import Sequential 
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPool2D 
from tensorflow.keras import layers 
from keras.utils import to_categorical 
from keras.layers import LeakyReLU
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# declaring the required variables
label = []
class_name=[]
d = 'D:/COVID19/2A_images'
train_d = 'D:/COVID19/train_COVIDx_CT-2A.txt'
test_d = 'D:/COVID19/test_COVIDx_CT-2A.txt'
val_d = 'D:/COVID19/val_COVIDx_CT-2A.txt'
class_names = ('Normal', 'Pneumonia', 'COVID-19')

# function for reading the label of images
def load_labels(label_file):
    fnames, classes, bboxes = [], [], []
    with open(label_file, 'r') as f:
        for line in f.readlines():
            fname, cls, xmin, ymin, xmax, ymax = line.strip('\n').split()
            fnames.append(fname)
            classes.append(int(cls))
            bboxes.append((int(xmin), int(ymin), int(xmin), int(ymax)))
    return fnames, classes, bboxes

# reading the image labels
fnames_val, classes_val, bboxes_val = load_labels(val_d)
fnames_test, classes_test, bboxes_test = load_labels(test_d)
fnames_train, classes_train, bboxes_train = load_labels(train_d)

# defination for loading the image for training, testing, and validation
def load_and_preprocess(image_file):
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (32, 32))
    image = image.astype(np.float32)/255.0
    return image
    
fnames_val_file = []
for i in range(len(fnames_val)):
    f = os.path.join(d, fnames_val[i])
    image = load_and_preprocess(f)
    fnames_val_file.append(image)
    
fnames_test_file = []
for i in range(len(fnames_test)):
    f = os.path.join(d, fnames_test[i])
    image = load_and_preprocess(f, bboxes_test[i])
    fnames_test_file.append(image)
    
fnames_train_file = []
for i in range(len(fnames_train)):
    f = os.path.join(d, fnames_train[i])
    image = load_and_preprocess(f, bboxes_train[i])
    fnames_train_file.append(image)
    
# convering to numpy array
fnames_val_file = np.array(fnames_val_file)
fnames_test_file = np.array(fnames_test_file)
fnames_train_file = np.array(fnames_train_file)

fnames_val_file.shape
fnames_test_file.shape
fnames_train_file.shape

#looking at the first image
index = 0
print(fnames_val_file[index])
#looking as an image
img = plt.imshow(fnames_val_file[index])
#printing the label of the image
print('The image label is: ', classes_val[index])
#print the image class
print('The image class is: ', class_names[classes_val[index]])
