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
from sklearn.model_selection import train_test_split

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

# combining the three datasets
fnames = list(itertools.chain(fnames_train, fnames_test, fnames_val))
classes = list(itertools.chain(classes_train, classes_test, classes_val))

# loading the images
def load_and_preprocess(image_file):
    image = cv2.imread(image_file)
    image = cv2.resize(image, (32, 32))
    image = image.astype(np.float32)/255.0
    return image

x = []
for i in range(len(fnames)):
    f = os.path.join(d, fnames[i])
    image = load_and_preprocess(f)
    x.append(image)
    
# plotting the dataset
sns.countplot(classes)

# reshaping the dataset
x = np.array(x)
x = np.reshape(x, (len(x), 32, 32, 3))

# checking the dataset
print('Shape of samples:', x.shape)
print('Shape of classes:', len(classes))

#looking at the first image
index = 0
print(x[index])
#looking as an image
img = plt.imshow(x[index])
#printing the label of the image
print('The image label is: ', classes[index])
#print the image class
print('The image class is: ', class_names[classes[index]])

# splitting the datasets
x_train, x_test, y_train, y_test = train_test_split(x, classes, test_size=0.2, random_state=42)
