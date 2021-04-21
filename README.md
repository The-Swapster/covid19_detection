# covid19_detection

**Conclusion**
* We read the dataset and resized it to 32x32x3
* This datset was then used to train a custom Convolutional Neural Network and VGG16
* With custom convolutional neural network we got the avg. training accuracy as 98.93% and avg. training loss as 0.0298, avg. validation accuracy as 96.03%, and the testing accuracy was achived as 99.36%
* With VGG16 we got the avg. training accuracy as 98.41% and avg. training loss as 0.0432, avg. validation accuracy as 99.09%, and the testing accuracy was achived as 99.42%
* On unseen dataset which had COVID 19 CT scan images, our custom convolutional neural network performed better than VGG16. Our model gave an accuracy of 78.68% and VGG16 gave an accuracy of 46.35%

**Future Scope**
* Finding out the reasons for the change in performance of VGG16
* Using other pretrained convolutional neural networks like ResNet, Xception, Inception, and LeNet
* Trying machine learning models on the dataset, after extracting features from the images
