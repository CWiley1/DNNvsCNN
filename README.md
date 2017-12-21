# DNNvsCNN
Deep Neural Networks VS Convolutional Neural Networks on MNIST dataset

## Goal:
To show the great performance improvement when using a convolutional neural network over
 the more traditional densely connected neural network using the MNIST dataset.

## MNIST_DNN:
MNIST_DNN is a densely connected neural network written using the tensorflow python library. 
The goal of this network is to correctly classify 28x28 images of handwritten digits. This 
is a fairly simple implementation using weights as well as bias terms initialized to zero.
This model was able to achive an accuracy of 91.82 on the testing data.

## MNIST_CNN:
MNIST_CNN is a convolutional neural network also written using the tensorflow module for completion 
of the same task. CNN is a more complex implementation as it uses a collection of CNN functions that I wrote
myself. 
This model was able to achive an accuracy of 99.56 on the testing data.
