# Neural Networks w/ Keras and Tensorflow

## Introduction

In this assignment, you will be using Keras, 
an open source Python library that provides a 
common wrapper for several popular neural network 
libraries including Theano, TensorFlow and CNTK. 
We will be using TensorFlow as our back end.

## Getting Started

### First Example - Pima Indians

I strongly recommend working through this [tutorial](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/) to get 
started. This tutorial corresponds with the following files in this repo:
* diabetes.py
* pima-indians-diabetes.csv

### Second Example - MNIST

Additionally, look at this following second example in the repo, which is 
loosely based on the following [tutorial](https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5):
* mnist.py

### Comparison between the examples

There is A LOT in tensorflow and keras, so it is easy to get overwhelmed here. Instead of focusing on all of this new code, let's think 
about the similarities between the two examples:

1. they both load in a dataset
    * MNIST additionally splits the dataset into a training and testing portion
1. they both define a "sequential" (feed-forward) neural network model 
   * they both define the number of input units, hidden units, and output units
   * Pima Indians has a boolean target
   * MNIST has a 0-9 classification target, so "softmax" is used
1. they both "compile" the model 
   * Pima Indians uses "binary_crossentropy" because the target attribute has > 2 classes
   * MNIST uses "catagorial_crossentropy" because the target attribute has > 2 classes
1. they both "fit" the model (this is where the neural network tries 
  to learn the optimal weights based on the training data)
1. they both use the learned model to evaluate/predict classes on data 

## Installing Necessary Python Libraries

Download the starter code and dataset for this assignment, 
and configure PyCharm as you have done for the previous 
programming assignments in this course. Make sure your Python 
Interpreter is set to Python 3.8

In addition, you will probably need to install these libraries into your project.
Fortunately, PyCharm makes this very easy:

* Go to (PyCharm Preferences > Python Interpreter). At the bottom of 
the window is a +, for adding a library to the project. Add the following libraries:
  
1. numpy
1. keras
1. tensorflow
1. matplotlib

## Dataset

In this programming assignment, we will be experimenting with 
the "classic" [Wine](https://archive.ics.uci.edu/ml/datasets/Wine) dataset from 
the [UCI Machine Learning Repository](https://archive.ics.uci.edu/). 

The wine dataset has 14 fields. 
The first is the class/target label – this is what you should be trying to 
learn/predict using the other 13 fields. 
The 13 fields represent a chemical analysis of different wines 
from a region in Italy. The wines are from 3 different cultivars 
and these cultivars are what you’re trying to identify.

I have already downloaded the dataset for you and have 
commit it to this starting repo. **Do not modify either 
of these two csv files in any way.**
* wine_training.csv
* wine_testing.csv

## What to code

Your goal is to complete the `train_model()` function 
in the `wine.py` file. This is most likely the only
place where you'll be coding.

After you complete the method, when you run the entire `wine.py` file 
(notice the "main" method at the bottom), two functions are run: 
1. Your `train_model()` function which must return your compiled and fitted Keras *model* object, 
ready for testing. 
1. The `eval_model()` function which uses the learned model against an unseen test set

**Input file:** You must read the input file as `wine_training.csv`. (Remember, don't 
hard code any absolute paths!) 
There's an easy way to read in csv data. Look up how to do it.


## Network structure:

You may use any number of hidden layers of any type 
(though I’d recommend sticking with "Dense") with 
any activation method and with any number of neurons that you wish. 
However, your output layer must be a softmax layer. 
Softmax layers have a neuron for each possible output category and 
the activations of the neurons sum to 1. Each neuron’s 
activation level can be viewed as the probability that the input 
corresponds to that particular class. 

### Exploration:

Experiment! You should obviously play with network 
structures but you should also experiment with 
training and testing. 
The `model.evaluate` method uses the input data and calculates loss and 
accuracy and any other metrics that you compiled. 
This is different than the `model.predict` method which will 
actually take inputs and predict the categories for those inputs. 
You can also experiment with validation sets or you can code 
your own k-fold cross-validation if you are curious about that. 

## Autograding / Testing

Anything that predicts 90% or higher on the held-out 
test set will be awarded full credit for performance in CSC575. (The bar is 80% in CSC481.)

The autotesting code is contained in `neuralnet_tester.py`.

Specifically, points are awarded for the following levels:

|            | CSC481    | CSC575    |
|------------|-----------|-----------|
| >= 60%     | 15 points | 15 points |
| >= 70%     | 15 points | 15 points |
| >= 80%     | 15 points | 15 points |
| >= 90%     |           | 15 points |
| Max Points | 45 / 45   | 60 / 60   |

## Submission Instructions

Submit this assignment by pushing/committing into our GitHub organization, exactly like HW 0. 
Test your project locally 
using pytest before pushing/committing. You may push/commit multiple times. Look at the Actions report on the 
repository to double check the most recent result of the unit tests.

