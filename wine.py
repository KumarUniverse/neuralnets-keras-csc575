
import matplotlib.pyplot as plt
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from numpy import loadtxt
import numpy as np


### INSERT ALL OF YOUR CODE HERE
def train_model():
	# load the training dataset
	dataset = loadtxt('wine_training.csv', delimiter=',')

	# split into input (X) and output (y) variables
	X_train = dataset[:, 1:]
	y_train = dataset[:, 0]
	y_train = to_categorical(y_train)
	num_attrs = X_train.shape[1]
	num_classes = y_train.shape[1]
	# Verify there are 13 attributes in the input and 3 classes in the output.
	# print("Num Input Attributes:", num_attrs)
	# print("Num Classes:", num_classes)

	# define the keras model
	model = Sequential()
	model.add(Dense(12, input_dim=num_attrs, activation='relu'))
	model.add(Dense(12, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))

	# compile the keras model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	# fit the keras model on the dataset
	history = model.fit(X_train, y_train, epochs=500, batch_size=30, verbose=1)

	return model


### DON'T MODIFY ANY CODE HERE
def eval_model(model):

	# load the testing dataset
	dataset = loadtxt('wine_test.csv', delimiter=',')
	print(type(dataset))

	# split into input (X) and output (y) variables
	X = dataset[:, 1:]
	y = dataset[:, 0]
	y = to_categorical(y)

	# PREDICTIONS & EVALUATION

	# Final evaluation of the model
	scores = model.evaluate(X, y, verbose=0)
	print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))
	# predict first 4 images in the test set
	print("First 4 predictions:", model.predict(X[:4]))
	# actual results for first 4 images in test set
	print("First 4 actual classes:", y[:4])

	# make class predictions with the model
	predictions = model.predict_classes(X)

	# summarize the first 5 cases
	for i in range(5):
		print('%s => %d ' % (X[i].tolist(), predictions[i]), '(expected ', y[i], ')')



if __name__ == '__main__':
	model = train_model()
	eval_model(model)
