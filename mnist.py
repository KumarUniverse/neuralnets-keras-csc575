# https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5
# https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/

from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

# download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# ---------------------------------- #

# EXPLORATORY DATA ANALYSIS: plot the first image in the dataset
plt.imshow(X_train[0])
plt.show()

# the shape of every image in the mnist dataset is 28 x 28
# verify image shape
print("X_train[0].shape:", X_train[0].shape)
print("X_train.shape:", X_train.shape)

# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# ----------------------------------- #

# DATA PRE-PROCESSING
# reshape data: number of images, image width, image height, grayscale (rather than RGB)
#X_train = X_train.reshape(60000,28,28,1)
#X_test = X_test.reshape(10000,28,28,1)

# one-hot encode target column from integer answer to vector of size 10
print("Old:", y_train[0])
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# verify one-hot encoding
print("New one-hot encoding:", y_train[0])
num_classes = y_test.shape[1]
print("Num Classes:", num_classes)

# ----------------------------------- #

# BUILDING THE MODEL
# define the keras model
model = Sequential()
model.add(Dense(num_pixels, input_dim=num_pixels, activation='relu'))
#model.add(Dense(8, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# ----------------------------------- #

# compile the keras model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# ------------------------------------ #

# TRAINING
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

# ------------------------------------ #

# PREDICTIONS & EVALUATION

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
# predict first 4 images in the test set
print("First 4 predictions:", model.predict(X_test[:4]))
# actual results for first 4 images in test set
print("First 4 actual classes:", y_test[:4])

# ------------------------------------- #

# plot learning curves
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()