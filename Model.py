
# importing required packages ..

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

# preparing Train, Validation and Test sets ..

data = []  # to store images of training set
labels = []  # to store labels of images in the training set
classes = 43
os.chdir('C:\\Users\\user\\PycharmProjects\\Traffic_Signs_Recognition')  # changing the current working directory to the given path
cur_path = os.getcwd()  # getting current working directory
for i in range(classes):
    path = os.path.join(cur_path, 'Train', str(i))  # to get the direction of each folder in Train Folder
    print('loading images from ' + path + ' ... ')
    images = os.listdir(path)  # gets 'name.format' for every file in the path direction as a list to images
    for a in images:
        try:
            image = Image.open(path + '\\' + a)
            image = image.resize((30, 30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print('error when loading ' + a + ' image in ' + str(i) + ' class of Training set')
data = np.array(data)
labels = np.array(labels)

print('the shape of Training data array : ', data.shape)  # (39209, 30, 30, 3)
print('the shape of Training labels array : ', labels.shape)  # (39209,)

x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
print('x_train shape : ', x_train.shape)  # (31367, 30, 30, 3) training set
print('x_test shape : ', x_val.shape)  # (7842, 30, 30, 3) validation set
print('y_train shape : ', y_train.shape)  # (31367,) labels for training
print('y_test shape : ', y_val.shape)  # (7842,) labels for validation
y_train = to_categorical(y_train, 43)
y_val = to_categorical(y_val, 43)

test_set = pd.read_csv('Test.csv')
test_labels = test_set['ClassId'].values
path = test_set['Path'].values
test_images = []
for i in path:
    image = Image.open(i)
    image = image.resize((30, 30))
    test_images.append(np.array(image))

test_images = np.array(test_images)

# Building and Training the Model ..

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=x_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=64,
                    epochs=15, validation_data=(x_val, y_val))

# Plotting Accuracy Curves on Training and Validation sets :

plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# Plotting Loss Curves on Training and Validation sets :

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# Testing and Saving the Model ..

pred = model.predict_classes(test_images)
print('the accuracy on testing set is : ' + str(accuracy_score(test_labels, pred)))
model.save('Traffic_Signs_Classifier.h5')
