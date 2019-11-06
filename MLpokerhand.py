import numpy as np
import keras
import losswise
from losswise.libs import LosswiseKerasCallback
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.metrics import categorical_crossentropy

losswise.set_api_key('HK5Q1JWEY')

def extract_data(filename):

    file = open(filename, 'r')

    samples = []
    labels = []

    for line in file:
        line_data = line.strip().split(sep=',')
        line_data = [int(x) for x in line_data]

        for i in range(10):
            if i % 2 == 0:
                line_data[i] /= 4
            else:
                line_data[i] /= 13

        samples.append(np.array(line_data[:-1]))
        labels.append(line_data[-1])


    return np.array(samples), np.array(labels)

l_relu = LeakyReLU(alpha=0.2)

model = Sequential([
    Dense(32, input_shape=(10,), activation='relu'),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(SGD(lr=0.15), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

samples, labels = extract_data('training_set.txt')

training = model.fit(samples,
          labels,
          batch_size=10,
          epochs=20,
          validation_split=0.05,
          shuffle=True,
          verbose=2,
          callbacks=[
              LosswiseKerasCallback(tag='keras test', params={'lstm_size': 32}, track_git=True, display_interval=500),
              ReduceLROnPlateau(monitor='val_acc', patience=5, factor=0.5, verbose=1, min_lr=0.0001),
              EarlyStopping(monitor='val_acc', min_delta=0.0003, patience=20, verbose=1)
          ])

test_samples, test_labels = extract_data('testing_set.txt')

scores = model.evaluate(test_samples, test_labels)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
