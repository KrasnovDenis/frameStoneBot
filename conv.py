import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import boston_housing


(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

class_names = ['Глубинные(интрузивные)', 'Излившиеся (эффузивные)', 'Цементированные', 'Рыхлые',
               'Осадочные - хим. осадки', 'Видоизмененные осадочные породы', 'Органические отложения']

x_train = x_train / 255
x_test = x_test / 255

model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),
                          keras.layers.Dense(128, activation="relu"),
                          keras.layers.Dense(10, activation="softmax")])

model.compile(optimizer=tf.keras.optimizers.SGD(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)

predictions = model.predict(x_test)
print(class_names[np.where(predictions[0] == predictions[0].max())[0][0]])