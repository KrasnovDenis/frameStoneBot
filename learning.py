import json
import os

import numpy as np
import requests
import telebot
import tensorflow as tf
from PIL import Image
from skimage import transform
from tensorflow import keras
from tensorflow.keras import layers

import config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
img_height = 400
img_width = 400
batch_size = 32
checkpoint_path = "training_1/cp.ckpt"

class_names = [
    "Биотит",
    "Борнит",
    "Хрисола",
    "Малахит",
    "Мисковит",
    "Пирит",
    "Кварц",
]

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    "C:\\Users\\KPACHOB\\Desktop\\AI\\dataset",
    labels='inferred',
    label_mode='int',
    color_mode="grayscale",
    batch_size=batch_size,
    image_size=(img_height, img_width),  # reshape if not in this size
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="training"
)

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    "C:\\Users\\KPACHOB\\Desktop\\AI\\dataset",
    labels='inferred',
    label_mode='int',
    color_mode="grayscale",
    batch_size=batch_size,
    image_size=(img_height, img_width),  # reshape if not in this size
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="validation"
)


def augment(x, y):
    image = tf.image.random_brightness(x, max_delta=0.05)
    return image, y


def create_model():
    model = keras.Sequential(
        [
            layers.Input((img_height, img_width, 1)),
            layers.Conv2D(16, 3, padding="same"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(10),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True)],
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    _model = create_model()
    train = ds_train.map(augment)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    _model.fit(train, epochs=10, callbacks=[cp_callback])
