import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
img_height = 256
img_width = 256
batch_size = 32
checkpoint_path = "training_1/cp.ckpt"
_model = None
class_names = ["биотит",
               "борнит",
               "хрикосола",
               "малахит",
               "мисковит",
               "пирит",
               "кварц"]

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    "C:\\Users\\KPACHOB\\Desktop\\dataset",
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
    "C:\\Users\\KPACHOB\\Desktop\\dataset",
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


def create_model():
    _model = keras.Sequential(
        [
            layers.Input((img_height, img_width, 1)),
            layers.Conv2D(16, 3, padding="same"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(10),
        ]
    )

    _model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True)],
        metrics=["accuracy"]
    )
    print(_model.summary())
    return _model


def augment(x, y):
    image = tf.image.random_brightness(x, max_delta=0.05)
    return image, y


def fit_model():
    train = ds_train.map(augment)
    model = create_model()

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    model.fit(train, epochs=5, callbacks=[cp_callback])
    predictions = model.predict(ds_validation)
    print(class_names[np.where(predictions[0] == predictions[0].max())[0][0]])


# fit_model()
