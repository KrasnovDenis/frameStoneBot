import os
import random

import cv2
import numpy as np
import tensorflow as tf
from imutils import paths
from keras import backend as K
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
img_height = 64
img_width = 64
batch_size = 32
EPOCHS = 30
checkpoint_path = "training_1/cp.ckpt"

class_names = [
    "Биотит",
    "Борнит",
    "Гранит",
    "Малахит",
    "Мрамор",
    "Пирит",
    "Кварц",
]
DATASET_PATH = "C:\\Users\\KPACHOB\\Desktop\\AI\\dataset"


def create_model():
    model = Sequential()
    inputShape = (img_height, img_width, 3)
    chanDim = -1

    # если мы используем порядок "channels first", обновляем
    # входное изображение и размер канала
    if K.image_data_format() == "channels_first":
        inputShape = (3, img_height, img_width)
        chanDim = 1

    # слои CONV => RELU => POOL
    model.add(Conv2D(32, (3, 3), padding="same",
                     input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # слои (CONV => RELU) * 2 => POOL
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # (CONV => RELU) * 3 => POOL layer set
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # первый (и единственный) набор слоев FC => RELU
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # классификатор softmax
    model.add(Dense(len(class_names)))
    model.add(Activation("softmax"))
    init_lr = 0.01
    model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=init_lr, decay=init_lr / EPOCHS),
                  metrics=["accuracy"])
    return model


if __name__ == "__main__":

    data = []
    labels = []

    # берём пути к изображениям и рандомно перемешиваем
    imagePaths = sorted(list(paths.list_images(DATASET_PATH)))
    random.seed(42)
    random.shuffle(imagePaths)

    # цикл по изображениям
    for imagePath in imagePaths:
        # загружаем изображение, меняем размер на 64x64 пикселей
        # (требуемые размеры для SmallVGGNet), изменённое изображение
        # добавляем в список
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (64, 64))
        data.append(image)

        # извлекаем метку класса из пути к изображению и обновляем
        # список меток
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)

    # масштабируем интенсивности пикселей в диапазон [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # разбиваем данные на обучающую и тестовую выборки, используя 75%
    # данных для обучения и оставшиеся 25% для тестирования
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

    # конвертируем метки из целых чисел в векторы (для 2х классов при
    # бинарной классификации вам следует использовать функцию Keras
    # “to_categorical” вместо “LabelBinarizer” из scikit-learn, которая
    # не возвращает вектор)
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)

    # создаём генератор для добавления изображений
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")

    _model = create_model()
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    _model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size), callbacks=[cp_callback],
                        validation_data=(testX, testY), steps_per_epoch=len(trainX) // batch_size,
                        epochs=EPOCHS)
