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
    "Гранит",
    "Кварц",
    "Малахит",
    "Мрамор",
    "Пирит", ]

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


def load(filename):
    np_image = Image.open(filename)
    image_tensor = tf.convert_to_tensor(np_image, dtype=tf.float32)
    image_tensor = transform.resize(image_tensor, (img_height, img_width, 1))
    image_tensor = tf.expand_dims(image_tensor, 0)
    return image_tensor


def get_file_path_by_id(file_id):
    link_to_get_path = 'https://api.telegram.org/bot' + config.token + '/getFile?file_id=' + file_id
    file_path = json.loads(requests.get(link_to_get_path).text)["result"]["file_path"]
    link_to_download = 'https://api.telegram.org/file/bot' + config.token + "/" + file_path
    return link_to_download


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
train = ds_train.map(augment)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
_model.fit(train, epochs=10, callbacks=[cp_callback])
client = telebot.TeleBot(config.token)


@client.message_handler(content_types=['text'])
def get_text(message):
    if message.text.lower() == 'ping':
        client.send_message(message.chat.id, "pong")


@client.message_handler(content_types=['photo'])
def get_photo(message):
    link = get_file_path_by_id(message.photo[0].file_id)
    img_data = requests.get(link).content
    with open('image_name.jpg', 'wb') as handler:
        handler.write(img_data)
    np_image = load('image_name.jpg')
    predictions = _model.predict(np_image)
    stat = "Вот что я узнал на этой картинке:\n"
    sum_predict = sum(map(abs, predictions[0][0:7])) + 0.00001
    for i in predictions[0][0:7]:
        percent = "{:.2f}".format(round(i / sum_predict * 100, 2))
        if predictions[0].max() == i:
            stat += "<b>" + class_names[np.where(predictions[0] == i)[0][0]] + ": " + percent + "%</b>\n"
        else:
            stat += class_names[np.where(predictions[0] == i)[0][0]] + ": " + percent + "%\n"
    client.send_message(message.chat.id, stat, parse_mode="HTML")


client.polling(none_stop=True, interval=0)
