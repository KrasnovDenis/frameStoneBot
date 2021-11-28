import json
import numpy as np
import requests
import tensorflow as tf
import telebot
from skimage import transform
from tensorflow.keras import layers

import config
from PIL import Image
from learning import create_model, checkpoint_path, ds_validation, class_names, img_height, img_width, ds_train

model = create_model()
client = telebot.TeleBot(config.token)
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(ds_train, verbose=2)


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

    predictions = model.predict(np_image)
    # predictions = model.predict(ds_validation)
    # client.send_photo(message.chat.id, caption=class_names[np.where(predictions[0] == predictions[0].max())[0][0]])
    client.send_message(message.chat.id, class_names[np.where(predictions[0] == predictions[0].max())[0][0]])


client.polling(none_stop=True, interval=0)
