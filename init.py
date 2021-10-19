import telebot
import config
import requests
import io
import json

client = telebot.TeleBot(config.token)


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
    if message.photo[0].height > 10000 and message.photo[0].width > 10000:
        client.send_message(message.chat.id, "File is big")

    link = get_file_path_by_id(message.photo[0].file_id)
    img_data = requests.get(link).content
    with open('image_name.jpg', 'wb') as handler:
        handler.write(img_data)


client.polling(none_stop=True, interval=0)
