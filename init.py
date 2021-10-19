import telebot
import config
import requests
import io

client = telebot.TeleBot(config.token)


@client.message_handler(content_types=['text'])
def get_text(message):
    if message.text.lower() == 'ping':
        client.send_message(message.chat.id, "pong");


@client.message_handler(content_types=['photo'])
def get_photo(message):
    if message.photo[0].height > 10000 and message.photo[0].width > 10000:
        client.send_message(message.chat.id, "File is big")

    response = requests.get("https://api.telegram.org/file/bot" + config.token + "/" + message.photo[0].file_unique_id)
    file = io.open("sample_image.png", "wb")
    file.write(response.content)
    file.close()


client.polling(none_stop=True, interval=0)
