import string
import numpy as np
from telebot import types
import random
import telebot
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

import nltk

import pickle

import requests

with open("filtered_tokenized_texts 5.pkl", "rb") as f:
    filtered_tokenized_texts = pickle.load(f)

BOS = "<BOS>"
EOS = "<EOS>"

ngrams_config = {
    "pad_left": True,
    "pad_right": True,
    "left_pad_symbol": BOS,
    "right_pad_symbol": EOS,
}


def build_ngram_counts(
    tokenized_texts: Iterable[Iterable[str]], n: int
) -> Dict[Tuple[str, ...], Dict[str, int]]:
    counts = defaultdict(Counter)

    for text in tokenized_texts:
        for ngram in nltk.ngrams(text, n, **ngrams_config):
            prefix = ngram[:-1]
            token = ngram[-1]
            counts[prefix][token] += 1
    return counts


class LanguageModel:
    def __init__(self, tokenized_texts: Iterable[Iterable[str]], n: int) -> None:
        self.n: int = n
        self.probs: Dict[Tuple[str, ...], Dict[str, float]] = defaultdict(Counter)
        ngrams = build_ngram_counts(tokenized_texts, n)
        for prefix, distribution in ngrams.items():
            total_count = sum(distribution.values())
            self.probs[prefix] = Counter(
                {token: count / total_count for token, count in distribution.items()}
            )

    def get_token_distribution(self, prefix: List[str]) -> Dict[str, float]:
        prefix = prefix[max(0, len(prefix) - self.n + 1) :]
        prefix = [BOS] * (self.n - 1 - len(prefix)) + prefix
        return self.probs[tuple(prefix)]

    def get_next_token_prob(self, prefix: List[str], token: str) -> float:
        return self.get_token_distribution(prefix)[token]


model = LanguageModel(filtered_tokenized_texts, n=4)
print("готово!")


def get_next_token(
    lm: LanguageModel, prefix: List[str], temperature: float = 1.0
) -> str:
    distribution: Dict[str, float] = lm.get_token_distribution(prefix)

    probs = [v ** (1 / temperature) for v in distribution.values()]
    total_proba = sum(probs)
    scaled_probs = [p / total_proba for p in probs]
    token_array = np.random.choice(list(distribution.keys()), size=1, p=scaled_probs)

    return token_array[0]


def generate_sentence():
    sentence = " ".join(
        filtered_tokenized_texts[np.random.randint(len(filtered_tokenized_texts))][:3]
    )
    max_num_of_words = 40
    temps = np.linspace(0.5, 5, max_num_of_words + 1)
    for i in range(max_num_of_words):
        token = get_next_token(lm=model, prefix=sentence.split(), temperature=temps[i])
        if token == EOS:
            break
        sentence += f" {token}"
        if token == "." and i > max_num_of_words / 2:
            break

    sentence = sentence[0].upper() + sentence[1:]
    pat = "\s+([{}]+)".format(re.escape(string.punctuation))
    sentence = re.sub("\s{2,}", " ", re.sub(pat, r"\1", sentence))
    sentence = sentence.replace("- ", "-")
    sentence = re.sub(r"(?<=[.!?]\s)(\w)", lambda x: x.group(1).upper(), sentence)
    if i == max_num_of_words - 1:
        sentence += ".."
    return sentence


def take_rated_sentence():
    if len(rated_texts) == 0:
        return "", 0
    random_text = np.random.choice(list(rated_texts.keys()))
    rating = rated_texts[random_text]
    return random_text, rating


bot = telebot.TeleBot("6657863250:AAG7UJg-_BPQH-v2SmOwzgNuyjXt6Eer5aY")
bot_name = bot.get_me().first_name


like_smiles = [
    "😘",
    "😊",
    "❤️️",
    "🥳",
    "🥰",
    "🙏",
    "💋",
    "😉",
    "🤩",
    "😎",
    "🤗",
    "🤭",
    "😁",
    "😄",
    "😇",
    "🤝",
    "😀",
    "😋",
    "🍆",
]
dislike_smiles = [
    "😢",
    "💩",
    "🤮️",
    "😱",
    "😭",
    "😳",
    "🤷",
    "👀",
    "💀",
    "😔",
    "😬",
    "😠",
    "😡",
    "🤬",
    "👎",
    "🖕",
    "💔",
    "👊",
    "☹️",
    "🤔",
    "😕",
    "🥺",
    "😿",
    "😰",
]

rated_texts = {}


sentence = ""


# старотовое меню
generate_button_text = "🧬 Генерировать новый контент"
watch_rated_button_text = "✅ Смотреть оцененный контент"

start_menu_markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
generate_button = types.KeyboardButton(generate_button_text)
watch_rated_button = types.KeyboardButton(watch_rated_button_text)
start_menu_markup.add(generate_button, watch_rated_button)

# меню оценки генерации
like_button_text = "👍 Смешно"
dislike_button_text = "👎 Не смешно"

rate_content_markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
like_button = types.KeyboardButton(like_button_text)
dislike_button = types.KeyboardButton(dislike_button_text)
rate_content_markup.add(like_button, dislike_button)


unknown_command_text = "Не понял тебя 😕"


@bot.message_handler(commands=["start"], content_types=["text"])
def start(message):
    bot.send_message(
        message.chat.id,
        f"Привет, {message.from_user.username}!\nЭто N-граммовая языковая модель *{bot_name}*. "
        + "Она построена на основе telegram каналов Russia Today и РИА новости. Все совпадения случайны.\n"
        + "Ты можешь *участвовать в генерации нового контента* или *посмотреть генерации*, отмеченные людьми, как смешные.\nHave fun!",
        parse_mode="markdown",
        reply_markup=start_menu_markup,
    )

    bot.register_next_step_handler(message, on_start_click)


def on_start_click(message):
    try:
        # нажата генерация контента
        if message.text == generate_button_text:
            sentence = generate_sentence()
            bot.send_message(
                message.chat.id,
                sentence,
                parse_mode="markdown",
                reply_markup=rate_content_markup,
            )
            bot.register_next_step_handler(message, on_generate_click)

        # нажат просмотр оцененного контента
        elif message.text == watch_rated_button_text:
            sentence, rating = take_rated_sentence()
            if sentence == "":
                bot.send_message(
                    message.chat.id,
                    "На данный момент оцененный контент отсутствует 😢",
                    parse_mode="markdown",
                    reply_markup=start_menu_markup,
                )
                bot.register_next_step_handler(message, on_start_click)
            else:
                # sentence = sentence + "\n\nРейтинг: " + str(rating)
                bot.send_message(
                    message.chat.id,
                    sentence,
                    parse_mode="markdown",
                    reply_markup=rate_content_markup,
                )
                bot.send_message(
                    message.chat.id,
                    "*Рейтинг*: " + str(rating),
                    parse_mode="markdown",
                    reply_markup=rate_content_markup,
                )
                bot.register_next_step_handler(message, on_watch_rated_click)
        else:
            bot.send_message(
                message.chat.id, unknown_command_text, reply_markup=start_menu_markup
            )
            bot.register_next_step_handler(message, on_start_click)
    except Exception as e:
        print(repr(e))


# сценарий при генерации контента
def on_generate_click(message):
    try:
        # контент оценен
        if message.text == like_button_text:
            # находим предыдущее сообщение в чате
            previous_message_text = bot.forward_message(
                message.chat.id, message.chat.id, message.message_id - 1
            ).text

            rating = 1
            rated_texts[previous_message_text] = rating
            # rated_texts.append([previous_message_text, rating])

            bot.send_message(
                message.chat.id, "Я добавлю эту генерацию в оцененный контент!"
            )
            bot.send_message(
                message.chat.id, like_smiles[np.random.randint(len(like_smiles))]
            )

        # контен не оценен
        elif message.text == dislike_button_text:
            bot.send_message(
                message.chat.id, dislike_smiles[np.random.randint(len(dislike_smiles))]
            )

        if message.text == dislike_button_text or message.text == like_button_text:
            # генерируем новую новость
            sentence = generate_sentence()
            bot.send_message(
                message.chat.id,
                sentence,
                parse_mode="markdown",
                reply_markup=rate_content_markup,
            )
            bot.register_next_step_handler(message, on_generate_click)
        else:
            bot.send_message(
                message.chat.id, unknown_command_text, reply_markup=start_menu_markup
            )
            bot.register_next_step_handler(message, on_start_click)
    except Exception as e:
        print(repr(e))


def on_watch_rated_click(message):
    try:
        # контент оценен
        if message.text == like_button_text:
            # находим предыдущее сообщение в чате
            previous_message_text = bot.forward_message(
                message.chat.id, message.chat.id, message.message_id - 2
            ).text

            # повышаем рейтинг
            rated_texts[previous_message_text] += 1

            bot.send_message(
                message.chat.id, "*Рейтинг повышен!*", parse_mode="markdown"
            )
            bot.send_message(
                message.chat.id, like_smiles[np.random.randint(len(like_smiles))]
            )

        # контен не оценен
        elif message.text == dislike_button_text:
            # находим предыдущее сообщение в чате
            previous_message_text = bot.forward_message(
                message.chat.id, message.chat.id, message.message_id - 2
            ).text

            # понижаем рейтинг
            rated_texts[previous_message_text] -= 1
            if rated_texts[previous_message_text] < 0:
                del rated_texts[previous_message_text]

            bot.send_message(
                message.chat.id, "*Рейтинг понижен!*", parse_mode="markdown"
            )
            bot.send_message(
                message.chat.id, dislike_smiles[np.random.randint(len(dislike_smiles))]
            )
        if message.text == dislike_button_text or message.text == like_button_text:
            # берем новую новость
            sentence, rating = take_rated_sentence()
            if sentence == "":
                bot.send_message(
                    message.chat.id,
                    "На данный момент оцененный контент отсутствует 😢",
                    parse_mode="markdown",
                    reply_markup=start_menu_markup,
                )
                bot.register_next_step_handler(message, on_start_click)
            else:
                bot.send_message(
                    message.chat.id,
                    sentence,
                    parse_mode="markdown",
                    reply_markup=rate_content_markup,
                )
                bot.send_message(
                    message.chat.id,
                    "*Рейтинг*: " + str(rating),
                    parse_mode="markdown",
                    reply_markup=rate_content_markup,
                )
                bot.register_next_step_handler(message, on_watch_rated_click)
        else:
            bot.send_message(
                message.chat.id, unknown_command_text, reply_markup=start_menu_markup
            )
            bot.register_next_step_handler(message, on_start_click)
    except Exception as e:
        print(repr(e))

# RUN
bot.polling(none_stop=True)
