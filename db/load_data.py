import os

import pandas as pd
import requests

# Загрузка переменных окружения из .env файла
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# Получение данных для подключения из переменных окружения
MONGO_USERNAME = os.getenv("MONGO_INITDB_ROOT_USERNAME")
MONGO_PASSWORD = os.getenv("MONGO_INITDB_ROOT_PASSWORD")
MONGO_HOST = os.getenv("MONGO_HOST")
MONGO_PORT = os.getenv("MONGO_INITDB_ROOT_PORT")

# Формирование URI для подключения к MongoDB
mongo_uri = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"

# Подключение к MongoDB
client = MongoClient(mongo_uri)

# Выбор базы данных и коллекции
db = client.llm_database
collection = db.dataset_all_exp2


# Загрузка данных из MongoDB в DataFrame
data = list(collection.find())  # Преобразование данных в список

# Преобразование данных в DataFrame
df = pd.DataFrame(data)

df.to_csv("data/dataset_all_exp2.csv")


### save separate files
def sanitize_filename(name):
    # Replace invalid characters with underscore or another valid character
    invalid_chars = {"/": "_", ":": "-", ".": "_"}
    for char, replacement in invalid_chars.items():
        name = name.replace(char, replacement)
    return name


model_names = df["model"].unique()  # Уникальные идентификаторы моделей

for name in model_names:
    model_data = df[df["model"] == name].head(2840)  # Первые 2840 записей для модели

    sanitized_name = sanitize_filename(name)
    model_data.to_csv(f"data/model_{sanitized_name}_data.csv")
