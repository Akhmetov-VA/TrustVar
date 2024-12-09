# jailbreak.py

import logging
import os

import pandas as pd
from pymongo import MongoClient

from utils.constants import (
    MODELS,
    MONGO_HOST,
    MONGO_PASSWORD,
    MONGO_PORT,
    MONGO_USERNAME,
)
from utils.src import filter_models, load_task_mongo, replace_curl

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logging.info("Logging configured successfully.")

# Получение имени текущего файла
filename = os.path.basename(__file__)
task_name = os.path.splitext(filename)[0]

# Подключение к MongoDB
mongo_uri = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
client = MongoClient(mongo_uri)
db = client["TrustLLM_ru"]

prompts_data = {"raw": ["{text}"]}

# Чтение данных из файла
df_for_llm = pd.read_excel("/home/vadim/work/TrustLLM_ru/data/safety/jailbreak_ru.xlsx")

df_for_llm = df_for_llm.rename(
    {"prompt ru": "init_prompt", "label ru": "label"}, axis=1
)

# Добавляем поле 'kind' в DataFrame
df_for_llm["kind"] = df_for_llm["label"].str.lower().str.replace(" ", "_")

# Указываем коллекцию 'jailbreak_ru'
collection = db["jailbreak_ru"]

# Получаем список моделей, уже присутствующих в коллекции
models_to_add = filter_models(MODELS, collection)

if not models_to_add:
    logging.info("Все модели из MODELS уже присутствуют в базе данных.")
else:
    # Загружаем задачи только для отсутствующих моделей
    load_task_mongo(
        models=models_to_add,
        collection=collection,
        prompts_data=prompts_data,
        df_for_llm=df_for_llm,
        placeholder="text",
        var_col="init_prompt",
        target="RtA",  # модель должна отказываться отвечать
    )
    logging.info(f"All Jailbreak tasks have been added for models: {models_to_add}")

print(
    f"Данные из файла '/home/vadim/work/TrustLLM_ru/data/safety/jailbreak_ru.xlsx' успешно загружены в коллекцию 'jailbreak_ru'."
)
