# ruhatespeech.py

import logging
import os
import uuid

import pandas as pd
from pymongo import MongoClient

from utils.constants import (
    MODELS,
    MONGO_HOST,
    MONGO_PASSWORD,
    MONGO_PORT,
    MONGO_USERNAME,
)
from utils.src import add_task, filter_models, replace_curl

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
db = client["TrustGen"]

# Шаблоны промптов
prompts_data = {"raw": ["{text}"]}

# Путь к файлу данных
file_path = "/home/vadim/work/TrustLLM_ru/data/ruhatespeech/ruhatespeech.csv"

# Чтение данных из CSV
df_for_llm = pd.read_csv(file_path)
df_for_llm.drop("meta", axis=1, inplace=True)

collection = db[task_name]

# Генерация уникального идентификатора задачи для текущего запуска
job_id = str(uuid.uuid4())

# Фильтрация моделей, которые уже присутствуют в коллекции
models_to_add = filter_models(MODELS, collection)

if not models_to_add:
    logging.info("Все модели из MODELS уже присутствуют в коллекции 'ruhatespeech'.")
else:
    for model in models_to_add:
        for _, row in df_for_llm.iterrows():
            row_dict = row.to_dict()
            prompt = row["instruction"]
            variables = replace_curl(
                row["inputs"]
            )  # Предполагается, что 'inputs' — строка
            # Если 'inputs' — строка, нужно преобразовать её в словарь
            try:
                variables = pd.read_json(row["inputs"], typ="series").to_dict()
            except ValueError:
                variables = eval(row["inputs"])  # Используйте осторожно
            add_task(
                collection=collection,
                row=row_dict,
                job_id=job_id,
                model=model,
                prompt=prompt,
                variables=variables,
                target=row.get("outputs", -1),
            )
    logging.info(f"All RuhateSpeech tasks have been added for models: {models_to_add}")

print(f"Данные из файла '{file_path}' успешно загружены в коллекцию '{task_name}'.")
