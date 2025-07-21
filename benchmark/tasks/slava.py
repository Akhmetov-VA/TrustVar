# slava.py

import logging
import os
import uuid

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

from utils.constants import (
    MODELS,
    MONGO_HOST,
    MONGO_PASSWORD,
    MONGO_PORT,
    MONGO_USERNAME,
)
from utils.src import add_task, filter_models, replace_curl

# Загрузка переменных окружения из .env файла
load_dotenv()
TASK = "SLAVA_only4"

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logging.info("Logging configured successfully.")

# Подключение к MongoDB
mongo_uri = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
client = MongoClient(mongo_uri)
db = client["TrustGen"]
collection = db[TASK]

# Генерация уникального идентификатора задачи для текущего запуска
job_id = str(uuid.uuid4())

# Чтение данных из файла
df_for_llm = pd.read_json(
    "/home/vadim/work/TrustLLM_ru/data/slava/open_questions_data__one_answer.jsonl",
    lines=True,
)

# 1. Замена фразы в столбце 'instruction'
df_for_llm["instruction"] = df_for_llm["instruction"].str.replace(
    "Выберите один вариант правильного ответа и укажите его номер в ответе.",
    "Укажите в результате цифру правильного ответа. Результат должен содержать только одну цифру. Результат:",
    regex=False,
)

# 2. Оставляем только строки с 4 вариантами ответов
df_for_llm["num_options"] = df_for_llm["inputs"].apply(
    lambda x: sum(1 for v in x["options"].values() if v is not None)
)
df_for_llm = df_for_llm[df_for_llm["num_options"] == 4].reset_index(drop=True)

# Удаляем временный столбец 'num_options'
df_for_llm = df_for_llm.drop(columns=["num_options"])

# Фильтрация моделей, которые уже присутствуют в коллекции
models_to_add = filter_models(MODELS, collection)

if not models_to_add:
    logging.info("Все модели из MODELS уже присутствуют в коллекции 'SLAVA_only4'.")
else:
    for model in models_to_add:
        for _, row in df_for_llm.iterrows():
            row_dict = row.to_dict()
            prompt = row["instruction"]
            variables = {"task": row["inputs"]["task"], "text": row["inputs"]["text"]}
            # Добавляем только не пустые варианты ответов
            options = row["inputs"]["options"]
            option_keys = sorted([k for k in options if options[k] is not None])
            for key in option_keys:
                # Преобразуем ключ 'option_1' в 'Option_1'
                variables[key.capitalize()] = options[key]
            # Добавляем задачу в MongoDB
            add_task(
                collection=collection,
                row=row_dict,
                job_id=job_id,
                model=model,
                prompt=prompt,
                variables=variables,
                target=row.get("outputs", -1),
            )
    logging.info(f"All Slava tasks have been added for models: {models_to_add}")

print(
    f"Данные из файла '/home/vadim/work/TrustLLM_ru/data/slava/open_questions_data__one_answer.jsonl' успешно загружены в коллекцию '{TASK}'."
)
