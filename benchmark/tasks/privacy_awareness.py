# awarness.py

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
from utils.src import filter_models, load_task_mongo

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logging.info("Logging configured successfully.")

# Получение имени текущего файла
filename = os.path.basename(__file__)
task_name = "privacy_awareness"  # Указываем постоянное название коллекции

# Подключение к MongoDB
mongo_uri = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
client = MongoClient(mongo_uri)
db = client["TrustLLM_ru"]
collection = db[task_name]

# Генерация уникального идентификатора задачи для текущего запуска
job_id = str(uuid.uuid4())

# Чтение данных из файла
file_path = "/home/vadim/work/TrustLLM_ru/data/privacy/privacy_awareness_query.json"
try:
    df_for_llm = pd.read_json(
        file_path,
        encoding="cp1251",
    )
except ValueError as e:
    logging.error(f"Ошибка при чтении файла JSON: {e}")
    raise

# Обработка данных
df_for_llm["type"].fillna("normal", inplace=True)
df_for_llm["type"].replace({"обычный": "normal", "нормальный": "normal"}, inplace=True)
df_for_llm.rename({"prompt": "init_prompt"}, axis=1, inplace=True)

# Фильтрация моделей, которые уже присутствуют в коллекции
models_to_add = filter_models(MODELS, collection)

if not models_to_add:
    logging.info(f"Все модели из MODELS уже присутствуют в коллекции '{task_name}'.")
else:
    # Загружаем задачи только для отсутствующих моделей
    load_task_mongo(
        models=models_to_add,
        collection=collection,
        prompts_data={"raw": ["{text}"]},
        df_for_llm=df_for_llm,
        placeholder="text",
        var_col="init_prompt",
        target="RtA",  # модель должна отказываться отвечать
    )
    logging.info(f"All Awareness tasks have been added for models: {models_to_add}")

logging.info(f"All tasks for job_id {job_id} have been added.")
print(f"Данные из файла '{file_path}' успешно загружены в коллекцию '{task_name}'.")
