# ood.py

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
db = client["TrustGen"]
collection = db[task_name]

# Генерация уникального идентификатора задачи для текущего запуска
job_id = str(uuid.uuid4())

# Загрузка данных
df_for_llm = pd.read_json("/home/vadim/work/TrustLLM_ru/data/ood/ood_detection_ru.json")

prompts_collection_ood = {"raw": ["{text}"]}

# Получаем список моделей, уже присутствующих в коллекции
models_to_add = filter_models(MODELS, collection)

if not models_to_add:
    logging.info("Все модели из MODELS уже присутствуют в базе данных.")
else:
    # Загружаем задачи только для отсутствующих моделей
    load_task_mongo(
        models=models_to_add,
        collection=collection,
        prompts_data=prompts_collection_ood,
        df_for_llm=df_for_llm,
        var_col="prompt",
        target="RtA",  # при необходимости можно изменить
    )
    logging.info(f"All OOD tasks have been added for models: {models_to_add}")
