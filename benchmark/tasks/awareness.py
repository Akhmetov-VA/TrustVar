import logging
import os
import uuid

import pandas as pd
from pymongo import MongoClient

from benchmark.constants import (
    MODELS,
    MONGO_HOST,
    MONGO_PASSWORD,
    MONGO_PORT,
    MONGO_USERNAME,
)
from utils.src import add_task, replace_curl

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

# Get the name of the current file
filename = os.path.basename(__file__)
# Remove the file extension to get just the name
task_name = os.path.splitext(filename)[0]

# Подключение к MongoDB
mongo_uri = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
client = MongoClient(mongo_uri)
db = client["TrustLLM_ru"]

prompts_collection = {"raw": ["{text}"]}

# Генерация уникального идентификатора задачи для текущего запуска
job_id = str(uuid.uuid4())

# Чтение данных из файла
try:
    df_for_llm = pd.read_json(
        "/home/vadim/work/TrustLLM_ru/data/privacy/privacy_awareness_query.json",
        encoding="cp1251",
    )
except ValueError as e:
    logging.error(f"Ошибка при чтении файла JSON: {e}")
    raise

df_for_llm["type"].fillna("normal", inplace=True)
df_for_llm["type"].replace({"обычный": "normal", "нормальный": "normal"}, inplace=True)
df_for_llm.rename({"prompt": "init_prompt"}, axis=1, inplace=True)

# Цикл добавления задач в Mongo
for model in MODELS:
    for kind, prompts in prompts_collection.items():
        for task_type, group_df in df_for_llm.groupby("type"):
            collection = db[f"{task_name}_{task_type}"]

            for _, row in group_df.iterrows():
                row_dict = row.to_dict()
                variables = {"text": replace_curl(row_dict["init_prompt"])}
                for prompt in prompts:
                    add_task(
                        collection,
                        row_dict,
                        job_id,
                        model,
                        prompt,
                        variables,
                    )

logging.info(f"All task_names for job_id {job_id} have been added.")
