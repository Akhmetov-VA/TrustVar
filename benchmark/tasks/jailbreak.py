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
from utils.src import load_task_mongo

# Получение имени текущего файла
filename = os.path.basename(__file__)
# Удаление расширения файла, чтобы получить только имя
task_name = os.path.splitext(filename)[0]

mongo_uri = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
client = MongoClient(mongo_uri)
db = client["TrustLLM_ru"]

prompts_data = {"raw": ["{text}"]}

# # Чтение данных из файла
# df_for_llm = pd.read_json(
#     "/home/vadim/work/TrustLLM_ru/data/safety/jailbreak.json",
# )

df_for_llm = pd.read_excel("/home/vadim/work/TrustLLM_ru/data/safety/jailbreak_ru.xlsx")

df_for_llm = df_for_llm.rename(
    {"prompt ru": "init_prompt", "label ru": "label"}, axis=1
)

# Добавляем поле 'kind' в DataFrame
df_for_llm["kind"] = df_for_llm["label"].str.lower().str.replace(" ", "_")

# Указываем коллекцию 'jailbreak_ru'
collection = db["jailbreak_ru"]

# Вызываем load_task_mongo для всей DataFrame
load_task_mongo(
    MODELS,
    collection,
    prompts_data,
    df_for_llm,
    placeholder="text",
    var_col="init_prompt",
    target="RtA",  # модель должна отказываться отвечать
)
