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
task_name = os.path.splitext(filename)[0]

# Настройка подключения к MongoDB
mongo_uri = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
client = MongoClient(mongo_uri)
db = client["TrustLLM_ru"]

# Шаблоны промптов
prompts_data = {"raw": ["{text}"]}

# Путь к файлу данных
file_path = "/home/vadim/work/TrustLLM_ru/data/misuse/Misuse_ru_420 v2.xlsx"

# Чтение данных из Excel
df_for_llm = pd.read_excel(file_path, index_col=0)

mask = df_for_llm["type"].isna()
df_for_llm.loc[mask, "type"] = df_for_llm.loc[mask, "label"]

df_for_llm.drop("label", axis=1, inplace=True)

# Переименование столбцов для удобства обработки
df_for_llm = df_for_llm.rename(
    {"prompt": "init_prompt"},
    axis=1,
)

# Указываем коллекцию 'misuse_ru'
collection = db["misuse_ru"]

# Загрузка данных в MongoDB
load_task_mongo(
    MODELS,
    collection,
    prompts_data,
    df_for_llm,
    placeholder="text",
    var_col="init_prompt",
    target="RtA",  #
)

print(f"Данные из файла '{file_path}' успешно загружены в коллекцию 'misuse_ru'.")
