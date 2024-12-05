import ast
import os
import uuid

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from pymongo import MongoClient

from utils.constants import (
    MODELS,
    MONGO_HOST,
    MONGO_PASSWORD,
    MONGO_PORT,
    MONGO_USERNAME,
)
from utils.src import add_task

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
file_path = "/home/vadim/work/TrustLLM_ru/data/ruhatespeech/ruhatespeech.csv"

# Чтение данных из Excel
df_for_llm = pd.read_csv(file_path)

df_for_llm.drop("meta", axis=1, inplace=True)


collection = db[task_name]

job_id = str(uuid.uuid4())

# Цикл для добавления задач в MongoDB
for model in MODELS:
    for i in range(len(df_for_llm)):
        row = df_for_llm.iloc[i].to_dict()
        prompt = row["instruction"]
        variables = ast.literal_eval(row["inputs"])
        add_task(
            collection,
            row,
            job_id,
            model,
            prompt,
            variables,
            target=row.get("outputs", -1),
        )

print(f"Данные из файла '{file_path}' успешно загружены в коллекцию {task_name}.")
