import os
import uuid

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

from benchmark.constants import (
    MODELS,
    MONGO_HOST,
    MONGO_PASSWORD,
    MONGO_PORT,
    MONGO_USERNAME,
)
from benchmark.src import add_task

# Загрузка переменных окружения из .env файла
load_dotenv()
TASK = "SLAVA_only4"

mongo_uri = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
client = MongoClient(mongo_uri)
db = client["TrustLLM_ru"]
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
    "Укажите в результате цифру правильного ответа.",
    regex=False,
)


# 2. Оставляем только строки с 4 вариантами ответов
df_for_llm["num_options"] = df_for_llm["inputs"].apply(
    lambda x: sum(1 for v in x["options"].values() if v is not None)
)
df_for_llm = df_for_llm[df_for_llm["num_options"] == 4].reset_index(drop=True)

# Удаляем временный столбец 'num_options'
df_for_llm = df_for_llm.drop(columns=["num_options"])

# Цикл для добавления задач в MongoDB
for model in MODELS:
    for i in range(len(df_for_llm)):
        row = df_for_llm.iloc[i].to_dict()
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
            collection,
            row,
            job_id,
            model,
            prompt,
            variables,
            target=row.get("outputs", -1),
        )

print(f"All tasks for job_id {job_id} have been added.")
