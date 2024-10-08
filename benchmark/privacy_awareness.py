import os
import uuid

import pandas as pd
from constants import MODELS, MONGO_HOST, MONGO_PASSWORD, MONGO_PORT, MONGO_USERNAME
from pymongo import MongoClient
from src import add_task_name

# Get the name of the current file
filename = os.path.basename(__file__)
# Remove the file extension to get just the name
task_name = os.path.splitext(filename)[0]

mongo_uri = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
client = MongoClient(mongo_uri)
db = client["TrustLLM_ru"]
collection = db[task_name]

prompts_collection = {"raw": ["{text}"]}


# Генерация уникального идентификатора задачи для текущего запуска
job_id = str(uuid.uuid4())

# Чтение данных из файла
df_for_llm = pd.read_json(
    "/home/vadim/work/TrustLLM_ru/data/privacy/privacy_awareness_query.json",
    encoding="cp1251",
)

# цикл добавления задач в Mongo
for model in MODELS:
    for kind, prompts in prompts_collection.items():
        for i in range(len(df_for_llm)):
            row = df_for_llm.iloc[i].to_dict()
            variables = {"text": row["prompt"]}
            for prompt in prompts:
                add_task_name(
                    collection,
                    row,
                    job_id,
                    model,
                    f"{task_name}_{kind}",
                    prompt,
                    variables,
                )

print(f"All task_names for job_id {job_id} have been added.")
