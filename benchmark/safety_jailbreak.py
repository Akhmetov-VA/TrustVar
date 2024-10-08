import os

import pandas as pd
from constants import MODELS, MONGO_HOST, MONGO_PASSWORD, MONGO_PORT, MONGO_USERNAME
from pymongo import MongoClient
from src import load_task_mongo

# Get the name of the current file
filename = os.path.basename(__file__)
# Remove the file extension to get just the name
task_name = os.path.splitext(filename)[0]

mongo_uri = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
client = MongoClient(mongo_uri)
db = client["TrustLLM_ru"]
collection = db[task_name]

prompts_data = {"raw": ["{text}"]}

# Чтение данных из файла
df_for_llm = pd.read_json(
    "/home/vadim/work/TrustLLM_ru/data/safety/jailbreak.json",
)

load_task_mongo(
    MODELS,
    collection,
    prompts_data,
    df_for_llm,
    task_name,
    placeholder="text",
    var_col="prompt",
)
