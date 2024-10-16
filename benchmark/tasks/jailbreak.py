import os

import pandas as pd
from pymongo import MongoClient

from benchmark.constants import (
    MODELS,
    MONGO_HOST,
    MONGO_PASSWORD,
    MONGO_PORT,
    MONGO_USERNAME,
)
from benchmark.src import load_task_mongo

# Get the name of the current file
filename = os.path.basename(__file__)
# Remove the file extension to get just the name
task_name = os.path.splitext(filename)[0]

mongo_uri = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
client = MongoClient(mongo_uri)
db = client["TrustLLM_ru"]

prompts_data = {"raw": ["{text}"]}

# Чтение данных из файла
df_for_llm = pd.read_json(
    "/home/vadim/work/TrustLLM_ru/data/safety/jailbreak.json",
)
df_for_llm = df_for_llm.rename({"prompt": "init_prompt"}, axis=1)
df_for_llm["label"] = df_for_llm["label"].apply(lambda x: x[0])

for kind, group_df in df_for_llm.groupby("label"):
    kind = kind.lower().replace(" ", "_")
    collection = db[f"{task_name}_{kind}"]

    load_task_mongo(
        MODELS,
        collection,
        prompts_data,
        group_df,
        placeholder="text",
        var_col="init_prompt",
        target=1,  # модель должна отказываться отвечать
    )
