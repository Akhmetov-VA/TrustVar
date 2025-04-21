import json
import logging
import os
from typing import List

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from utils.constants import (
    MONGO_DB,
    MONGO_HOST,
    MONGO_PASSWORD,
    MONGO_PORT,
    MONGO_SOURCE_DB_NAME,
    MONGO_SOURCE_URI,
    MONGO_USERNAME,
)

datasets_name: List[str] = [
    "dataset_ConfAIDe",
    "dataset_LibrusecHistory",
    "dataset_LibrusecMHQA",
    "dataset_Misuse_ru_420",
    "dataset_regestry",
    "dataset_ruBABILongQA_1",
    "dataset_ruBABILongQA_2",
    "dataset_ruBABILongQA_3",
    "dataset_ruBABILongQA_4",
    "dataset_ruBABILongQA_5",
    "dataset_RuBLiMP",
    "dataset_ruQasper",
    "tasks",
]


def get_mongo_client(uri: str = None) -> MongoClient:
    logging.info("Attempting to connect to MongoDB...")
    if uri is None:
        mongo_uri = (
            f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
        )
    else:
        mongo_uri = uri
    try:
        client = MongoClient(mongo_uri)
        client.admin.command("ping")
        logging.info("Connected to MongoDB successfully.")
        return client
    except Exception as e:
        logging.error("Failed to connect to MongoDB.")
        raise e


logging.basicConfig(level=logging.INFO)


def fill_bd_from_datasets() -> None:
    client = get_mongo_client()
    db = client[MONGO_DB]

    for collection_name in datasets_name:
        with open(
            f"{os.path.dirname(__file__)}/datasets/{collection_name}.json", "r"
        ) as f:
            data = json.load(f)
            if collection_name not in db.list_collection_names():
                db[collection_name].insert_many(data)
                logging.info(f"collection {collection_name} is processed")


def fill_from_another_mongo() -> None:
    try:
        client = get_mongo_client()
        db = client[MONGO_DB]

        source_client = get_mongo_client(MONGO_SOURCE_URI)
        source_db = source_client[MONGO_SOURCE_DB_NAME]

        for collection_name in source_db.list_collection_names():
            filling = source_db[collection_name].find()
            if filling and (collection_name not in db.list_collection_names()):
                db[collection_name].insert_many(source_db[collection_name].find())
            else:
                pass
    except:
        logging.error("Failed to fill from source-MongoDB. Filling from datasets")
        fill_bd_from_datasets()


if __name__ == "__main__":
    try:
        if MONGO_SOURCE_DB_NAME and MONGO_SOURCE_DB_NAME:
            fill_from_another_mongo()
        else:
            fill_bd_from_datasets()
        logging.info(f" Datasets' insertion succeeded!")
    except Exception as e:
        logging.error(f" Datasets' insertion failed! {e}")
