import logging
import os
import sys

from pymongo import MongoClient

from utils.constants import (
    MONGO_HOST,
    MONGO_PASSWORD,
    MONGO_PORT,
    MONGO_USERNAME,
)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MONGO_DB = os.environ.get("MONGO_DB", "TrustGen")


def get_mongo_client():
    mongo_uri = (
        f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
    )
    client = MongoClient(mongo_uri)
    logger.info("Подключение к MongoDB установлено.")
    return client


def update_task_name_in_rta_queue(task_name: str):
    collection_name = f"queue_rta_{task_name}"
    client = get_mongo_client()
    db = client[MONGO_DB]
    coll_names = db.list_collection_names()

    if collection_name not in coll_names:
        logger.error(f"Коллекция {collection_name} не найдена в базе {MONGO_DB}.")
        return

    result = db[collection_name].update_many({}, {"$set": {"task_name": task_name}})
    logger.info(
        f"Обновлено {result.modified_count} документов в коллекции {collection_name}."
    )


if __name__ == "__main__":
    task_name = "exaggerated_safety_eng"
    update_task_name_in_rta_queue(task_name)
