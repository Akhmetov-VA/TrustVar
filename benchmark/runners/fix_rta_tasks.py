import logging
import os

from bson import ObjectId
from pymongo import MongoClient
from pymongo.database import Database

from utils.constants import (
    MONGO_DB,
    MONGO_HOST,
    MONGO_PASSWORD,
    MONGO_PORT,
    MONGO_USERNAME,
)

task = "jailbreak"

# Имена коллекций
SOURCE_COLLECTION = os.environ.get("SOURCE_COLLECTION", f"queue_{task}")
RTA_COLLECTION = os.environ.get("RTA_COLLECTION", f"queue_rta_{task}")

# Новая модель, которую нужно проставить
NEW_RTA_MODEL = "qwen2.5:32b-instruct-q4_0"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_mongo_client() -> MongoClient:
    uri = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
    client = MongoClient(uri)
    logger.info("Подключились к MongoDB")
    return client


def get_db(client: MongoClient) -> Database:
    return client[MONGO_DB]


def fix_untransferred_tasks(db: Database):
    src = db[SOURCE_COLLECTION]
    rta = db[RTA_COLLECTION]

    # 1) находим все задачи, которые считались перенесёнными
    cursor = src.find({"metric": "RtA", "status": "transfered_to_rta"}, {"_id": 1})

    count = 0
    for doc in cursor:
        task_id = doc["_id"]
        # 2) проверяем, есть ли результат в RTA_COLLECTION
        exists = rta.find_one({"source_id": str(task_id)})
        if not exists:
            # 3) обновляем задачу
            result = src.update_one(
                {"_id": task_id},
                {"$set": {"status": "completed", "rta_model": NEW_RTA_MODEL}},
            )
            if result.modified_count:
                count += 1
                logger.info(
                    f"Задача {task_id} → status=completed, rta_model={NEW_RTA_MODEL}"
                )
    logger.info(f"Всего обновлено: {count} задач")


def main():
    client = get_mongo_client()
    db = get_db(client)
    fix_untransferred_tasks(db)


if __name__ == "__main__":
    main()
