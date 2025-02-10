import logging
import os

from pymongo import MongoClient
from pymongo.database import Database

from utils.constants import MONGO_HOST, MONGO_PASSWORD, MONGO_PORT, MONGO_USERNAME

# Используем переменную MONGO_DB из окружения (или значение по умолчанию)
MONGO_DB = os.environ.get("MONGO_DB", "TrustGen")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_mongo_client() -> MongoClient:
    """
    Создаёт подключение к MongoDB.
    """
    mongo_uri = (
        f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
    )
    client = MongoClient(mongo_uri)
    logger.info("Успешно подключились к MongoDB.")
    return client


def get_db() -> Database:
    client = get_mongo_client()
    return client[MONGO_DB]


def reset_rta_tasks_to_completed(db: Database) -> None:
    """
    Проходит по всем обычным очередям (коллекциям, начинающимся с "queue_",
    но исключая коллекции rta-очередей) и устанавливает статус документов с метрикой "RtA"
    в значение "completed". Это позволит основному процессу переноса (fetch_rta_tasks)
    вновь подобрать эти задачи.
    """
    # Получаем список коллекций, начинающихся с "queue_" и не содержащих "queue_rta_"
    collections = [
        c
        for c in db.list_collection_names()
        if c.startswith("queue_") and not c.startswith("queue_rta_")
    ]

    for coll_name in collections:
        coll = db[coll_name]
        # Обновляем документы с metric "RtA", у которых статус не равен "completed"
        result = coll.update_many(
            {"metric": "RtA", "status": {"$ne": "completed"}},
            {"$set": {"status": "completed"}},
        )
        logger.info(
            f"В коллекции '{coll_name}' обновлено {result.modified_count} документов до статуса 'completed'."
        )


def main():
    db = get_db()
    reset_rta_tasks_to_completed(db)


if __name__ == "__main__":
    main()
