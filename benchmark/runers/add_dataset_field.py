import logging
import os
from typing import List

from dotenv import load_dotenv
from pymongo import MongoClient, collection
from pymongo.database import Database


def configure_logging() -> None:
    """
    Настраивает логирование для отображения сообщений в консоли.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.info("Логирование успешно настроено.")


def get_mongo_client() -> MongoClient:
    """
    Создает и возвращает подключение к MongoDB на основе переменных окружения.

    Returns:
        MongoClient: Клиент для подключения к MongoDB.
    """
    load_dotenv()
    mongo_username = os.getenv("MONGO_INITDB_ROOT_USERNAME")
    mongo_password = os.getenv("MONGO_INITDB_ROOT_PASSWORD")
    mongo_host = os.getenv("MONGO_HOST")
    mongo_port = os.getenv("MONGO_INITDB_ROOT_PORT")

    mongo_uri = (
        f"mongodb://{mongo_username}:{mongo_password}@{mongo_host}:{mongo_port}/"
    )

    try:
        client = MongoClient(mongo_uri)
        client.admin.command("ping")
        logging.info("Успешное подключение к MongoDB.")
        return client
    except Exception as e:
        logging.exception("Ошибка подключения к MongoDB.")
        raise e


def add_source_collection_to_rta(db: Database, rta_collection_name: str) -> None:
    """
    Проходит по всем коллекциям базы данных и добавляет имя исходной коллекции
    в записи коллекции `RtA` на основе совпадения идентификаторов задач.

    Args:
        db (Database): Экземпляр базы данных MongoDB.
        rta_collection_name (str): Имя коллекции `RtA`.
    """
    rta_collection = db[rta_collection_name]

    for collection_name in db.list_collection_names():
        # Пропускаем коллекцию `RtA` (целевая коллекция)
        if collection_name == rta_collection_name:
            continue

        source_collection = db[collection_name]
        logging.info(
            f"Обработка коллекции '{collection_name}' для сопоставления с 'RtA'."
        )

        # Собираем все идентификаторы из текущей коллекции
        task_ids = list(source_collection.find({}, {"_id": 1}))
        task_ids = [doc["_id"] for doc in task_ids]

        if not task_ids:
            logging.info(f"В коллекции '{collection_name}' нет записей для обработки.")
            continue

        # Обновляем все записи в RtA, соответствующие этим идентификаторам
        result = rta_collection.update_many(
            {"init_id": {"$in": task_ids}},  # Условие: init_id входит в список task_ids
            {"$set": {"dataset": collection_name}},  # Добавляем поле dataset
        )

        logging.info(
            f"Добавлено поле 'dataset' для {result.modified_count} задач "
            f"в коллекции 'RtA' из коллекции '{collection_name}'."
        )


def main() -> None:
    """
    Основная функция для выполнения обработки задач в коллекциях базы данных.
    """
    try:
        # Настройка логирования
        configure_logging()

        # Имя базы данных и коллекции `RtA`
        database_name = "TrustLLM_ru"
        rta_collection_name = "RtA"

        # Подключение к MongoDB
        client = get_mongo_client()
        db = client[database_name]

        # Добавление поля 'source_collection' в коллекцию `RtA`
        add_source_collection_to_rta(db, rta_collection_name)

        logging.info("Обработка завершена, все коллекции сопоставлены с 'RtA'.")
    except Exception as e:
        logging.exception(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    main()
