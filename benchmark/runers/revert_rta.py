import logging
import os
from typing import List

from pymongo import MongoClient
from pymongo.database import Database

from utils.constants import (
    COLLECTIONS_TO_PROCESS,
    MONGO_HOST,
    MONGO_PASSWORD,
    MONGO_PORT,
    MONGO_USERNAME,
)


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
    Создает и возвращает подключение к MongoDB на основе переменных окружения или констант.

    Returns:
        MongoClient: Клиент для подключения к MongoDB.
    """
    try:
        mongo_uri = (
            f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
        )
        client = MongoClient(mongo_uri)
        # Проверка подключения
        client.admin.command("ping")
        logging.info("Успешное подключение к MongoDB.")
        return client
    except Exception as e:
        logging.exception("Ошибка подключения к MongoDB.")
        raise e


def revert_transferred_to_rta_status(db: Database) -> None:
    """
    Сбрасывает флаг 'transferred_to_rta' во всех задачах коллекций из COLLECTIONS_TO_PROCESS.

    Args:
        db (Database): Экземпляр базы данных MongoDB.

    Returns:
        None
    """
    for collection_name in COLLECTIONS_TO_PROCESS:
        collection = db[collection_name]
        logging.info(f"Обработка коллекции '{collection_name}'")
        try:
            result = collection.update_many(
                {"transferred_to_rta": True},
                {"$unset": {"transferred_to_rta": False}},
            )
            logging.info(
                f"В коллекции '{collection_name}' сброшен флаг 'transferred_to_rta' для {result.modified_count} задач."
            )
        except Exception as e:
            logging.error(f"Ошибка при обработке коллекции '{collection_name}': {e}")


def main() -> None:
    """
    Основная функция для сброса флага 'transferred_to_rta' в коллекциях.

    """
    try:
        # Настройка логирования
        configure_logging()

        # Имя базы данных
        database_name = "TrustLLM_ru"

        # Подключение к MongoDB
        client = get_mongo_client()
        db = client[database_name]

        # Сброс флага 'transferred_to_rta' в коллекциях
        revert_transferred_to_rta_status(db)

        logging.info("Все указанные коллекции обработаны.")
    except Exception as e:
        logging.exception(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    main()
