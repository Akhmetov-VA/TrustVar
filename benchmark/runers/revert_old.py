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
    # Загрузка переменных окружения из файла .env
    load_dotenv()

    # Получение деталей подключения из переменных окружения
    mongo_username = os.getenv("MONGO_INITDB_ROOT_USERNAME")
    mongo_password = os.getenv("MONGO_INITDB_ROOT_PASSWORD")
    mongo_host = os.getenv("MONGO_HOST")
    mongo_port = os.getenv("MONGO_INITDB_ROOT_PORT")

    # Формирование URI для подключения
    mongo_uri = (
        f"mongodb://{mongo_username}:{mongo_password}@{mongo_host}:{mongo_port}/"
    )

    try:
        client = MongoClient(mongo_uri)
        # Проверка подключения
        client.admin.command("ping")
        logging.info("Успешное подключение к MongoDB.")
        return client
    except Exception as e:
        logging.exception("Ошибка подключения к MongoDB.")
        raise e


def revert_task_status(collection: collection.Collection) -> None:
    """
    Отменяет статус задач в коллекции с 'transferred' и 'measured' на 'completed'
    и удаляет поля 'pred' и 'metric'.

    Args:
        collection (Collection): Коллекция MongoDB, в которой выполняется операция.

    Returns:
        None
    """
    try:
        # Определяем статусы для отката
        statuses_to_revert = ["transferred", "measured"]

        result = collection.update_many(
            {"status": {"$in": statuses_to_revert}},
            {"$set": {"status": "completed"}, "$unset": {"pred": "", "metric": ""}},
        )
        logging.info(
            f"Отменено {result.modified_count} задач из {statuses_to_revert} в 'completed' в коллекции '{collection.name}'."
        )
    except Exception as e:
        logging.error(f"Ошибка при обработке коллекции '{collection.name}': {e}")


def process_collections(db: Database) -> None:
    """
    Обрабатывает все коллекции в базе данных, выполняя обновление статусов задач.

    Args:
        db (Database): Экземпляр базы данных MongoDB.

    Returns:
        None
    """
    for collection_name in db.list_collection_names():
        collection = db[collection_name]
        logging.info(f"Обработка коллекции '{collection_name}'")
        revert_task_status(collection)


def main() -> None:
    """
    Основная функция для выполнения обработки задач в коллекциях базы данных.
    """
    try:
        # Настройка логирования
        configure_logging()

        # Имя базы данных
        database_name = "TrustLLM_ru"

        # Подключение к MongoDB
        client = get_mongo_client()
        db = client[database_name]

        # Обработка коллекций
        process_collections(db)

        logging.info("Все указанные коллекции обработаны.")
    except Exception as e:
        logging.exception(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    main()
