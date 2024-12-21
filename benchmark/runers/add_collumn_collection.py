import logging
import os

from dotenv import load_dotenv
from pymongo import MongoClient
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


def add_target_field_to_collection(db: Database, collection_name: str, value) -> None:
    """
    Добавляет поле 'target' со значением 0 для всех документов в указанной коллекции.

    Args:
        db (Database): Экземпляр базы данных MongoDB.
        collection_name (str): Имя коллекции, в которую добавляется поле.
    """
    collection = db[collection_name]
    logging.info(f"Добавление поля 'target' в коллекцию '{collection_name}'.")

    try:
        # Обновляем все документы, добавляя поле target со значением 0
        result = collection.update_many(
            {},  # Условие: все документы
            {
                "$set": {"task_name": value}
            },  # Действие: добавить поле target со значением 0
        )
        logging.info(
            f"Добавлено поле 'target' со значением 0 для {result.modified_count} документов в коллекции '{collection_name}'."
        )
    except Exception as e:
        logging.error(
            f"Ошибка при добавлении поля 'target' в коллекцию '{collection_name}': {e}"
        )


def main() -> None:
    """
    Основная функция для добавления поля 'target' в коллекцию.
    """
    try:
        # Настройка логирования
        configure_logging()

        # Имя базы данных и коллекции
        database_name = "TrustGen"
        collection_name = "queue_rta_Misuse_ru"
        value = "Misuse_ru"

        # Подключение к MongoDB
        client = get_mongo_client()
        db = client[database_name]

        # Добавление поля 'target'
        add_target_field_to_collection(db, collection_name, value)

        logging.info("Добавление поля 'target' завершено.")
    except Exception as e:
        logging.exception(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    main()
