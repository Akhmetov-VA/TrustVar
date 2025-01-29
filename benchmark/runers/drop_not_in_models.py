import os
from typing import List

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from utils.constants import COLLECTIONS_TO_PROCESS, MODELS


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

    # Формирование URI для подключения к MongoDB
    mongo_uri = f"mongodb://{mongo_username}:{mongo_password}@{mongo_host}:{mongo_port}"

    return MongoClient(mongo_uri)


def delete_pending_tasks(
    db: Database, collections_to_process: List[str], allowed_models: List[str]
) -> None:
    """
    Удаляет записи со статусом 'pending' и моделями, не входящими в allowed_models,
    из указанных коллекций.

    Args:
        db (Database): Экземпляр базы данных MongoDB.
        collections_to_process (List[str]): Список коллекций для обработки.
        allowed_models (List[str]): Список допустимых моделей.

    Returns:
        None
    """
    # Создание фильтра для удаления: статус 'pending' и модель не в allowed_models
    query = {"model": {"$nin": allowed_models}}

    for collection_name in collections_to_process:
        collection: Collection = db[collection_name]

        # Удаление всех документов, соответствующих запросу
        result = collection.delete_many(query)

        # Вывод количества удаленных документов
        print(
            f"Из коллекции '{collection_name}' удалено {result.deleted_count} документов со статусом 'pending' и недопустимыми моделями."
        )


def main() -> None:
    """
    Основная функция для удаления определенных записей из указанных коллекций.
    """
    # Подключение к MongoDB
    client = get_mongo_client()
    db = client["TrustLLM_ru"]

    # Удаление записей
    delete_pending_tasks(db, COLLECTIONS_TO_PROCESS, MODELS)


if __name__ == "__main__":
    main()
