import os
from typing import List

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database


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
    mongo_uri = (
        f"mongodb://{mongo_username}:{mongo_password}@{mongo_host}:{mongo_port}/"
    )

    return MongoClient(mongo_uri)


def delete_pending_tasks(
    db: Database, excluded_collections: List[str], query: dict
) -> None:
    """
    Удаляет задачи со статусом 'pending' из всех коллекций базы данных,
    кроме указанных в списке исключений.

    Args:
        db (Database): Экземпляр базы данных MongoDB.
        excluded_collections (List[str]): Список коллекций, которые не нужно очищать.
        query (dict): Условие для поиска задач (например, {"status": "pending"}).

    Returns:
        None
    """
    # Перебор всех коллекций в базе данных
    for collection_name in db.list_collection_names():
        if collection_name in excluded_collections:
            continue  # Пропускаем коллекции из списка исключений

        collection: Collection = db[collection_name]

        # Удаление всех задач, соответствующих запросу
        result = collection.delete_many(query)

        # Вывод количества удаленных задач
        print(
            f"Из коллекции '{collection_name}' удалено {result.deleted_count} задач со статусом 'pending'."
        )


def main() -> None:
    """
    Основная функция для удаления задач со статусом 'pending' из всех коллекций,
    кроме исключенных.
    """
    # Подключение к MongoDB
    client = get_mongo_client()
    db = client["TrustLLM_ru"]

    # Список исключаемых коллекций
    excluded_collections = ["delete_me", "test"]

    # Условие для поиска задач со статусом 'pending'
    query = {"status": "pending"}

    # Удаление задач
    delete_pending_tasks(db, excluded_collections, query)


if __name__ == "__main__":
    main()
