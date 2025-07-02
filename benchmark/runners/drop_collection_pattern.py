import os
from typing import List

from dotenv import load_dotenv
from pymongo import MongoClient


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


def delete_collections_by_pattern(db, pattern: str) -> None:
    """
    Удаляет коллекции из базы данных MongoDB, названия которых начинаются с заданного паттерна.

    Args:
        db: Экземпляр базы данных MongoDB.
        pattern (str): Паттерн, с которого начинаются названия коллекций.

    Returns:
        None
    """
    # Получение списка всех коллекций в базе данных
    collections = db.list_collection_names()

    # Фильтрация коллекций, начинающихся с заданного паттерна
    collections_to_delete = [col for col in collections if col.startswith(pattern)]

    # Удаление найденных коллекций
    for collection_name in collections_to_delete:
        db.drop_collection(collection_name)
        print(f"Коллекция '{collection_name}' успешно удалена.")

    print(f"Все коллекции, начинающиеся с '{pattern}', были удалены.")


def main() -> None:
    """
    Основная функция для удаления коллекций, начинающихся с определенного паттерна.
    """
    # Паттерн для фильтрации коллекций
    pattern = "rubia_"

    # Подключение к MongoDB
    client = get_mongo_client()
    db = client["TrustLLM_ru"]

    # Удаление коллекций по паттерну
    delete_collections_by_pattern(db, pattern)


if __name__ == "__main__":
    main()
