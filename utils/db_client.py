import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()


class MongoDBClient:
    """Класс для работы с MongoDB."""

    def __init__(self):
        # Получение данных для подключения из .env файла
        MONGO_USERNAME = os.getenv("MONGO_INITDB_ROOT_USERNAME")
        MONGO_PASSWORD = os.getenv("MONGO_INITDB_ROOT_PASSWORD")
        MONGO_HOST = os.getenv("MONGO_HOST")
        MONGO_PORT = os.getenv("MONGO_INITDB_ROOT_PORT")

        # Формирование URI для подключения к MongoDB
        mongo_uri = (
            f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
        )

        # Подключение к MongoDB
        self.client = MongoClient(mongo_uri)

        # Выбор базы данных
        self.db = self.client.TrustLLM_ru

    def get_collection(self, collection_name):
        return self.db[collection_name]

    def list_collections(self):
        return self.db.list_collection_names()

    def list_collections_starting_with(self, prefix):
        return [
            col for col in self.db.list_collection_names() if col.startswith(prefix)
        ]

    def insert_data(self, collection_name, data):
        collection = self.get_collection(collection_name)
        if data:
            collection.insert_many(data)

    def update_tasks_status(self, collection_name, current_status, new_status):
        collection = self.get_collection(collection_name)
        result = collection.update_many(
            {"status": current_status}, {"$set": {"status": new_status}}
        )
        return result.modified_count

    def get_tasks_by_status(self, collection_name, status):
        collection = self.get_collection(collection_name)
        return list(collection.find({"status": status}))

    def count_tasks_by_status(self, collection_name, status):
        collection = self.get_collection(collection_name)
        return collection.count_documents({"status": status})

    def count_total_tasks(self, collection_name):
        collection = self.get_collection(collection_name)
        return collection.count_documents({})

    # Метод для удаления коллекции
    def delete_collection(self, collection_name):
        self.db.drop_collection(collection_name)
