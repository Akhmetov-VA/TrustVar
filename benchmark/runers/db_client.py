import logging
import os

from dotenv import load_dotenv
from pymongo import MongoClient

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

# Константы
load_dotenv()
MONGO_USERNAME = os.getenv("MONGO_INITDB_ROOT_USERNAME")
MONGO_PASSWORD = os.getenv("MONGO_INITDB_ROOT_PASSWORD")
MONGO_HOST = os.getenv("MONGO_HOST")
MONGO_PORT = os.getenv("MONGO_INITDB_ROOT_PORT")

# Формирование URI для подключения к MongoDB
MONGO_URI = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"


class DBClient:
    """
    Класс для управления подключением к базе данных MongoDB.
    """

    def __init__(self, uri=MONGO_URI, db_name=None):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]

    def get_collection(self, collection_name):
        """
        Получает коллекцию из базы данных.
        """
        return self.db[collection_name]
