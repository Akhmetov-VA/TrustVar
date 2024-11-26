import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

# Загрузка переменных окружения из .env файла
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MongoDBConfig:
    """Класс для управления конфигурациями MongoDB."""

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[str] = None,
        database: str = "TrustLLM_ru",
    ):
        self.username = username or os.getenv("MONGO_INITDB_ROOT_USERNAME")
        self.password = password or os.getenv("MONGO_INITDB_ROOT_PASSWORD")
        self.host = host or os.getenv("MONGO_HOST", "localhost")
        self.port = port or os.getenv("MONGO_INITDB_ROOT_PORT", "27017")
        self.database = database

    def get_uri(self) -> str:
        """Формирование URI для подключения к MongoDB."""
        return f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/"


class MongoDBClient:
    """Класс для работы с MongoDB."""

    def __init__(self, config: Optional[MongoDBConfig] = None):
        self.config = config or MongoDBConfig()
        try:
            self.client = MongoClient(self.config.get_uri())
            self.db = self.client[self.config.database]
            logger.info("Успешно подключились к MongoDB.")
        except PyMongoError as e:
            logger.error(f"Ошибка подключения к MongoDB: {e}")
            raise

    def get_collection(self, collection_name: str) -> Collection:
        """Получение коллекции по имени."""
        return self.db[collection_name]

    def list_collections(self) -> List[str]:
        """Получение списка всех коллекций."""
        return self.db.list_collection_names()

    def list_collections_starting_with(self, prefix: str) -> List[str]:
        """Получение списка коллекций, начинающихся с заданного префикса."""
        return [col for col in self.list_collections() if col.startswith(prefix)]

    def insert_data(self, collection_name: str, data: List[Dict[str, Any]]) -> None:
        """Вставка нескольких документов в коллекцию."""
        if not data:
            logger.warning("Нет данных для вставки.")
            return
        try:
            collection = self.get_collection(collection_name)
            collection.insert_many(data, ordered=False)
            logger.info(
                f"Вставлено {len(data)} документов в коллекцию '{collection_name}'."
            )
        except PyMongoError as e:
            logger.error(f"Ошибка вставки данных в MongoDB: {e}")
            raise

    def update_tasks_status(
        self, collection_name: str, current_status: str, new_status: str
    ) -> int:
        """Обновление статуса задач."""
        try:
            collection = self.get_collection(collection_name)
            result = collection.update_many(
                {"status": current_status}, {"$set": {"status": new_status}}
            )
            logger.info(
                f"Обновлено {result.modified_count} документов из статуса '{current_status}' на '{new_status}'."
            )
            return result.modified_count
        except PyMongoError as e:
            logger.error(f"Ошибка обновления статуса задач: {e}")
            raise

    def get_tasks_by_status(
        self, collection_name: str, status: str
    ) -> List[Dict[str, Any]]:
        """Получение задач по статусу."""
        try:
            collection = self.get_collection(collection_name)
            tasks = list(collection.find({"status": status}))
            logger.info(f"Найдено {len(tasks)} задач со статусом '{status}'.")
            return tasks
        except PyMongoError as e:
            logger.error(f"Ошибка получения задач по статусу: {e}")
            raise

    def count_tasks_by_status(self, collection_name: str, status: str) -> int:
        """Подсчет количества задач по статусу."""
        try:
            collection = self.get_collection(collection_name)
            count = collection.count_documents({"status": status})
            logger.info(f"Количество задач со статусом '{status}': {count}.")
            return count
        except PyMongoError as e:
            logger.error(f"Ошибка подсчета задач по статусу: {e}")
            raise

    def count_total_tasks(self, collection_name: str) -> int:
        """Подсчет общего количества задач в коллекции."""
        try:
            collection = self.get_collection(collection_name)
            count = collection.count_documents({})
            logger.info(
                f"Общее количество задач в коллекции '{collection_name}': {count}."
            )
            return count
        except PyMongoError as e:
            logger.error(f"Ошибка подсчета общего количества задач: {e}")
            raise

    def delete_collection(self, collection_name: str) -> None:
        """Удаление коллекции."""
        try:
            self.db.drop_collection(collection_name)
            logger.info(f"Коллекция '{collection_name}' успешно удалена.")
        except PyMongoError as e:
            logger.error(f"Ошибка удаления коллекции '{collection_name}': {e}")
            raise
