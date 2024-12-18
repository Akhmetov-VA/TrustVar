import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from bson.objectid import ObjectId
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

    # ---------------- Дополнительные методы ----------------
    def get_all_tasks(self) -> pd.DataFrame:
        tasks_collection = self.get_collection("tasks")
        tasks = list(tasks_collection.find({}))
        if not tasks:
            return pd.DataFrame()
        return pd.DataFrame(tasks)

    def get_all_datasets(self) -> List[str]:
        """Получить список всех датасетов (те, что имеют префикс dataset_)."""
        collections = self.list_collections()
        dataset_colls = [col for col in collections if col.startswith("dataset_")]
        datasets = [col.replace("dataset_", "") for col in dataset_colls]
        return datasets

    def get_dataset_head(self, dataset_name: str, limit: int = 10) -> pd.DataFrame:
        coll = self.get_collection(f"dataset_{dataset_name}")
        docs = list(coll.find({}).limit(limit))
        if not docs:
            return pd.DataFrame()
        df = pd.DataFrame(docs)
        if "_id" in df.columns:
            df = df.drop(columns=["_id"])
        return df

    def get_prompt_docs_for_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Получить полный список промптов для датасета (name, prompt)."""
        coll_name = f"prompt_{dataset_name}"
        if coll_name not in self.list_collections():
            return []
        coll = self.get_collection(coll_name)
        return list(coll.find({}))

    def get_rta_prompt_docs(self) -> List[Dict[str, Any]]:
        """Получить полный список RTA промптов (name, prompt)."""
        if "prompt_rta" not in self.list_collections():
            return []
        coll = self.get_collection("prompt_rta")
        return list(coll.find({}))

    def get_regexp_docs_for_metric(self, metric: str) -> List[Dict[str, Any]]:
        """Получить полный список регулярок для метрики (name, pattern)."""
        coll_name = f"regexp_{metric}"
        if coll_name not in self.list_collections():
            return []
        coll = self.get_collection(coll_name)
        return list(coll.find({}))

    def get_prompts_for_dataset(self, dataset_name: str) -> List[str]:
        """Получить список имен промптов для датасета."""
        prompts = self.get_prompt_docs_for_dataset(dataset_name)
        return [p["name"] for p in prompts if "name" in p]

    def get_rta_prompts(self) -> List[str]:
        """Получить список имен RTA промптов."""
        rta_prompts = self.get_rta_prompt_docs()
        return [rp["name"] for rp in rta_prompts if "name" in rp]

    def insert_prompt_for_dataset(self, dataset_name: str, prompt: str, name: str):
        """Вставить новый промпт для датасета."""
        coll = self.get_collection(f"prompt_{dataset_name}")
        coll.insert_one({"name": name, "prompt": prompt})

    def insert_rta_prompt(self, prompt: str, name: str):
        """Вставить новый RTA промпт."""
        coll = self.get_collection("prompt_rta")
        coll.insert_one({"name": name, "prompt": prompt})

    def insert_regexp_for_metric(self, metric: str, pattern: str, name: str):
        """Вставить новую регулярку для метрики."""
        coll = self.get_collection(f"regexp_{metric}")
        coll.insert_one({"name": name, "pattern": pattern})

    def insert_task(self, task_data: Dict[str, Any]):
        """Вставить новую задачу."""
        coll = self.get_collection("tasks")
        coll.insert_one(task_data)

    def update_task(self, task_id, update_data: Dict[str, Any]):
        """Обновить существующую задачу."""
        coll = self.get_collection("tasks")
        if not isinstance(task_id, ObjectId):
            try:
                task_id = ObjectId(task_id)
            except Exception as e:
                logger.error(f"Некорректный ID задачи: {e}")
                raise
        coll.update_one({"_id": task_id}, {"$set": update_data})

    def validate_regex(self, pattern: str) -> bool:
        """Проверка корректности регулярного выражения."""
        try:
            re.compile(pattern)
            return True
        except re.error:
            return False

    def get_regexp_for_metric(self, metric: str) -> List[str]:
        """Получить список имен регулярок для метрики."""
        docs = self.get_regexp_docs_for_metric(metric)
        return [d["name"] for d in docs if "name" in d]

    def list_metrics(self) -> List[str]:
        """Получить список метрик, основываясь на префиксах results_ или regexp_."""
        regexp_cols = self.list_collections_starting_with("regexp_")
        result_cols = self.list_collections_starting_with("results_")
        metrics_regexp = [c.replace("regexp_", "") for c in regexp_cols]
        metrics_results = [c.replace("results_", "") for c in result_cols]
        metrics = list(set(metrics_regexp + metrics_results))
        return metrics

    def insert_dataset_into_registry(
        self,
        dataset_name: str,
        var_cols: List[str],
        metric: str,
        target_column: Optional[str] = None,
    ):
        """Сохранить информацию о датасете в dataset_regestry."""
        coll_name = "dataset_regestry"
        if "dataset_regestry" not in self.list_collections():
            pass  # Коллекция создастся автоматически при вставке
        coll = self.get_collection(coll_name)
        doc = coll.find_one({"dataset_name": dataset_name})
        data = {"var_cols": var_cols, "metric": metric}
        if target_column:
            data["target_column"] = target_column
        if doc:
            coll.update_one({"dataset_name": dataset_name}, {"$set": data})
        else:
            data["dataset_name"] = dataset_name
            coll.insert_one(data)

    def get_dataset_registry_info(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Получить информацию о датасете из registry."""
        coll = self.get_collection("dataset_regestry")
        doc = coll.find_one({"dataset_name": dataset_name})
        return doc

    def insert_dataset_records(self, dataset_name: str, df: pd.DataFrame):
        """Загрузить датасет в коллекцию dataset_{dataset_name}."""
        coll_name = f"dataset_{dataset_name}"
        records = df.to_dict(orient="records")
        if records:
            self.insert_data(coll_name, records)
