import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

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


def convert_numpy_objects(data):
    """Recursively convert numpy objects to Python native types."""
    if isinstance(data, np.ndarray):
        return data.tolist()  # Convert numpy array to list
    elif isinstance(data, dict):
        return {key: convert_numpy_objects(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_objects(item) for item in data]
    elif isinstance(data, np.generic):
        return data.item()  # Convert numpy scalars to Python scalars
    else:
        return data

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
        """
        При создании экземпляра MongoDBConfig можно переопределить параметры или
        они будут взяты из переменных окружения.
        """
        self.username = username or os.getenv("MONGO_INITDB_ROOT_USERNAME")
        self.password = password or os.getenv("MONGO_INITDB_ROOT_PASSWORD")
        self.host = host or os.getenv("MONGO_HOST", "localhost")
        self.port = port or os.getenv("MONGO_INITDB_ROOT_PORT", "27017")
        self.database = database

    def get_uri(self) -> str:
        """
        Формирование URI для подключения к MongoDB.
        """
        return f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/"


class MongoDBClient:
    """Класс для высокоуровневой работы с MongoDB."""

    def __init__(self, config: Optional[MongoDBConfig] = None):
        """
        Инициализация клиента MongoDB с помощью заданного или дефолтного конфига.
        """
        self.config = config or MongoDBConfig()
        try:
            self.client = MongoClient(self.config.get_uri())
            self.db = self.client[self.config.database]
            logger.info("Успешно подключились к MongoDB.")
        except PyMongoError as e:
            logger.error(f"Ошибка подключения к MongoDB: {e}")
            raise

    # ---------------- Основные методы для работы с коллекциями ----------------

    def get_collection(self, collection_name: str) -> Collection:
        """Получение коллекции по имени."""
        return self.db[collection_name]

    def list_collections(self) -> List[str]:
        """Получение списка всех коллекций."""
        return self.db.list_collection_names()

    def list_collections_starting_with(self, prefix: str) -> List[str]:
        """Получение списка коллекций, начинающихся с заданного префикса."""
        return [col for col in self.list_collections() if col.startswith(prefix)]

    def delete_collection(self, collection_name: str) -> None:
        """Удаление коллекции по имени."""
        try:
            self.db.drop_collection(collection_name)
            logger.info(f"Коллекция '{collection_name}' успешно удалена.")
        except PyMongoError as e:
            logger.error(f"Ошибка удаления коллекции '{collection_name}': {e}")
            raise

    # ---------------- Методы для вставки и обновления документов ----------------

    def insert_data(self, collection_name: str, data: List[Dict[str, Any]]) -> None:
        """
        Вставка нескольких документов в указанную коллекцию.
        Используется, например, для загрузки датасета.
        """
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

    def insert_task(self, task_data: Dict[str, Any]):
        """
        Вставить новую задачу в коллекцию tasks.
        """
        coll = self.get_collection("tasks")
        coll.insert_one(task_data)

    def insert_prompt_for_dataset(self, dataset_name: str, prompt: str, name: str):
        """
        Вставить новый промпт (prompt, name) в коллекцию prompt_{dataset_name}.
        """
        coll = self.get_collection(f"prompt_{dataset_name}")
        coll.insert_one({"name": name, "prompt": prompt})

    def insert_rta_prompt(self, prompt: str, name: str):
        """
        Вставить новый RTA-промпт в prompt_rta.
        """
        coll = self.get_collection("prompt_rta")
        coll.insert_one({"name": name, "prompt": prompt})

    def insert_regexp_for_metric(self, metric: str, pattern: str, name: str):
        """
        Вставить новую регулярку (name, pattern) в коллекцию regexp_{metric}.
        """
        coll = self.get_collection(f"regexp_{metric}")
        coll.insert_one({"name": name, "pattern": pattern})

    def insert_dataset_records(self, dataset_name: str, df: pd.DataFrame):
        """
        Загрузить датасет в коллекцию dataset_{dataset_name}.
        """
        coll_name = f"dataset_{dataset_name}"
        
        import numpy as np





        records = df.to_dict(orient="records")
        converted_records = [convert_numpy_objects(record) for record in records]
        if records:
            self.insert_data(coll_name, records)

    def insert_dataset_into_registry(self, doc: Dict[str, Any]):
        """
        Добавить информацию о датасете в dataset_regestry.
        Ожидается, что doc уже содержит поля:
            dataset_name, var_cols, metric, target_column, include_column, exclude_column ...
        """
        coll_name = "dataset_regestry"
        coll = self.get_collection(coll_name)
        coll.insert_one(doc)

    def update_task(self, task_id: Any, update_data: Dict[str, Any]):
        """
        Обновить существующую задачу в коллекции tasks по _id.
        """
        coll = self.get_collection("tasks")
        if not isinstance(task_id, ObjectId):
            try:
                task_id = ObjectId(task_id)
            except Exception as e:
                logger.error(f"Некорректный ID задачи: {e}")
                raise
        coll.update_one({"_id": task_id}, {"$set": update_data})

    def update_tasks_status(self, collection_name: str, current_status: str, new_status: str) -> int:
        """
        Обновление статуса задач в collection_name: current_status -> new_status.
        Возвращает количество обновлённых документов.
        """
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

    # ---------------- Методы для подсчётов и получения документов ----------------

    def get_tasks_by_status(self, collection_name: str, status: str) -> List[Dict[str, Any]]:
        """
        Получение всех задач из collection_name, у которых status == status.
        """
        try:
            collection = self.get_collection(collection_name)
            tasks = list(collection.find({"status": status}))
            logger.info(f"Найдено {len(tasks)} задач со статусом '{status}' в '{collection_name}'.")
            return tasks
        except PyMongoError as e:
            logger.error(f"Ошибка получения задач по статусу: {e}")
            raise

    def count_tasks_by_status(self, collection_name: str, status: str) -> int:
        """
        Подсчёт количества задач со статусом status в collection_name.
        """
        try:
            collection = self.get_collection(collection_name)
            count = collection.count_documents({"status": status})
            logger.info(f"Количество задач со статусом '{status}' в '{collection_name}': {count}.")
            return count
        except PyMongoError as e:
            logger.error(f"Ошибка подсчёта задач по статусу: {e}")
            raise

    def count_total_tasks(self, collection_name: str) -> int:
        """
        Подсчёт общего количества документов (задач) в collection_name.
        """
        try:
            collection = self.get_collection(collection_name)
            count = collection.count_documents({})
            logger.info(
                f"Общее количество задач в коллекции '{collection_name}': {count}."
            )
            return count
        except PyMongoError as e:
            logger.error(f"Ошибка подсчёта общего количества задач: {e}")
            raise

    # ---------------- Методы для чтения данных (tasks, datasets, prompts, regexp) ----------------

    def get_all_tasks(self) -> pd.DataFrame:
        """
        Возвращает все задачи из коллекции tasks в виде DataFrame.
        Если задач нет, возвращает пустой DataFrame.
        """
        tasks_collection = self.get_collection("tasks")
        tasks = list(tasks_collection.find({}))
        if not tasks:
            return pd.DataFrame()
        return pd.DataFrame(tasks)

    def get_all_datasets(self) -> List[str]:
        """
        Возвращает список всех датасетов (названия),
        основываясь на коллекциях, начинающихся с 'dataset_'.
        """
        collections = self.list_collections()
        dataset_colls = [col for col in collections if col.startswith("dataset_")]
        datasets = [col.replace("dataset_", "") for col in dataset_colls]
        return datasets

    def get_dataset_head(self, dataset_name: str, limit: int = 10) -> pd.DataFrame:
        """
        Возвращает первые 'limit' строк датасета dataset_{dataset_name} в виде DataFrame.
        """
        coll = self.get_collection(f"dataset_{dataset_name}")
        docs = list(coll.find({}).limit(limit))
        if not docs:
            return pd.DataFrame()
        df = pd.DataFrame(docs)
        if "_id" in df.columns:
            df = df.drop(columns=["_id"])
        return df

    def get_dataset_registry_info(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Получить информацию о датасете (из 'dataset_regestry')
        по его имени dataset_name.
        """
        coll = self.get_collection("dataset_regestry")
        doc = coll.find_one({"dataset_name": dataset_name})
        return doc

    def get_prompt_docs_for_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """
        Получить полный список (documents) промптов для dataset_{dataset_name},
        т.е. коллекция prompt_{dataset_name}.
        """
        coll_name = f"prompt_{dataset_name}"
        if coll_name not in self.list_collections():
            return []
        coll = self.get_collection(coll_name)
        return list(coll.find({}))

    def get_rta_prompt_docs(self) -> List[Dict[str, Any]]:
        """
        Получить полный список RTA-промптов (documents) из коллекции prompt_rta (если есть).
        """
        if "prompt_rta" not in self.list_collections():
            return []
        coll = self.get_collection("prompt_rta")
        return list(coll.find({}))

    def get_prompts_for_dataset(self, dataset_name: str) -> List[str]:
        """
        Вернуть список имён промптов (name) для указанного датасета.
        """
        prompts = self.get_prompt_docs_for_dataset(dataset_name)
        return [p["name"] for p in prompts if "name" in p]

    def get_rta_prompts(self) -> List[str]:
        """
        Вернуть список имен RTA-промптов (name) из prompt_rta.
        """
        rta_prompts = self.get_rta_prompt_docs()
        return [rp["name"] for rp in rta_prompts if "name" in rp]

    def get_regexp_docs_for_metric(self, metric: str) -> List[Dict[str, Any]]:
        """
        Получить полный список документов (name, pattern) из regexp_{metric}.
        """
        coll_name = f"regexp_{metric}"
        if coll_name not in self.list_collections():
            return []
        coll = self.get_collection(coll_name)
        return list(coll.find({}))

    def get_regexp_for_metric(self, metric: str) -> List[str]:
        """
        Получить список имён регулярок (name) для заданной метрики (regexp_{metric}).
        """
        docs = self.get_regexp_docs_for_metric(metric)
        return [d["name"] for d in docs if "name" in d]

    def validate_regex(self, pattern: str) -> bool:
        """
        Проверить корректность регулярного выражения.
        """
        try:
            re.compile(pattern)
            return True
        except re.error:
            return False

    def list_metrics(self) -> List[str]:
        """
        Получить список метрик, основываясь на коллекциях, начинающихся с regexp_ или results_.
        """
        regexp_cols = self.list_collections_starting_with("regexp_")
        result_cols = self.list_collections_starting_with("results_")
        metrics_regexp = [c.replace("regexp_", "") for c in regexp_cols]
        metrics_results = [c.replace("results_", "") for c in result_cols]
        metrics = list(set(metrics_regexp + metrics_results))
        return metrics
