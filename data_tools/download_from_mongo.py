import json
import logging
import os
from typing import List

from pymongo import MongoClient

from utils.constants import (
    MONGO_HOST,
    MONGO_PASSWORD,
    MONGO_PORT,
    MONGO_USERNAME,
    MONGO_SOURCE_URI,
    MONGO_SOURCE_DB_NAME,
    TASK_NAMES,
)

# Функция для очистки имени от суффиксов
def clean_task_name(task_name: str) -> str:
    """Убирает _emnlp из любого места в имени задачи"""
    return task_name.replace('_emnlp', '')

# Автоматическая генерация имён коллекций на основе префикса 'dataset_'
# Используем оригинальные имена для поиска в MongoDB
datasets_name: List[str] = [f"dataset_{name}" for name in TASK_NAMES]

# Добавляем коллекции tasks и dataset_regestry для выгрузки
datasets_name.append("tasks")
datasets_name.append("dataset_regestry")

# Добавляем коллекции с промптами, regexp и метриками
datasets_name.extend([
    "prompt_storage",
    "regexp_RtA",
    "regexp_accuracy", 
    "regexp_correlation",
    "regexp_storage",
    "Accuracy",
    "Accuracy_Groups",
    "Correlation",
    "IncludeExclude",
    "RtAR",
    "TFNR",
    "TFNR_Groups"
])


def get_mongo_client(uri: str = None) -> MongoClient:
    logging.info("Attempting to connect to MongoDB...")
    if uri is None:
        mongo_uri = (
            f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
        )
    else:
        mongo_uri = uri
    logging.info(f"Connecting to: {mongo_uri}")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000, connectTimeoutMS=10000)
    client.admin.command("ping")
    logging.info("Connected to MongoDB successfully.")
    return client


logging.basicConfig(level=logging.INFO)


def dump_datasets_to_files() -> None:
    # Подключаемся к source MongoDB
    client = get_mongo_client(MONGO_SOURCE_URI)
    db = client[MONGO_SOURCE_DB_NAME]

    # Добавляем все коллекции с суффиксом _Groups, если они есть
    all_collections = db.list_collection_names()
    groups_collections = [col for col in all_collections if col.endswith('_Groups')]
    for group_col in groups_collections:
        if group_col not in datasets_name:
            datasets_name.append(group_col)
            logging.info(f"Added groups collection: {group_col}")

    os.makedirs("datasets", exist_ok=True)

    for collection_name in datasets_name:
        if collection_name not in db.list_collection_names():
            logging.warning(f"Collection {collection_name} not found in MongoDB.")
            continue

        # Специальная обработка для коллекции tasks
        if collection_name == "tasks":
            # Фильтруем задачи только по тем датасетам, которые есть в списке TASK_NAMES
            documents = list(db[collection_name].find(
                {"dataset_name": {"$in": TASK_NAMES}}, 
                {"_id": 0}
            ))
            if not documents:
                logging.warning(f"Collection {collection_name} has no tasks for specified datasets.")
                continue
            
            # Очищаем dataset_name и task_name от суффиксов в tasks
            for doc in documents:
                if "dataset_name" in doc:
                    original_dataset = doc["dataset_name"]
                    doc["dataset_name"] = clean_task_name(doc["dataset_name"])
                    if original_dataset != doc["dataset_name"]:
                        logging.info(f"Cleaned dataset_name: {original_dataset} -> {doc['dataset_name']}")
                if "task_name" in doc:
                    original_task = doc["task_name"]
                    doc["task_name"] = clean_task_name(doc["task_name"])
                    if original_task != doc["task_name"]:
                        logging.info(f"Cleaned task_name: {original_task} -> {doc['task_name']}")
            
            logging.info(f"Filtered and cleaned {len(documents)} tasks for specified datasets")
        elif collection_name == "dataset_regestry":
            # Фильтруем записи только по тем датасетам, которые есть в списке TASK_NAMES
            documents = list(db[collection_name].find(
                {"dataset_name": {"$in": TASK_NAMES}}, 
                {"_id": 0}
            ))
            if not documents:
                logging.warning(f"Collection {collection_name} has no records for specified datasets.")
                continue
            
            # Очищаем dataset_name от суффиксов в dataset_registry
            for doc in documents:
                if "dataset_name" in doc:
                    original_dataset = doc["dataset_name"]
                    doc["dataset_name"] = clean_task_name(doc["dataset_name"])
                    if original_dataset != doc["dataset_name"]:
                        logging.info(f"Cleaned dataset_name in registry: {original_dataset} -> {doc['dataset_name']}")
            
            logging.info(f"Filtered and cleaned {len(documents)} records in dataset_registry for specified datasets")
        else:
            documents = list(db[collection_name].find({}, {"_id": 0}))
            if not documents:
                logging.warning(f"Collection {collection_name} is empty.")
                continue

        # Определяем имя файла для сохранения
        if collection_name == "tasks":
            # Для tasks сохраняем как есть
            file_name = f"{collection_name}.json"
        else:
            # Для dataset_* коллекций убираем префикс dataset_ и суффиксы
            task_name = collection_name.replace("dataset_", "")
            cleaned_name = clean_task_name(task_name)
            file_name = f"{cleaned_name}.json"

        json_path = os.path.join("datasets", file_name)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)
            logging.info(f"Saved {len(documents)} docs to {json_path}")


if __name__ == "__main__":
    try:
        dump_datasets_to_files()
        logging.info("Datasets successfully dumped to ./datasets/")
    except Exception as e:
        logging.error(f"Failed to dump datasets: {e}")
