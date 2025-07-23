import json
import logging
import os
from typing import List

from pymongo import MongoClient

from utils.constants import (
    MONGO_DB,
    MONGO_HOST,
    MONGO_PASSWORD,
    MONGO_PORT,
    MONGO_USERNAME,
    TASK_NAMES,
)

logging.basicConfig(level=logging.INFO)


def get_mongo_client() -> MongoClient:
    """Подключение к локальной MongoDB"""
    logging.info("Attempting to connect to local MongoDB...")
    # Используем localhost для подключения с хоста
    # MongoDB работает на порту 27364 (как видно из docker ps)
    mongo_uri = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@localhost:27364/"
    logging.info(f"Connecting to: {mongo_uri}")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000, connectTimeoutMS=10000)
    client.admin.command("ping")
    logging.info("Connected to local MongoDB successfully.")
    return client


def upload_datasets_to_mongodb() -> None:
    """Загружает датасеты из JSON файлов в локальную MongoDB"""
    client = get_mongo_client()
    db = client[MONGO_DB]
    
    datasets_dir = "datasets"
    if not os.path.exists(datasets_dir):
        logging.error(f"Directory {datasets_dir} not found!")
        return
    
    # Получаем список всех JSON файлов в папке datasets
    json_files = [f for f in os.listdir(datasets_dir) if f.endswith('.json')]
    
    for json_file in json_files:
        file_path = os.path.join(datasets_dir, json_file)
        collection_name = json_file.replace('.json', '')
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            
            if not documents:
                logging.warning(f"File {json_file} is empty, skipping...")
                continue
            
            # Определяем имя коллекции в MongoDB
            if collection_name == "tasks":
                mongo_collection_name = "tasks"
            elif collection_name in ["Accuracy", "Accuracy_Groups", "Correlation", "IncludeExclude", "RtAR", "TFNR", "TFNR_Groups", "regexp_RtA", "regexp_accuracy", "regexp_correlation", "regexp_storage"]:
                # Метрики и regexp загружаем без префикса dataset_
                mongo_collection_name = collection_name
            else:
                # Для остальных файлов добавляем префикс dataset_
                mongo_collection_name = f"dataset_{collection_name}"
            
            # Очищаем коллекцию перед загрузкой
            db[mongo_collection_name].drop()
            
            # Загружаем документы
            if isinstance(documents, list):
                result = db[mongo_collection_name].insert_many(documents)
                logging.info(f"Uploaded {len(result.inserted_ids)} documents to collection '{mongo_collection_name}' from {json_file}")
            else:
                result = db[mongo_collection_name].insert_one(documents)
                logging.info(f"Uploaded 1 document to collection '{mongo_collection_name}' from {json_file}")
                
        except Exception as e:
            logging.error(f"Error uploading {json_file}: {e}")
            continue
    
    logging.info("Datasets upload completed!")


if __name__ == "__main__":
    try:
        upload_datasets_to_mongodb()
        logging.info("All datasets successfully uploaded to local MongoDB!")
    except Exception as e:
        logging.error(f"Failed to upload datasets: {e}") 