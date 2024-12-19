import logging
import os
import time
from typing import Any, Dict, List

import pandas as pd
from bson.objectid import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database

from utils.constants import MONGO_HOST, MONGO_PASSWORD, MONGO_PORT, MONGO_USERNAME

MONGO_DB = os.environ.get("MONGO_DB", "TrustGen")

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_mongo_client() -> MongoClient:
    """
    Создает подключение к MongoDB на основе переменных окружения.
    """
    mongo_uri = (
        f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
    )
    client = MongoClient(mongo_uri)
    logger.info("Успешно подключились к MongoDB.")
    return client


def get_db() -> Database:
    client = get_mongo_client()
    db = client[MONGO_DB]
    return db


def fetch_tasks(db: Database) -> List[Dict[str, Any]]:
    """
    Получаем все задачи из коллекции tasks.
    Предполагается, что здесь можно отфильтровать по статусу, если нужно.
    В текущей версии возвращаем все.
    """
    tasks_coll = db["tasks"]
    new_tasks = list(tasks_coll.find({}))
    return new_tasks


def get_dataset_head(db: Database, dataset_name: str) -> pd.DataFrame:
    """
    Возвращаем весь датасет в формате DataFrame.
    """
    coll_name = f"dataset_{dataset_name}"
    coll = db[coll_name]
    docs = list(coll.find({}))
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])
    return df


def create_queue_entries_for_task(db: Database, task: Dict[str, Any]) -> None:
    """
    На основе задачи из таблицы tasks создаем записи в коллекции queue_{task_name}.
    """
    task_name = task["task_name"]
    dataset_name = task["dataset_name"]
    prompt_text = task["prompt"]
    var_cols = task.get("variables_cols", [])
    models = task["models"]
    metric = task["metric"]
    target = task["target"]

    df = get_dataset_head(db, dataset_name)
    if df.empty:
        logger.warning(f"Датасет для {dataset_name} пуст. Нечего добавлять в очередь.")
        return

    queue_coll_name = f"queue_{task_name}"
    queue_coll = db[queue_coll_name]

    operations = []
    for i, row in df.iterrows():
        variables = {}
        for c in var_cols:
            variables[c] = row[c] if c in row else None

        for model in models:
            existing = queue_coll.find_one({"model": model, "variables": variables})
            if existing:
                continue

            doc = {
                "line_index": i,
                "prompt": prompt_text,
                "variables": variables,
                "model": model,
                "metric": metric,
                "target": row[target] if target != "RtA" else "RtA",
                "status": "pending",
                "response": None,
            }

            if metric == "RtA":
                rta_prompt = task.get("rta_prompt")
                rta_model = task.get("rta_model")
                if rta_prompt and rta_model:
                    doc["rta_prompt"] = rta_prompt
                    doc["rta_model"] = rta_model

            operations.append(doc)

    if operations:
        queue_coll.insert_many(operations)
        logger.info(f"Вставлено {len(operations)} документов в {queue_coll_name}.")
    else:
        logger.info(f"Нет новых документов для добавления в {queue_coll_name}.")


def update_task_status(db: Database, task: Dict[str, Any], new_status: str) -> None:
    """
    Обновление статуса задачи в коллекции tasks.
    """
    tasks_coll = db["tasks"]
    tasks_coll.update_one({"_id": task["_id"]}, {"$set": {"status": new_status}})
    logger.info(f"Статус задачи {task['task_name']} обновлен на {new_status}.")


def delete_unused_queues(db: Database) -> None:
    """
    Удаляет таблицы queue_{} которые есть в базе, но которых нет в tasks.
    Логика:
    - Получаем список всех коллекций, начинающихся с queue_
    - Получаем список всех task_name из tasks
    - Если queue_{task_name} не соответствует ни одной задаче из tasks, удаляем ее
    """
    all_collections = db.list_collection_names()
    queue_collections = [c for c in all_collections if c.startswith("queue_")]

    # Получаем все task_name из tasks
    tasks_coll = db["tasks"]
    all_tasks = list(tasks_coll.find({}, {"task_name": 1}))
    existing_tasks = {t["task_name"] for t in all_tasks if "task_name" in t}

    for q_col in queue_collections:
        # q_col в формате queue_{task_name}, надо извлечь task_name
        task_name = q_col.replace("queue_", "")
        if (
            f"task_{task_name}" not in existing_tasks
            and task_name not in existing_tasks
        ):
            # Возможно в tasks task_name уже хранится с префиксом task_
            # Проверим оба варианта
            # Обычно task_name уже содержит префикс task_ согласно коду выше?
            # Если в tasks мы храним без префикса, то нужно подстроиться.
            # Посмотрим на код, выше создаем задачи как "task_{task_name}".
            # Это значит, что в tasks у нас task_name уже с "task_" впереди.
            # Тогда нужно привести к одному формату:
            # queue_collections названы queue_{task_name}, где task_name уже в формате "task_..."
            # Значит task_name из q_col уже task_...
            # Тогда нам не нужна лишняя проверка.
            # Сразу проверим: if task_name not in existing_tasks:
            if task_name not in existing_tasks:
                # Очередь не соответствует ни одной задаче
                db.drop_collection(q_col)
                logger.info(f"Удалена коллекция: {q_col}")


def main():
    """
    Основной цикл:
    - Подключаемся к БД
    - Каждые N секунд просматриваем tasks
    - Для каждой задачи создаем соответствующие записи в queue_{task_name} (если не созданы)
    - Вызываем функцию удаления неиспользуемых очередей
    """
    db = get_db()
    interval = 60  # 60 секунд, можно изменить

    while True:
        tasks = fetch_tasks(db)
        if tasks:
            for task in tasks:
                create_queue_entries_for_task(db, task)
        else:
            logger.info("Нет новых задач для создания очередей.")

        # Удаляем неиспользуемые очереди
        delete_unused_queues(db)

        time.sleep(interval)


if __name__ == "__main__":
    main()
