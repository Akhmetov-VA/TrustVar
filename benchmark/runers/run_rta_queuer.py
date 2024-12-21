import logging
import os
import time
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database

from utils.constants import MONGO_HOST, MONGO_PASSWORD, MONGO_PORT, MONGO_USERNAME

# Предполагается, что переменные окружения для MONGO_USERNAME, MONGO_PASSWORD, MONGO_HOST, MONGO_PORT, MONGO_DB уже заданы
MONGO_DB = os.environ.get("MONGO_DB", "TrustGen")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_mongo_client() -> MongoClient:
    """
    Создаем подключение к MongoDB.
    """
    mongo_uri = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
    client = MongoClient(mongo_uri)
    logger.info("Успешно подключились к MongoDB.")
    return client


def get_db() -> Database:
    client = get_mongo_client()
    return client[MONGO_DB]


def fetch_rta_tasks(db: Database):
    """
    Обходим все очереди (queue_*) и ищем задачи с метрикой 'RtA' и статусом 'completed'.
    Возвращаем итератор (coll_name, doc) для каждой такой задачи.
    """
    collections = [c for c in db.list_collection_names() if c.startswith("queue_")]
    for coll_name in collections:
        # ищем все задачи метрика=RtA и status=completed
        coll = db[coll_name]
        tasks = list(coll.find({"metric": "RtA", "status": "completed"}))
        for t in tasks:
            yield coll_name, t


def create_rta_queue_entry(db: Database, coll_name: str, task: Dict[str, Any]) -> None:
    """
    Переносим задачу в rta_queue_{task_name}:
    Логика:
    - Новая коллекция rta_queue_{task_name}, где task_name из coll_name: coll_name = queue_{task_name}, 
      значит rta_coll_name = rta_queue_{task_name}.
    - model -> init_model = оригинальная model 
      а model = rta_model из task
    - prompt -> init_prompt = старый prompt, prompt = rta_prompt из task
    - variables_new = {"input": filled_prompt, "answer": response}
    - metric = 'accuracy'
    - Проверить дубликаты (например, по line_index и model)
    - После переноса status исходной меняем на 'transfered_to_rta'
    """
    # Извлекаем task_name из coll_name: coll_name = queue_{task_name}
    # Значит task_name = coll_name.replace("queue_", "")
    task_name = coll_name.replace("queue_", "")
    rta_coll_name = f"queue_rta_{task_name}"
    rta_coll = db[rta_coll_name]

    # Достаем поля из task
    original_model = task["model"]  # init_model
    rta_model = task.get("rta_model")
    if not rta_model:
        logger.warning("Задача RtA без rta_model? Пропускаем.")
        db[coll_name].update_one({"_id": task["_id"]}, {"$set": {"status": "error", 'error': 'Задача RtA без rta_model'}})
        return

    original_prompt = task["prompt"]   # init_prompt
    rta_prompt = task.get("rta_prompt")
    if not rta_prompt:
        logger.warning("Задача RtA без rta_prompt? Пропускаем.")
        db[coll_name].update_one({"_id": task["_id"]}, {"$set": {"status": "error", 'error': 'Задача RtA без rta_prompt'}})
        return

    variables = task.get("variables", {})
    
    response = task.get("response", "")
    if response is None:
        logger.warning("Задача RtA без response? Пропускаем.")
        db[coll_name].update_one({"_id": task["_id"]}, {"$set": {"status": "error", 'error': 'Задача RtA без response'}})
        return

    # Формируем input - подстановка variables в init_prompt (original_prompt)
    filled_input = original_prompt.format(**variables)
    new_variables = {
        "input": filled_input,
        "answer": response
    }

    # Проверяем дубликат: например, по model=rta_model и line_index
    # или можно по input+answer, но надежнее по line_index и model
    existing = rta_coll.find_one({"model": rta_model, "variables.input": filled_input, "variables.answer": response})

    if existing:
        logger.info("Дубликат найден, не добавляем запись в rta_queue.")
        # Меняем статус исходной на transfered_to_rta, чтобы не повторять в будущем
        db[coll_name].update_one({"_id": task["_id"]}, {"$set": {"status": "error", 'error': 'Дубликат в rta_queue'}})
        return

    # Формируем новый документ
    doc = {
        'task_name': task.get('task_name', None),
        'dataset_name': task.get('dataset_name', None),
        "init_prompt": original_prompt,
        "init_model": original_model,
        "regexp": task.get('regexp', None),
        "prompt": rta_prompt,
        "model": rta_model,
        "variables": new_variables,
        "status": "pending",   # новая запись ожидает обработки
        "metric": "accuracy",  # по условию
        'target': task.get('target', None)
    }


    # Вставляем в rta_queue
    rta_coll.insert_one(doc)
    logger.info(f"Задача RtA добавлена в {rta_coll_name}.")

    # Меняем статус исходной задачи на transfered_to_rta
    db[coll_name].update_one({"_id": task["_id"]}, {"$set": {"status": "transfered_to_rta"}})


def run_rta_transfer_loop(db: Database, interval: int = 10):
    """
    Запускаем бесконечный цикл:
    - Ищем задачи RtA (metric=RtA, status=completed) в очередях
    - Переносим их в rta_queue_{} с учетом вышеописанной логики
    """
    while True:
        found_any = False
        for coll_name, task in fetch_rta_tasks(db):
            found_any = True
            create_rta_queue_entry(db, coll_name, task)

        if not found_any:
            logger.info("Нет задач RtA для переноса. Ожидание...")
        time.sleep(interval)


def fetch_rta_tasks(db: Database):
    """
    Функция-генератор: обходим все queue_* коллекции и ищем задачи:
    metric=RtA, status=completed
    """
    collections = [c for c in db.list_collection_names() if c.startswith("queue_")]
    for coll_name in collections:
        coll = db[coll_name]
        tasks = list(coll.find({"metric": "RtA", "status": "completed"}))
        for t in tasks:
            yield coll_name, t


def main():
    db = get_db()
    run_rta_transfer_loop(db, interval=10)


if __name__ == "__main__":
    main()
