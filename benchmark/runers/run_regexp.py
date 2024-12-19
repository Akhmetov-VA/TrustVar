import logging
import os
import re
import time
from typing import Any, Dict, Optional

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database

from utils.constants import MONGO_USERNAME, MONGO_PASSWORD, MONGO_HOST, MONGO_PORT

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


def fetch_completed_tasks(db: Database):
    """
    Находим все задачи в очередях queue_* со статусом 'completed' и наличием поля response.
    Возвращаем итератор по таким задачам.
    """
    # Ищем все коллекции, начинающиеся на queue_
    collections = [c for c in db.list_collection_names() if c.startswith("queue_")]
    for coll_name in collections:
        coll = db[coll_name]
        # Выберем все задачи со статусом completed и response
        # Можно выбрать по одному, потом обновить статус, затем брать следующий
        # или просто все сразу
        tasks = list(coll.find({"status": "completed", "response": {"$ne": None}, "metric": {"$ne": "RtA"}}))
        for t in tasks:
            yield coll_name, t


def apply_regexp_to_response(response: str, regexp: str) -> str:
    """
    Применяем регулярку к response.
    Если находит совпадение — берем найденное значение.
    Если нет — 'TFN'.
    """
    pattern = re.compile(regexp, re.DOTALL)
    match = pattern.search(response)
    if match:
        # Предполагается, что берем первую группу, если есть группы
        # Если групп нет, то берем весь match.
        # Исходя из условия, вероятно берем первую группу, если есть.
        if match.groups():
            # Возьмем первую непустую группу
            for g in match.groups():
                if g is not None:
                    return g
            # Если все группы None, берем просто match.group(0)
            return match.group(0)
        else:
            return match.group(0)
    else:
        return "TFN"


def update_task_with_pred(db: Database, coll_name: str, task_id: Any, pred: str):
    """
    Обновляем в задаче поле pred и статус на extracted.
    """
    coll = db[coll_name]
    coll.update_one({"_id": task_id}, {"$set": {"pred": pred, "status": "extracted"}})
    logger.info(f"Обновлен документ {task_id} в {coll_name}: pred={pred}, status=extracted")


def run_extraction_loop(db: Database, interval: int = 10):
    """
    Запускаем бесконечный цикл опроса очередей.
    Каждые interval секунд смотрим, есть ли задачи для обработки:
    - Находим все completed задачи с response
    - Для каждой применяем regexp из задачи (task['regexp'])
    - Сохраняем результат в pred
    - Меняем статус на extracted
    """
    while True:
        found_any = False
        for coll_name, task in fetch_completed_tasks(db):
            found_any = True
            task_id = task["_id"]
            response = task["response"]
            regexp = task.get("regexp", None)

            if not regexp:
                # Если нет регулярки - не можем извлечь pred
                # Можно поставить pred='TFN' или пропустить
                # Но по условию метрика опирается на regexp, лучше TFN
                pred = "TFN"
            else:
                pred = apply_regexp_to_response(response, regexp)

            update_task_with_pred(db, coll_name, task_id, pred)

        if not found_any:
            logger.info("Нет задач для извлечения pred. Ожидание...")
        time.sleep(interval)


def main():
    db = get_db()
    run_extraction_loop(db, interval=60)


if __name__ == "__main__":
    main()
