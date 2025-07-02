import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Union

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
    mongo_uri = (
        f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
    )
    client = MongoClient(mongo_uri)
    logger.info("Успешно подключились к MongoDB.")
    return client


def get_db() -> Database:
    """Возвращает объект базы данных MongoDB."""
    client = get_mongo_client()
    return client[MONGO_DB]


def fetch_completed_tasks(db: Database):
    """
    Находим все задачи в очередях queue_* со статусом 'completed' и наличием поля response.
    Исключаем метрику RtA, т.к. она обрабатывается другим скриптом.

    Возвращаем итератор (coll_name, task).
    """
    collections = [c for c in db.list_collection_names() if c.startswith("queue_")]
    for coll_name in collections:
        coll = db[coll_name]
        # metric != 'RtA'
        tasks = list(
            coll.find(
                {
                    "status": "completed",
                    "response": {"$ne": None},
                    "metric": {"$ne": "RtA"},
                }
            )
        )
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
        # Предполагается, что берем первую подходящую группу.
        if match.groups():
            for g in match.groups():
                if g is not None:
                    return g
            return match.group(0)
        else:
            return match.group(0)
    else:
        return "TFN"


def apply_exact_match(response: str, target: Union[str, List[str]]) -> str:
    """
    Для метрики exact_match:
    Если target - список строк, проверяем каждую.
    Если хоть одна найдена в response, она включается в pred.
    Если target - одна строка (не список), делаем её списком из одного элемента.
    Если ничего не найдено - pred='TFN'.
    """
    if isinstance(target, str):
        target = [target]  # Превращаем строку в список

    found = []
    for t in target:
        if t in response:
            found.append(t)
    if not found:
        return "TFN"
    else:
        # Вернем список найденных строк (или, например, через запятую).
        # Для удобства пусть будет просто список в виде string.
        return str(found)


def update_task_with_pred(db: Database, coll_name: str, task_id: Any, pred: str):
    """
    Обновляем в задаче поле pred и статус на extracted.
    """
    coll = db[coll_name]
    coll.update_one({"_id": task_id}, {"$set": {"pred": pred, "status": "extracted"}})
    logger.info(
        f"Обновлен документ {task_id} в {coll_name}: pred={pred}, status=extracted"
    )


def run_extraction_loop(db: Database, interval: int = 10):
    """
    Запускаем бесконечный цикл опроса очередей:
    - Находим все задачи в статусе completed (response != None) и metric != RtA
    - В зависимости от metric:
       1) exact_match: используем apply_exact_match
       2) include_exclude: просто берем response в pred
       3) любые другие: используем regexp (если есть) -> apply_regexp_to_response
         если нет - TFN
    - Меняем статус на extracted
    - Ждем interval секунд и повторяем
    """
    while True:
        found_any = False
        for coll_name, task in fetch_completed_tasks(db):
            found_any = True
            task_id = task["_id"]
            response = task["response"]
            metric = task.get("metric", None)
            target = task.get("target", [])

            if metric == "exact_match":
                # exact_match логика
                pred = apply_exact_match(response, target)

            elif metric == "include_exclude":
                # По условию "просто берем response и переносим в pred"
                # Логику проверки include/exclude выполняет следующий ранер.
                pred = response

            else:
                # Любая другая метрика -> regexp
                regexp = task.get("regexp", None)
                if not regexp:
                    pred = "TFN"
                else:
                    pred = apply_regexp_to_response(response, regexp)

            update_task_with_pred(db, coll_name, task_id, pred)

        if not found_any:
            logger.info("Нет задач для извлечения pred. Ожидание...")
        time.sleep(interval)


def main():
    """
    Точка входа:
    1) Подключаемся к базе
    2) Запускаем цикл обработки
    """
    db = get_db()
    run_extraction_loop(db, interval=60)


if __name__ == "__main__":
    main()
