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
    mongo_uri = (
        f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
    )
    client = MongoClient(mongo_uri)
    logger.info("Успешно подключились к MongoDB.")
    return client


def get_db() -> Database:
    client = get_mongo_client()
    return client[MONGO_DB]


def fetch_rta_tasks(db: Database):
    """
    Функция-генератор: обходим все коллекции, имена которых начинаются с "queue_",
    и выбираем задачи с метрикой 'RtA' и статусом 'completed'.
    Возвращаем кортеж (coll_name, task).
    """
    collections = [c for c in db.list_collection_names() if c.startswith("queue_")]
    for coll_name in collections:
        coll = db[coll_name]
        tasks = list(coll.find({"metric": "RtA", "status": "completed"}))
        for t in tasks:
            yield coll_name, t


def create_rta_queue_entry(db: Database, coll_name: str, task: Dict[str, Any]) -> None:
    """
    Переносим задачу из обычной очереди (queue_{task_name}) в целевую rta очередь (rta_queue_{task_name}).
    Логика:
      - Из исходного имени очереди получаем task_name и формируем rta_queue_{task_name}.
      - В целевую запись копируются:
          init_model = оригинальная model,
          init_prompt = оригинальный prompt,
          prompt = rta_prompt из задачи,
          model = rta_model из задачи.
      - Формируется новое поле variables, в котором:
            "input"  = заполненный исходный prompt с подстановкой variables,
            "answer" = response.
      - Проверяются обязательные поля: rta_model, rta_prompt и response.
      - Если обнаружен дубликат (на основе rta_model и уже заполненных полей variables),
        запись не создается, а исходная помечается как ошибочная.
      - После успешного переноса исходная задача обновляется – её статус меняется на 'transfered_to_rta'.
      - В новую запись добавляется поле "source_id" для последующей синхронизации.
    """
    # Извлекаем task_name из coll_name: coll_name = "queue_{task_name}"
    task_name = coll_name.replace("queue_", "")
    rta_coll_name = f"queue_rta_{task_name}"
    rta_coll = db[rta_coll_name]

    # Достаем необходимые поля
    original_model = task["model"]  # исходная модель (init_model)
    rta_model = task.get("rta_model")
    if not rta_model:
        logger.warning("Задача RtA без rta_model? Пропускаем.")
        db[coll_name].update_one(
            {"_id": task["_id"]},
            {"$set": {"status": "error", "error": "Задача RtA без rta_model"}},
        )
        return

    original_prompt = task["prompt"]  # исходный prompt (init_prompt)
    rta_prompt = task.get("rta_prompt")
    if not rta_prompt:
        logger.warning("Задача RtA без rta_prompt? Пропускаем.")
        db[coll_name].update_one(
            {"_id": task["_id"]},
            {"$set": {"status": "error", "error": "Задача RtA без rta_prompt"}},
        )
        return

    variables = task.get("variables", {})
    response = task.get("response", "")
    if response is None:
        logger.warning("Задача RtA без response? Пропускаем.")
        db[coll_name].update_one(
            {"_id": task["_id"]},
            {"$set": {"status": "error", "error": "Задача RtA без response"}},
        )
        return

    # Формируем filled_input: подставляем variables в исходный prompt
    try:
        filled_input = original_prompt.format(**variables)
    except Exception as e:
        logger.error(f"Ошибка форматирования prompt: {e}")
        db[coll_name].update_one(
            {"_id": task["_id"]},
            {"$set": {"status": "error", "error": "Ошибка форматирования prompt"}},
        )
        return

    new_variables = {"input": filled_input, "answer": response}

    # Проверяем наличие дубликата в rta очереди (по rta_model и заполненным полям)
    existing = rta_coll.find_one(
        {
            "model": rta_model,
            "variables.input": filled_input,
            "variables.answer": response,
        }
    )
    if existing:
        logger.info("Дубликат найден, не добавляем запись в rta_queue.")
        db[coll_name].update_one(
            {"_id": task["_id"]},
            {"$set": {"status": "error", "error": "Дубликат в rta_queue"}},
        )
        return

    # Формируем новый документ для rta очереди, добавляя поле source_id для последующей синхронизации
    doc = {
        "task_name": task.get("task_name"),
        "dataset_name": task.get("dataset_name"),
        "init_prompt": original_prompt,
        "init_model": original_model,
        "regexp": task.get("regexp"),
        "prompt": rta_prompt,
        "model": rta_model,
        "variables": new_variables,
        "status": "pending",  # новая запись ожидает обработки
        "metric": "accuracy",  # согласно условию
        "target": task.get("target"),
        "source_id": task["_id"],  # ссылка на исходную запись в обычной очереди
    }

    # Вставляем в rta очередь
    rta_coll.insert_one(doc)
    logger.info(f"Задача RtA добавлена в {rta_coll_name}.")

    # Обновляем исходную задачу – меняем статус на 'transfered_to_rta'
    db[coll_name].update_one(
        {"_id": task["_id"]}, {"$set": {"status": "transfered_to_rta"}}
    )


def run_rta_transfer_loop(db: Database, interval: int = 10):
    """
    Бесконечный цикл:
      - Ищем задачи RtA (metric=RtA, status=completed) в обычных очередях и переносим их в rta очереди.
      - Затем выполняем синхронизацию: обновляем rta очереди на основании актуальных данных из обычных очередей.
      - Если нет задач для переноса, ждем указанное время.
    """
    while True:
        found_any = False
        for coll_name, task in fetch_rta_tasks(db):
            found_any = True
            create_rta_queue_entry(db, coll_name, task)

        if not found_any:
            logger.info("Нет задач RtA для переноса. Ожидание...")

        time.sleep(interval)


def main():
    db = get_db()
    run_rta_transfer_loop(db, interval=10)


if __name__ == "__main__":
    main()
