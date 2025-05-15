import logging
import os
import time
from typing import Any, Dict

import requests
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from utils.constants import (
    API_URL,
    MONGO_HOST,
    MONGO_PASSWORD,
    MONGO_PORT,
    MONGO_USERNAME,
)


def configure_logging() -> None:
    """
    Настраивает логирование для отображения сообщений в консоли.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.info("Логирование успешно настроено.")


def get_mongo_client() -> MongoClient:
    """
    Создает подключение к MongoDB на основе переменных окружения.

    Returns:
        MongoClient: Экземпляр MongoDB клиента.
    """
    logging.info("Попытка подключения к MongoDB...")
    mongo_uri = (
        f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
    )
    try:
        client = MongoClient(mongo_uri)
        logging.info("Успешно подключились к MongoDB.")
        return client
    except Exception as e:
        logging.exception("Ошибка подключения к MongoDB.")
        raise e


def make_request(
    model: str, prompt: str, variables: Dict[str, Any], session: requests.Session
) -> Dict:
    """
    Отправляет POST-запрос к API с указанной моделью, промптом и переменными.

    Args:
        model (str): Имя модели.
        prompt (str): Текст запроса.
        variables (Dict[str, Any]): Переменные для запроса.
        session (requests.Session): Сессия requests для повторного использования соединений.

    Returns:
        Dict: JSON-ответ от API.

    Raises:
        Exception: Если запрос не удался или ответ некорректный.
    """
    logging.info(
        f"Отправка запроса к API для модели '{model}' с промптом: {prompt[:100]}..."
    )
    try:
        response = session.post(
            API_URL,
            json={
                "model": model,
                "stream": False,
                "prompt": prompt,
                "variables": variables,
            },
        )
        response.raise_for_status()
        logging.info(f"Успешный ответ от API для модели '{model}'.")
        if response.json() is None:
            raise Exception("null response")
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Ошибка при выполнении запроса к API для модели '{model}': {e}")
        raise e


def process_task(task: Dict, collection: Collection, session: requests.Session) -> None:
    """
    Обрабатывает отдельную задачу, отправляя запрос к модели и обновляя статус задачи в базе данных.

    Args:
        task (Dict): Документ задачи из MongoDB.
        collection (Collection): Коллекция MongoDB, содержащая задачи.
        session (requests.Session): Сессия requests для повторного использования соединений.
    """
    task_id = task["_id"]
    logging.info(f"Начало обработки задачи с id: {task_id}")
    prompt = task["prompt"]
    model = task["model"]
    variables = task.get("variables", {})
    try:
        response = make_request(model, prompt, variables, session)
        collection.update_one(
            {"_id": task_id},
            {"$set": {"status": "completed", "response": response}},
        )
        logging.info(
            f"Задача с id: {task_id} успешно завершена и обновлена в базе данных."
        )
    except Exception as e:
        collection.update_one(
            {"_id": task_id},
            {"$set": {"status": "error", "error": str(e)}},
        )
        logging.error(f"Ошибка обработки задачи с id: {task_id}: {e}")


def process_collection(
    db: Database, collection_name: str, session: requests.Session
) -> None:
    """
    Обрабатывает задачи в указанной коллекции.

    Args:
        db (Database): Экземпляр базы данных MongoDB.
        collection_name (str): Название коллекции.
        session (requests.Session): Сессия requests для повторного использования соединений.
    """
    logging.info(f"Начало обработки коллекции '{collection_name}'.")
    collection = db[collection_name]
    unique_models = collection.distinct("model")

    if not unique_models:
        logging.warning(
            f"В коллекции '{collection_name}' отсутствуют модели для обработки."
        )
        return

    logging.info(
        f"Найдено {len(unique_models)} уникальных моделей в коллекции '{collection_name}'."
    )
    for model in unique_models:
        logging.info(
            f"Обработка задач для модели '{model}' в коллекции '{collection_name}'."
        )
        while True:
            task = collection.find_one_and_update(
                {"status": "pending", "model": model},
                {"$set": {"status": "processing"}},
                return_document=False,
            )
            if task:
                logging.info(f"Найдена задача с id: {task['_id']} для обработки.")
                process_task(task, collection, session)
            else:
                logging.info(
                    f"Нет ожидающих задач для модели '{model}' в коллекции '{collection_name}'."
                )
                break


def run_processing_loop(db: Database) -> None:
    """
    Запускает цикл обработки задач во всех коллекциях.

    Args:
        db (Database): Экземпляр базы данных MongoDB.
    """
    logging.info("Запуск основного цикла обработки задач.")
    session = requests.Session()

    try:
        collections_to_process = [
            col
            for col in db.list_collection_names()
            if col not in ["delete_me", "test"]
        ]

        logging.info(f"Найдено {len(collections_to_process)} коллекций для обработки.")
        for collection_name in collections_to_process:
            process_collection(db, collection_name, session)

        logging.info("Все коллекции обработаны. Ожидание новых задач...")
        time.sleep(5)
    except Exception as e:
        logging.exception(f"Ошибка в процессе обработки: {e}")


def main() -> None:
    """
    Основная функция для запуска обработки задач в MongoDB.
    """
    configure_logging()
    logging.info("Загрузка переменных окружения и инициализация подключения...")
    client = get_mongo_client()
    db_name = "TrustGen"
    while True:
        db = client[db_name]
        run_processing_loop(db)
        time.sleep(10)


if __name__ == "__main__":
    main()
