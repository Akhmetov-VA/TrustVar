import logging
import os
import time

import requests
from dotenv import load_dotenv
from pymongo import MongoClient

from benchmark.constants import (
    API_URL,
    # MODELS,  # Удален импорт MODELS, так как будем использовать модели из коллекции
    MONGO_HOST,
    MONGO_PASSWORD,
    MONGO_PORT,
    MONGO_USERNAME,
)

# Загрузка переменных окружения из .env файла, если необходимо
load_dotenv()

# Формирование URI для подключения к MongoDB
mongo_uri = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"

# Подключение к MongoDB
client = MongoClient(mongo_uri)
db = client.TrustLLM_ru

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def make_request(model, prompt, variables, session):
    """
    Отправляет POST-запрос к API с заданной моделью, промптом и переменными.

    :param model: Имя модели
    :param prompt: Текст запроса
    :param variables: Дополнительные переменные для запроса
    :param session: Сессия requests для повторного использования соединений
    :return: JSON-ответ от API
    """
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
        return response.json()

    except requests.exceptions.HTTPError as http_err:
        # Обработка HTTP-ошибок
        response = http_err.response
        try:
            error_json = response.json()
        except ValueError:
            error_json = "Нет доступного JSON-ответа"

        error_details = (
            f"Произошла HTTP-ошибка: {http_err} - "
            f"Код состояния: {response.status_code} - "
            f"Ответ: {response.text} - "
            f"Ошибка JSON: {error_json}"
        )
        logging.error(error_details)
        raise Exception(error_details)

    except requests.exceptions.RequestException as req_err:
        # Обработка других ошибок запроса
        logging.error(f"Произошла ошибка запроса: {req_err}")
        raise Exception(f"Ошибка запроса: {req_err}")


def process_task(task, collection, session):
    """
    Обрабатывает отдельную задачу из коллекции.
    Отправляет запрос к модели и обновляет статус задачи в базе данных.

    :param task: Документ задачи из MongoDB
    :param collection: Коллекция MongoDB, содержащая задачи
    :param session: Сессия requests для повторного использования соединений
    """
    logging.info(f"Обработка задачи с id: {task['_id']}")
    prompt = task["prompt"]
    model = task["model"]
    variables = task.get("variables", {})

    try:
        response = make_request(model, prompt, variables, session)
        if response:
            collection.update_one(
                {"_id": task["_id"]},
                {
                    "$set": {
                        "status": "completed",
                        "response": response,
                    }
                },
            )
            logging.info(f"Задача с id: {task['_id']} завершена")
        else:
            raise Exception("Не удалось получить допустимый ответ от API")
    except Exception as e:
        collection.update_one(
            {"_id": task["_id"]},
            {"$set": {"status": "failed", "error": str(e)}},
        )
        logging.error(f"Не удалось обработать задачу с id: {task['_id']} - Ошибка: {e}")


def run():
    """
    Основная функция, запускающая бесконечный цикл обработки задач во всех коллекциях.
    Вместо использования списка MODELS, получает все уникальные модели из каждой коллекции.
    """
    session = requests.Session()
    while True:
        try:
            # Получаем все коллекции, кроме тех, которые нужно пропустить
            collections_to_process = [
                col
                for col in db.list_collection_names()
                if col not in ["delete_me", "test"]
            ]

            for collection_name in collections_to_process:
                collection = db[collection_name]

                # Получаем список уникальных моделей из текущей коллекции
                unique_models = collection.distinct("model")
                if not unique_models:
                    logging.info(
                        f"В коллекции '{collection_name}' нет моделей для обработки."
                    )
                    continue

                for model in unique_models:
                    logging.info(
                        f"Обработка модели '{model}' в коллекции '{collection_name}'"
                    )
                    while True:
                        # Атомарно находим одну задачу с указанной моделью и статусом 'pending'
                        task = collection.find_one_and_update(
                            {"status": "pending", "model": model},
                            {"$set": {"status": "processing"}},
                            return_document=False,
                        )

                        if task:
                            process_task(task, collection, session)
                        else:
                            logging.info(
                                f"Нет ожидающих задач для модели '{model}' в коллекции '{collection_name}'."
                            )
                            break  # Переходим к следующей модели, если задач нет

            # Все коллекции обработаны, ждем перед повторной проверкой
            logging.info("Все коллекции обработаны, ожидание новых задач...")
            time.sleep(5)

        except Exception as e:
            logging.exception(f"Произошла ошибка во время обработки: {e}")
            # В случае ошибки, можно решить, продолжать цикл или прервать
            # Здесь продолжаем цикл после ожидания
            time.sleep(60)


if __name__ == "__main__":
    run()
