import logging
import os
import time

import requests
from dotenv import load_dotenv
from pymongo import MongoClient

from benchmark.constants import (
    API_URL,
    MODELS,
    MONGO_HOST,
    MONGO_PASSWORD,
    MONGO_PORT,
    MONGO_USERNAME,
)

# Формирование URI для подключения к MongoDB
mongo_uri = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"

# Подключение к MongoDB
client = MongoClient(mongo_uri)
db = client.TrustLLM_ru

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def make_request(model, prompt, variables, session):
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
        # Используем http_err.response для доступа к ответу
        response = http_err.response
        try:
            error_json = response.json()
        except ValueError:
            error_json = "No JSON response available"

        error_details = (
            f"HTTP error occurred: {http_err} - "
            f"Status Code: {response.status_code} - "
            f"Response: {response.text} - "
            f"Error JSON: {error_json}"
        )
        logging.error(error_details)
        raise Exception(error_details)

    except requests.exceptions.RequestException as req_err:
        logging.error(f"Request error occurred: {req_err}")
        raise Exception(f"Request error: {req_err}")


def process_task(task, collection, session):
    logging.info(f"Processing task with id: {task['_id']}")
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
            logging.info(f"Completed task with id: {task['_id']}")
        else:
            raise Exception("Failed to get a valid response from the API")
    except Exception as e:
        collection.update_one(
            {"_id": task["_id"]},
            {"$set": {"status": "failed", "error": str(e)}},
        )
        logging.error(f"Failed task with id: {task['_id']} - Error: {e}")


def run():
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

                # Обрабатываем задачи по моделям
                for model in MODELS:
                    logging.info(
                        f"Processing model '{model}' in collection '{collection_name}'"
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
                                f"No more pending tasks for model '{model}' in collection '{collection_name}'."
                            )
                            break  # Переходим к следующей модели, если задач нет

            # Все коллекции обработаны, ждем перед повторной проверкой
            logging.info("All collections processed, waiting for new tasks...")
            time.sleep(5)

        except Exception as e:
            logging.exception(f"An error occurred during processing: {e}")
            # Здесь можно решить, нужно ли прерывать цикл или продолжать
            # break


if __name__ == "__main__":
    run()
