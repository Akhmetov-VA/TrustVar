import os
import time

import requests

# Загрузка переменных окружения из .env файла
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# Получение данных для подключения из переменных окружения
MONGO_USERNAME = os.getenv("MONGO_INITDB_ROOT_USERNAME")
MONGO_PASSWORD = os.getenv("MONGO_INITDB_ROOT_PASSWORD")
MONGO_HOST = os.getenv("MONGO_HOST")
MONGO_PORT = os.getenv("MONGO_INITDB_ROOT_PORT")

API_URL = os.getenv("API_URL")

# Формирование URI для подключения к MongoDB
mongo_uri = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"

# Подключение к MongoDB
client = MongoClient(mongo_uri)

# Выбор базы данных и коллекции
db = client.TrustLLM_ru


def make_request(model, prompt, variables):
    try:
        response = requests.post(
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
        # Capture the JSON response if available
        try:
            error_json = (
                response.json()
            )  # This might throw another exception if response is not in JSON
        except ValueError:
            error_json = "No JSON response available"

        error_details = f"HTTP error occurred: {http_err} - Status Code: {response.status_code} - Response: {response.text} - Error JSON: {error_json}"
        print(error_details)
        raise Exception(error_details)  # Raise with detailed error info

    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
        raise Exception(f"Request error: {req_err}")


def process_task(task, collection):
    print(f"Processing task with id: {task['_id']}")
    prompt = task["prompt"]
    prompt = task["prompt"]
    model = task["model"]
    variables = task["variables"]

    try:
        response = make_request(model, prompt, variables)
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
            print(f"Completed task with id: {task['_id']}")
        else:
            raise Exception("Failed to get a valid response from the API")
    except Exception as e:
        collection.update_one(
            {"_id": task["_id"]},
            {"$set": {"status": "failed", "error": str(e)}},
        )
        print(f"Failed task with id: {task['_id']} - Error: {e}")


def run():
    while True:
        # Get all collections except for the ones to skip
        collections_to_process = [
            col
            for col in db.list_collection_names()
            if col not in ["delete_me", "test"]
        ]

        for collection_name in collections_to_process:
            collection = db[collection_name]

            while True:
                # Find one pending task
                task = collection.find_one({"status": "pending"})

                if task:
                    process_task(task, collection)
                else:
                    print(
                        f"No more pending tasks in collection {collection_name}. Moving to the next collection."
                    )
                    break  # Move to the next collection when there are no more pending tasks

        # All collections have been processed, wait before checking again
        print("All collections processed, waiting for new tasks...")
        time.sleep(5)


if __name__ == "__main__":
    run()
