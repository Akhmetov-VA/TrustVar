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

API_URL = "http://localhost:27361/generate/ollama"

# Формирование URI для подключения к MongoDB
mongo_uri = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"

# Подключение к MongoDB
client = MongoClient(mongo_uri)

# Выбор базы данных и коллекции
db = client.TrustLLM_ru
collection = db.test


# Функция для отправки запроса к модели
def make_request(model, prompt, variables):
    response = requests.post(
        API_URL,
        json={
            "model": model,  ###меняем на свое название модели####
            "stream": False,  ###чтобы был весь ответ сразу а не по 1 токену###
            "prompt": prompt,  ###наш промпт###
            "variables": variables,
        },
    )
    # print(response.text)
    response.raise_for_status()
    return response.json()  # .get("response")


def run():
    while True:
        for collection_name in db.list_collection_names():
            if collection_name in ["delete_me", "test"]:
                continue
            collection = db[collection_name]
            task = collection.find_one(
                {"status": "pending"},  # {"$set": {"status": "processing"}}
            )
            if task:
                print(f"Processing task with id: {task['_id']}")
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
            else:
                print("No pending tasks. Waiting for new tasks...")
                time.sleep(5)


if __name__ == "__main__":
    run()
