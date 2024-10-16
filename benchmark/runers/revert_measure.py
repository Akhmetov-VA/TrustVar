import os

from dotenv import load_dotenv
from pymongo import MongoClient

# Загрузка переменных окружения из .env файла
load_dotenv()

MONGO_USERNAME = os.getenv("MONGO_INITDB_ROOT_USERNAME")
MONGO_PASSWORD = os.getenv("MONGO_INITDB_ROOT_PASSWORD")
MONGO_HOST = os.getenv("MONGO_HOST")
MONGO_PORT = os.getenv("MONGO_INITDB_ROOT_PORT")

# Формирование URI для подключения к MongoDB
mongo_uri = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
client = MongoClient(mongo_uri)
db = client["TrustLLM_ru"]

# Определяем запрос для поиска всех документов со статусом "measured"
query = {"status": "measured"}

for collection_name in db.list_collection_names():
    # Пропускаем служебные или ненужные коллекции
    if collection_name in ["delete_me", "test", "results", "top_questions"]:
        continue
    collection = db[collection_name]
    # Обновляем статус "measured" на "completed"
    result = collection.update_many(query, {"$set": {"status": "completed"}})

    # Выводим количество обновленных документов
    print(
        f"Обновлено {result.modified_count} документов в коллекции '{collection_name}'."
    )
