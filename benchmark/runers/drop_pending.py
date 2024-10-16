import os

from dotenv import load_dotenv
from pymongo import MongoClient

# Загрузка переменных окружения из .env файла
load_dotenv()

MONGO_USERNAME = os.getenv("MONGO_INITDB_ROOT_USERNAME")
MONGO_PASSWORD = os.getenv("MONGO_INITDB_ROOT_PASSWORD")
MONGO_HOST = os.getenv("MONGO_HOST")
MONGO_PORT = os.getenv("MONGO_INITDB_ROOT_PORT")

mongo_uri = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
client = MongoClient(mongo_uri)
db = client["TrustLLM_ru"]

# Define the query to find all pending tasks
query = {"status": "pending"}

for collection_name in db.list_collection_names():
    if collection_name in ["delete_me", "test"]:
        continue
    collection = db[collection_name]
    # Delete all pending tasks
    result = collection.delete_many(query)

    # Output the number of tasks deleted
    print(f"Deleted {result.deleted_count} pending tasks.")
