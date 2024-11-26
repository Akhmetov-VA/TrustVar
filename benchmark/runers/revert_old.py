import logging
import os

from dotenv import load_dotenv
from pymongo import MongoClient

# Загрузка переменных окружения из .env файла
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

# Параметры подключения к MongoDB
MONGO_USERNAME = os.getenv("MONGO_INITDB_ROOT_USERNAME")
MONGO_PASSWORD = os.getenv("MONGO_INITDB_ROOT_PASSWORD")
MONGO_HOST = os.getenv("MONGO_HOST")
MONGO_PORT = os.getenv("MONGO_INITDB_ROOT_PORT")
DATABASE_NAME = "TrustLLM_ru"


def revert_task_status(collection):
    """
    Отменяет статус 'measured' на 'completed' и удаляет поля 'pred' и 'metric' для задач в коллекции.
    """
    try:
        result = collection.update_many(
            {"status": "transferred"},
            {"$set": {"status": "completed"}, "$unset": {"pred": "", "metric": ""}},
        )
        logging.info(
            f"Отменено {result.modified_count} задач из 'measured' в 'completed' в коллекции '{collection.name}'."
        )
    except Exception as e:
        logging.error(f"Ошибка при отмене задач в коллекции '{collection.name}': {e}")


def main():
    try:
        # Формирование URI для подключения к MongoDB
        mongo_uri = (
            f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
        )
        client = MongoClient(mongo_uri)
        db = client[DATABASE_NAME]

        for collection_name in db.list_collection_names():
            collection = db[collection_name]
            logging.info(f"Обработка коллекции '{collection_name}'")
            revert_task_status(collection)

        logging.info("Все указанные коллекции обработаны.")

    except Exception as e:
        logging.exception(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    main()
