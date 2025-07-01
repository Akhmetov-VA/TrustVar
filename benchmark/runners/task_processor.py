import logging
import os
import time
from typing import Any, Dict, List

import pandas as pd
from pymongo import MongoClient
from pymongo.database import Database

from utils.constants import MONGO_HOST, MONGO_PASSWORD, MONGO_PORT, MONGO_USERNAME

MONGO_DB = os.environ.get("MONGO_DB", "TrustGen")

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_mongo_client() -> MongoClient:
    """
    Создает подключение к MongoDB на основе переменных окружения.
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


def fetch_tasks(db: Database) -> List[Dict[str, Any]]:
    """
    Получаем все задачи из коллекции tasks.
    """
    tasks_coll = db["tasks"]
    tasks = list(tasks_coll.find({}))
    return tasks


def get_dataset_head(db: Database, dataset_name: str) -> pd.DataFrame:
    """
    Возвращаем датасет в формате DataFrame из коллекции dataset_<dataset_name>.
    """
    coll_name = f"dataset_{dataset_name}"
    coll = db[coll_name]
    docs = list(coll.find({}))
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])
    return df


def insert_queue_entries_for_task(db: Database, task: Dict[str, Any]) -> None:
    """
    Для задачи из коллекции tasks:
      - Создаем записи в очереди (коллекция queue_<task_name>) для каждой строки датасета и для каждой модели.
      - Если запись уже существует (определяется по паре (line_index, model)), она пропускается.
    """
    task_name = task["task_name"]
    dataset_name = task["dataset_name"]
    prompt_text = task["prompt"]
    var_cols = task.get("variables_cols", [])
    models = task["models"]
    metric = task["metric"]
    target = task.get("target", None)
    regexp = task.get("regexp", None)
    include_col = task.get("include_column", None)
    exclude_col = task.get("exclude_column", None)
    rta_prompt = task.get("rta_prompt")
    rta_model = task.get("rta_model")
    dynamic_augments = task.get('dynamic_augments', [])

    df = get_dataset_head(db, dataset_name)
    if df.empty:
        logger.warning(f"Датасет для '{dataset_name}' пуст. Нечего обрабатывать.")
        return

    queue_coll_name = f"queue_{task_name}"
    queue_coll = db[queue_coll_name]

    # Собираем ключи уже существующих записей (line_index, model)
    existing_keys = set()
    try:
        for entry in queue_coll.find({}, {"line_index": 1, "model": 1}):
            existing_keys.add((entry.get("line_index"), entry.get("model")))
    except Exception as e:
        logger.error(
            f"Ошибка при получении существующих записей из '{queue_coll_name}': {e}"
        )
        return

    new_inserts = []
    rows = df.to_dict("records")
    for i, row in enumerate(rows):
        variables = {col: row.get(col, None) for col in var_cols}
        for model in models:
            key = (i, model)
            if key in existing_keys:
                continue  # запись уже существует – пропускаем
            # Формируем новый документ
            doc = {
                "task_name": task_name,
                "line_index": i,
                "dataset_name": dataset_name,
                "prompt": prompt_text,
                "variables": variables,
                "model": model,
                "metric": metric,
                "regexp": regexp,
                #"status": "pending",
                "response": None,
            }
            if dynamic_augments:
                doc["status"] = "augmenting"
                doc["dynamic_augments"] = dynamic_augments
            else:
                doc["status"] = "pending"
            if metric == "RtA":
                if rta_prompt and rta_model:
                    doc["rta_prompt"] = rta_prompt
                    doc["rta_model"] = rta_model
                doc["target"] = target if isinstance(target, str) else metric
            elif metric == "include_exclude":
                if include_col and include_col in row:
                    value = row.get(include_col)
                    doc["include_list"] = [value] if isinstance(value, str) else value
                if exclude_col and exclude_col in row:
                    value = row.get(exclude_col)
                    doc["exclude_list"] = [value] if isinstance(value, str) else value
                doc["target"] = target if isinstance(target, str) else metric
            else:
                if target and target in row:
                    doc["target"] = row[target]
                else:
                    doc["target"] = None

            new_inserts.append(doc)

    if new_inserts:
        try:
            result = queue_coll.insert_many(new_inserts, ordered=False)
            logger.info(
                f"Вставлено {len(result.inserted_ids)} новых документов в '{queue_coll_name}'."
            )
        except Exception as e:
            logger.error(f"Ошибка при вставке документов в '{queue_coll_name}': {e}")
    else:
        logger.info(f"Нет новых документов для вставки в '{queue_coll_name}'.")


def main():
    """
    Основной цикл:
      - Подключаемся к БД.
      - Каждые N секунд перебираем задачи из коллекции tasks и для каждой вызываем функцию создания записей очереди.
    """
    db = get_db()
    interval = 10  # интервал в секундах

    try:
        while True:
            tasks = fetch_tasks(db)
            if tasks:
                for task in tasks:
                    logger.info(f"Обработка задачи: {task['task_name']}")
                    insert_queue_entries_for_task(db, task)
            else:
                logger.info("Нет задач для обработки.")
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Остановка процесса по KeyboardInterrupt")


if __name__ == "__main__":
    main()
