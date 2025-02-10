#!/usr/bin/env python3
import logging
import os
import time
from typing import Dict, Tuple

import pandas as pd
from pymongo import DeleteOne, InsertOne, MongoClient, UpdateOne
from pymongo.database import Database

from utils.constants import MONGO_HOST, MONGO_PASSWORD, MONGO_PORT, MONGO_USERNAME

MONGO_DB = os.environ.get("MONGO_DB", "TrustGen")

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_mongo_client() -> MongoClient:
    mongo_uri = (
        f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
    )
    client = MongoClient(mongo_uri)
    logger.info("Подключились к MongoDB (sync_queues).")
    return client


def get_db() -> Database:
    client = get_mongo_client()
    return client[MONGO_DB]


def get_dataset_head(db: Database, dataset_name: str) -> pd.DataFrame:
    """
    Загружает документы из коллекции dataset_<dataset_name> и возвращает DataFrame.
    Если данных нет, возвращается пустой DataFrame.
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


def compute_expected_queue_entries(
    task: dict, df: pd.DataFrame
) -> Dict[Tuple[int, str], dict]:
    """
    Для данного задания и датасета вычисляет, какие документы должны быть в коллекции queue_{task_name}.
    Ключ – (line_index, model).
    """
    expected = {}
    task_name = task["task_name"]
    dataset_name = task["dataset_name"]
    prompt_text = task["prompt"]
    var_cols = task.get("variables_cols", [])
    models = task["models"]
    metric = task["metric"]
    regexp = task.get("regexp")
    target = task.get("target", None)
    rta_prompt = task.get("rta_prompt")
    rta_model = task.get("rta_model")
    include_col = task.get("include_column")
    exclude_col = task.get("exclude_column")

    rows = df.to_dict("records")
    for i, row in enumerate(rows):
        # Формируем переменные из указанных колонок
        variables = {col: row.get(col) for col in var_cols}
        for model in models:
            doc = {
                "line_index": i,
                "dataset_name": dataset_name,
                "prompt": prompt_text,
                "variables": variables,
                "model": model,
                "metric": metric,
                "regexp": regexp,
                "response": None,
                "status": "pending",  # при синхронизации core-поля меняются – статус сбрасывается
            }
            if metric == "RtA":
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
            expected[(i, model)] = doc
    return expected


def synchronize_task_queue(db: Database, task: dict):
    """
    Синхронизирует основную очередь для задания:
      – Если изменились core‑поля ([prompt, variables_cols, models]), то производится полный апдейт (status -> pending).
      – Если изменились только rta‑поля ([rta_prompt, rta_model, target, regexp]), то обновляются только эти поля.
    """
    task_name = task["task_name"]
    dataset_name = task["dataset_name"]
    queue_coll_name = f"queue_{task_name}"
    queue_coll = db[queue_coll_name]

    df = get_dataset_head(db, dataset_name)
    if df.empty:
        logger.warning(
            f"Датасет '{dataset_name}' пуст – пропускаем задание '{task_name}'."
        )
        return

    expected_entries = compute_expected_queue_entries(task, df)
    expected_keys = set(expected_entries.keys())

    # Получаем текущие записи очереди, ключом будет (line_index, model)
    existing_entries = {}
    for doc in queue_coll.find(
        {},
        {
            "line_index": 1,
            "model": 1,
            "prompt": 1,
            "variables": 1,
            "regexp": 1,
            "target": 1,
            "metric": 1,
            "rta_prompt": 1,
            "rta_model": 1,
            "status": 1,
        },
    ):
        key = (doc.get("line_index"), doc.get("model"))
        existing_entries[key] = doc

    operations = []

    # Для сравнения определим списки полей
    core_fields = ["prompt", "variables"]
    rta_fields = ["rta_prompt", "rta_model", "target", "regexp"]

    for key, expected_doc in expected_entries.items():
        if key in existing_entries:
            current_doc = existing_entries[key]
            update_fields = {}

            # Если изменились core-поля – выполняем полный апдейт (перезаписываем все, статус -> pending)
            if any(
                expected_doc.get(field) != current_doc.get(field)
                for field in core_fields
            ):
                update_fields.update(expected_doc)
            else:
                # Если изменились только rta-поля, обновляем только их
                if any(
                    expected_doc.get(field) != current_doc.get(field)
                    for field in rta_fields
                ):
                    for field in rta_fields:
                        if expected_doc.get(field) != current_doc.get(field):
                            update_fields[field] = expected_doc.get(field)
                    update_fields["status"] = "pending"
            if update_fields:
                operations.append(
                    UpdateOne({"_id": current_doc["_id"]}, {"$set": update_fields})
                )
        else:
            # Если записи нет – вставляем
            operations.append(InsertOne(expected_doc))

    # Удаляем записи, которые больше не должны присутствовать (например, если изменился список моделей или датасета)
    for key, doc in existing_entries.items():
        if key not in expected_keys:
            operations.append(DeleteOne({"_id": doc["_id"]}))

    if operations:
        try:
            result = queue_coll.bulk_write(operations, ordered=False)
            # Для InsertOne в результате может не быть прямого счётчика, поэтому выводим информацию, если возможно.
            logger.info(
                f"Синхронизирована очередь '{queue_coll_name}': "
                f"modified {result.modified_count}, deleted {result.deleted_count}."
            )
        except Exception as e:
            logger.error(f"Ошибка синхронизации очереди '{queue_coll_name}': {e}")
    else:
        logger.info(f"Очередь '{queue_coll_name}' уже синхронизирована.")

    # Если задание с метрикой RtA – синхронизируем соответствующую rta‑очередь
    if task.get("metric") == "RtA":
        synchronize_rta_queue(db, task, queue_coll)


def synchronize_rta_queue(db: Database, task: dict, main_queue_coll):
    """
    Для задач с метрикой RtA:
      – Проверяем, чтобы для каждого документа основной очереди существовала соответствующая запись в rta‑очереди.
      – Если изменились rta‑поля (или target/regexp) – обновляем их в rta‑очереди.
    """
    task_name = task["task_name"]
    rta_coll_name = f"queue_rta_{task_name}"
    rta_coll = db[rta_coll_name]

    main_docs = list(main_queue_coll.find({"metric": "RtA"}))
    operations = []

    for doc in main_docs:
        source_id = doc["_id"]
        # Ожидаемые rta‑поля берём из задания
        expected_rta = {
            "prompt": task.get("rta_prompt"),
            "model": task.get("rta_model"),
            "target": task.get("target")
            if isinstance(task.get("target"), str)
            else task["metric"],
            "regexp": task.get("regexp"),
        }
        rta_doc = rta_coll.find_one({"source_id": source_id})
        if rta_doc:
            update_fields = {}
            for field, exp_val in expected_rta.items():
                if rta_doc.get(field) != exp_val:
                    update_fields[field] = exp_val
            if update_fields:
                update_fields["status"] = "pending"
                operations.append(
                    UpdateOne({"_id": rta_doc["_id"]}, {"$set": update_fields})
                )
        else:
            # Если записи ещё нет – создаём новую rta запись
            new_rta_doc = {
                "task_name": task.get("task_name"),
                "dataset_name": task.get("dataset_name"),
                "init_prompt": doc.get("prompt"),
                "init_model": doc.get("model"),
                "regexp": task.get("regexp"),
                "prompt": task.get("rta_prompt"),
                "model": task.get("rta_model"),
                "variables": doc.get("variables"),
                "status": "pending",
                "metric": "accuracy",  # согласно условию
                "target": task.get("target")
                if isinstance(task.get("target"), str)
                else task["metric"],
                "source_id": source_id,
            }
            operations.append(InsertOne(new_rta_doc))

    if operations:
        try:
            result = rta_coll.bulk_write(operations, ordered=False)
            logger.info(
                f"Синхронизирована RTA очередь '{rta_coll_name}': modified {result.modified_count}."
            )
        except Exception as e:
            logger.error(f"Ошибка синхронизации RTA очереди '{rta_coll_name}': {e}")
    else:
        logger.info(f"RTA очередь '{rta_coll_name}' уже синхронизирована.")


def delete_unused_queues(db: Database):
    """
    Удаляет коллекции очередей, для которых отсутствует задание в tasks.
    """
    all_cols = db.list_collection_names()
    queue_cols = [c for c in all_cols if c.startswith("queue_")]

    tasks_coll = db["tasks"]
    tasks = list(tasks_coll.find({}, {"task_name": 1}))
    valid_tasks = {t.get("task_name") for t in tasks}

    for col in queue_cols:
        # Для rta‑очередей имя имеет вид "queue_rta_{task_name}"
        if col.startswith("queue_rta_"):
            task_name = col[len("queue_rta_") :]
        else:
            task_name = col[len("queue_") :]
        if task_name not in valid_tasks:
            try:
                db.drop_collection(col)
                logger.info(f"Удалена неиспользуемая коллекция '{col}'.")
            except Exception as e:
                logger.error(f"Ошибка удаления коллекции '{col}': {e}")


def synchronize_all_tasks(db: Database):
    tasks_coll = db["tasks"]
    tasks = list(tasks_coll.find({}))
    for task in tasks:
        try:
            logger.info(f"Синхронизация задания '{task.get('task_name')}'.")
            synchronize_task_queue(db, task)
        except Exception as e:
            logger.error(f"Ошибка синхронизации задания '{task.get('task_name')}': {e}")


def main():
    db = get_db()
    interval = 10  # интервал синхронизации в секундах
    while True:
        synchronize_all_tasks(db)
        delete_unused_queues(db)
        time.sleep(interval)


if __name__ == "__main__":
    main()
