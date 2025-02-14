#!/usr/bin/env python3
import logging
import os
import time
from typing import Dict, Tuple

import pandas as pd
from pymongo import DeleteOne, InsertOne, UpdateOne, MongoClient
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
    docs = list(db[coll_name].find({}))
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    df.drop(columns=["_id"], errors="ignore", inplace=True)
    return df


def compute_expected_main_queue_entries(
    task: dict, df: pd.DataFrame
) -> Dict[Tuple[int, str], dict]:
    """
    Вычисляет ожидаемые записи для основной очереди queue_<task_name>.
    Для каждого ряда датасета и для каждой модели из task["models"] формируется документ.
    В зависимости от metric добавляются специфичные поля.
    """
    expected = {}
    task_name = task["task_name"]
    dataset_name = task["dataset_name"]
    prompt_text = task["prompt"]
    var_cols = task.get("variables_cols", [])
    models = task["models"]
    metric = task["metric"]
    regexp = task.get("regexp")
    target = task.get("target")
    include_col = task.get("include_column")
    exclude_col = task.get("exclude_column")

    rows = df.to_dict("records")
    for i, row in enumerate(rows):
        # Формируем словарь с переменными по выбранным колонкам
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
                "task_name": task_name,
            }
            if metric == "RtA":
                # Даже если задание с RtA попадает в основную очередь, здесь остаются поля rta для синхронизации
                doc["rta_prompt"] = task.get("rta_prompt")
                doc["rta_model"] = task.get("rta_model")
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
                doc["target"] = row.get(target) if target in row else None
            # Изначально статус выставляем как pending (он может измениться при сравнении)
            doc["status"] = "pending"
            expected[(i, model)] = doc
    return expected


def compute_expected_rta_queue_entries(task: dict, df: pd.DataFrame) -> Dict[int, dict]:
    """
    Вычисляет ожидаемые записи для RTA-очереди queue_rta_<task_name>.
    Для заданий с metric "RtA" для каждого ряда датасета формируется документ,
    в котором вместо стандартных prompt и model используются rta_prompt и rta_model.
    """
    expected = {}
    task_name = task["task_name"]
    dataset_name = task["dataset_name"]
    rta_prompt = task.get("rta_prompt")
    rta_model = task.get("rta_model")
    metric = task["metric"]
    regexp = task.get("regexp")
    target = task.get("target")
    rows = df.to_dict("records")
    for i, _ in enumerate(rows):
        doc = {
            "line_index": i,
            "dataset_name": dataset_name,
            "prompt": rta_prompt,
            "model": rta_model,
            "metric": metric,
            "regexp": regexp,
            "response": None,
            "task_name": task_name,
            "target": target if isinstance(target, str) else metric,
            "status": "pending",
        }
        expected[i] = doc
    return expected


def diff_update(
    expected_doc: dict, current_doc: dict, critical_keys: list
) -> Tuple[dict, bool]:
    """
    Сравнивает ожидаемый и текущий документ (за исключением служебных полей) и возвращает:
      - словарь обновлений,
      - флаг, изменилось ли хоть одно из критических полей.
    Критическими считаются ключи из critical_keys.
    """
    updates = {}
    critical_changed = False
    for key, expected_value in expected_doc.items():
        if key in ["_id", "status", "response"]:
            continue
        current_value = current_doc.get(key)
        if expected_value != current_value:
            updates[key] = expected_value
            if key in critical_keys:
                critical_changed = True
    return updates, critical_changed


def synchronize_queue(db: Database, task: dict):
    """
    Синхронизирует задание из tasks с очередями:
      - Для основной очереди (queue_<task_name>): сравниваются ожидаемые записи и существующие.
        Если изменились поля prompt или variables (critical для основной очереди), статус переводится в pending,
        а если изменились только остальные поля – в completed.
      - Если количество моделей изменилось, то соответствующие документы добавляются или удаляются.
      - Для RTA-очереди (queue_rta_<task_name>) для заданий с metric "RtA" аналогичным образом сравниваются rta_prompt и rta_model
        (критические поля для RTA).
    """
    task_name = task["task_name"]
    dataset_name = task["dataset_name"]

    df = get_dataset_head(db, dataset_name)
    if df.empty:
        logger.warning(
            f"Датасет '{dataset_name}' пуст – пропускаем задание '{task_name}'."
        )
        return

    # --- Основная очередь ---
    expected_main = compute_expected_main_queue_entries(task, df)
    main_coll_name = f"queue_{task_name}"
    main_coll = db[main_coll_name]
    projection = {
        "line_index": 1,
        "model": 1,
        "prompt": 1,
        "variables": 1,
        "regexp": 1,
        "target": 1,
        "metric": 1,
        "include_list": 1,
        "exclude_list": 1,
        "task_name": 1,
        "rta_prompt": 1,
        "rta_model": 1,
        "status": 1,
    }
    existing_main = {}
    for doc in main_coll.find({}, projection):
        key = (doc.get("line_index"), doc.get("model"))
        existing_main[key] = doc

    main_operations = []
    # Критические для основной очереди поля – prompt и variables
    critical_main = ["prompt", "variables"]
    for key, exp_doc in expected_main.items():
        if key in existing_main:
            curr_doc = existing_main[key]
            updates, critical_changed = diff_update(exp_doc, curr_doc, critical_main)
            if updates:
                new_status = "pending" if critical_changed else "completed"
                updates["status"] = new_status
                main_operations.append(
                    UpdateOne({"_id": curr_doc["_id"]}, {"$set": updates})
                )
        else:
            main_operations.append(InsertOne(exp_doc))
    # Удаляем документы, которые больше не соответствуют заданию
    for key, curr_doc in existing_main.items():
        if key not in expected_main:
            main_operations.append(DeleteOne({"_id": curr_doc["_id"]}))
    if main_operations:
        try:
            result = main_coll.bulk_write(main_operations, ordered=False)
            logger.info(
                f"Синхронизирована очередь '{main_coll_name}': modified {result.modified_count}, "
                f"inserted {getattr(result, 'inserted_count', 0)}, deleted {result.deleted_count}."
            )
        except Exception as e:
            logger.error(f"Ошибка синхронизации очереди '{main_coll_name}': {e}")
    else:
        logger.info(f"Очередь '{main_coll_name}' уже синхронизирована.")

    # --- Очередь RTA (для metric == "RtA") ---
    if task.get("metric") == "RtA":
        expected_rta = compute_expected_rta_queue_entries(task, df)
        rta_coll_name = f"queue_rta_{task_name}"
        rta_coll = db[rta_coll_name]
        projection_rta = {
            "line_index": 1,
            "prompt": 1,
            "model": 1,
            "regexp": 1,
            "target": 1,
            "metric": 1,
            "task_name": 1,
            "status": 1,
        }
        existing_rta = {}
        for doc in rta_coll.find({}, projection_rta):
            key = doc.get("line_index")
            existing_rta[key] = doc

        rta_operations = []
        # Критические для RTA очереди поля – prompt и model (которые представляют rta_prompt и rta_model)
        critical_rta = ["prompt", "model"]
        for key, exp_doc in expected_rta.items():
            if key in existing_rta:
                curr_doc = existing_rta[key]
                updates, critical_changed = diff_update(exp_doc, curr_doc, critical_rta)
                if updates:
                    new_status = "pending" if critical_changed else "completed"
                    updates["status"] = new_status
                    rta_operations.append(
                        UpdateOne({"_id": curr_doc["_id"]}, {"$set": updates})
                    )
            else:
                rta_operations.append(InsertOne(exp_doc))
        for key, curr_doc in existing_rta.items():
            if key not in expected_rta:
                rta_operations.append(DeleteOne({"_id": curr_doc["_id"]}))
        if rta_operations:
            try:
                result = rta_coll.bulk_write(rta_operations, ordered=False)
                logger.info(
                    f"Синхронизирована RTA очередь '{rta_coll_name}': modified {result.modified_count}, "
                    f"inserted {getattr(result, 'inserted_count', 0)}, deleted {result.deleted_count}."
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
    """
    Синхронизирует все задания из коллекции tasks с очередями.
    """
    tasks_coll = db["tasks"]
    tasks = list(tasks_coll.find({}))
    for task in tasks:
        try:
            logger.info(f"Синхронизация задания '{task.get('task_name')}'.")
            synchronize_queue(db, task)
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
