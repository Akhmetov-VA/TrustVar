#!/usr/bin/env python3
import logging
import os
import time
from typing import Dict, Tuple, Any, List

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


def canonical_variables(variables: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    """
    Приводит словарь переменных к каноническому виду – кортеж отсортированных пар,
    чтобы использовать его в качестве части уникального ключа.
    """
    return tuple(sorted(variables.items()))


def compute_expected_entries(
    task: dict, df: pd.DataFrame
) -> Tuple[Dict[Tuple, dict], Dict[Tuple, dict]]:
    """
    Для данного задания и датасета вычисляет ожидаемые записи для очередей.
    Возвращаются два словаря:
      - expected_main: для коллекции queue_{task_name}
      - expected_rta: для коллекции queue_rta_{task_name} (только если metric == 'RtA')

    Для каждой строки датасета и для каждого значения из списка моделей (task["models"])
    формируется документ со следующими полями:
      - Обязательные ключевые поля: "prompt", "model", "variables" (где variables – словарь, полученный из columns,
        указанных в task["variables_cols"]). Уникальность определяется как (prompt, model, canonical_variables(variables)).
      - Остальные поля: dataset_name, metric, regexp, target и task_name.
      - Для metric == "include_exclude": дополнительно include_list и exclude_list, если заданы соответствующие колонки.
      - Для metric == "RtA": в основном документе также присутствуют rta_prompt и rta_model;
        отдельно формируется ожидаемая запись для RTA-очереди с ключом (rta_prompt, rta_model, variables).
    """
    expected_main = {}
    expected_rta = {}

    task_name = task["task_name"]
    dataset_name = task["dataset_name"]
    prompt = task["prompt"]
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
    for row in rows:
        # Извлекаем переменные из строки по заданным колонкам
        variables = {col: row.get(col) for col in var_cols}
        canon_vars = canonical_variables(variables)
        for model in models:
            # Формируем документ для основной очереди
            doc_main = {
                "prompt": prompt,
                "variables": variables,
                "model": model,
                "dataset_name": dataset_name,
                "metric": metric,
                "regexp": regexp,
                "task_name": task_name,
                "status": "pending",  # по умолчанию новая задача pending
            }
            if metric == "RtA":
                # Для RtA в основной очереди сохраняем rta-поля для справки
                doc_main["rta_prompt"] = rta_prompt
                doc_main["rta_model"] = rta_model
                doc_main["target"] = target if isinstance(target, str) else metric
            elif metric == "include_exclude":
                if include_col and include_col in row:
                    value = row.get(include_col)
                    doc_main["include_list"] = (
                        [value] if isinstance(value, str) else value
                    )
                if exclude_col and exclude_col in row:
                    value = row.get(exclude_col)
                    doc_main["exclude_list"] = (
                        [value] if isinstance(value, str) else value
                    )
                doc_main["target"] = target if isinstance(target, str) else metric
            else:
                # Для остальных метрик target берётся из строки, если присутствует нужный ключ
                doc_main["target"] = row.get(target) if target in row else None

            key_main = (prompt, model, canon_vars)
            expected_main[key_main] = doc_main

            # Если метрика RtA – формируем ожидаемую запись для RTA-очереди
            if metric == "RtA":
                doc_rta = {
                    "prompt": rta_prompt,
                    "variables": variables,
                    "model": rta_model,
                    "dataset_name": dataset_name,
                    "metric": metric,
                    "task_name": task_name,
                    "status": "pending",
                    # target для RTA можно задать аналогично
                    "target": target if isinstance(target, str) else metric,
                }
                key_rta = (rta_prompt, rta_model, canon_vars)
                expected_rta[key_rta] = doc_rta

    return expected_main, expected_rta


def synchronize_queue(
    db: Database,
    queue_coll_name: str,
    expected_entries: Dict[Tuple, dict],
    key_fields: List[str],
    update_status: str,
):
    """
    Синхронизирует коллекцию очереди (queue_ или queue_rta_) с ожидаемыми записями.
    key_fields – список имен полей, которые входят в уникальный ключ (например, ["prompt", "model", "variables"]).
    update_status – статус, который присваивается при обновлении существующей записи (completed),
                    если изменения произошли только в неключевых полях.

    Логика:
      - Если в очереди отсутствует запись с ожидаемым ключом – вставляем её (status оставляем как в expected).
      - Если запись есть, сравниваем оставшиеся поля:
            • Если значения отличаются, обновляем запись, устанавливая статус = update_status.
      - Если в очереди есть запись, для которой нет ожидаемого ключа – удаляем её.
    """
    coll = db[queue_coll_name]
    projection = {field: 1 for field in key_fields}
    # Добавляем также служебные поля, которые могут быть обновлены
    extra_fields = [
        "dataset_name",
        "metric",
        "regexp",
        "target",
        "include_list",
        "exclude_list",
        "rta_prompt",
        "rta_model",
        "task_name",
        "status",
    ]
    for f in extra_fields:
        projection[f] = 1

    existing_entries = {}
    for doc in coll.find({}, projection):
        # Для формирования ключа используем именно поля key_fields
        key = tuple(doc.get(f) for f in key_fields)
        # Если поле variables – преобразуем его в каноническую форму
        if "variables" in key_fields and isinstance(doc.get("variables"), dict):
            # Перестраиваем ключ так, чтобы variables было каноническим кортежем
            idx = key_fields.index("variables")
            key = list(key)
            key[idx] = canonical_variables(doc.get("variables"))
            key = tuple(key)
        existing_entries[key] = doc

    operations = []

    # Обрабатываем ожидаемые записи: вставка или обновление
    for key, expected_doc in expected_entries.items():
        if key in existing_entries:
            current_doc = existing_entries[key]
            update_fields = {}
            # Сравниваем все поля, кроме ключевых
            for field, value in expected_doc.items():
                if field in key_fields:
                    continue
                if current_doc.get(field) != value:
                    update_fields[field] = value
            if update_fields:
                # При обновлении, если разница обнаружена только в неключевых полях – статус переводим в update_status (completed)
                update_fields["status"] = update_status
                operations.append(
                    UpdateOne({"_id": current_doc["_id"]}, {"$set": update_fields})
                )
        else:
            # Новая запись – вставляем как есть (status уже pending)
            operations.append(InsertOne(expected_doc))

    # Удаляем документы, которые есть в очереди, но отсутствуют в ожидаемом наборе
    expected_keys = set(expected_entries.keys())
    for key, current_doc in existing_entries.items():
        if key not in expected_keys:
            operations.append(DeleteOne({"_id": current_doc["_id"]}))

    if operations:
        try:
            result = coll.bulk_write(operations, ordered=False)
            logger.info(
                f"Синхронизация коллекции '{queue_coll_name}': "
                f"modified {result.modified_count}, deleted {result.deleted_count}, inserted {getattr(result, 'inserted_count', 0)}."
            )
        except Exception as e:
            logger.error(f"Ошибка синхронизации коллекции '{queue_coll_name}': {e}")
    else:
        logger.info(f"Коллекция '{queue_coll_name}' уже синхронизирована.")


def synchronize_task_queue(db: Database, task: dict):
    """
    Синхронизирует задание из tasks с очередями.
      - Обновляются документы в основной очереди (queue_{task_name}).
          • Если меняется список моделей, добавляются или удаляются записи.
          • Если меняются поля prompt или variables_cols (то есть изменяется ключ), старые записи удаляются,
            а новые вставляются с status = pending.
          • Если меняются только остальные поля, то соответствующие записи обновляются с переводом в status = completed.
      - Если задание имеет metric == "RtA", аналогичным образом синхронизируется очередь queue_rta_{task_name},
        но с логикой: при изменении rta_prompt или rta_model – status = pending, иначе – status = completed.
    """
    task_name = task["task_name"]
    dataset_name = task["dataset_name"]
    queue_coll_name = f"queue_{task_name}"
    df = get_dataset_head(db, dataset_name)
    if df.empty:
        logger.warning(
            f"Датасет '{dataset_name}' пуст – пропускаем задание '{task_name}'."
        )
        return

    expected_main, expected_rta = compute_expected_entries(task, df)
    # Для основной очереди уникальный ключ – (prompt, model, variables)
    synchronize_queue(
        db,
        queue_coll_name,
        expected_main,
        key_fields=["prompt", "model", "variables"],
        update_status="completed",
    )

    # Если метрика RtA – синхронизируем и очередь для RTA
    if task["metric"] == "RtA":
        rta_coll_name = f"queue_rta_{task_name}"
        # Уникальный ключ для RTA – (rta_prompt, rta_model, variables)
        synchronize_queue(
            db,
            rta_coll_name,
            expected_rta,
            key_fields=["prompt", "model", "variables"],
            update_status="completed",
        )


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
        # Для rta-очередей имя имеет вид "queue_rta_{task_name}"
        if col.startswith("queue_rta_"):
            tname = col[len("queue_rta_") :]
        else:
            tname = col[len("queue_") :]
        if tname not in valid_tasks:
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
