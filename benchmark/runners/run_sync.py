import logging
import os
import time
from typing import Any, Dict

import pandas as pd
from pymongo import MongoClient
from pymongo.database import Database

from utils.constants import MONGO_HOST, MONGO_PASSWORD, MONGO_PORT, MONGO_USERNAME

MONGO_DB = os.environ.get("MONGO_DB", "TrustGen")

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
    logger.info("Подключение к MongoDB успешно установлено.")
    return client


def get_db() -> Database:
    client = get_mongo_client()
    db = client[MONGO_DB]
    logger.info(f"Используется база данных: {MONGO_DB}")
    return db


def collection_exists(db: Database, coll_name: str) -> bool:
    """Вспомогательная функция для проверки существования коллекции."""
    exists = coll_name in db.list_collection_names()
    logger.debug(f"Проверка существования коллекции {coll_name}: {exists}")
    return exists


def get_dataset_head(db: Database, dataset_name: str) -> pd.DataFrame:
    """
    Возвращает датасет из коллекции dataset_<dataset_name> в виде DataFrame.
    При этом идентификатор _id не удаляется – он используется для связывания с очередями.
    """
    coll_name = f"dataset_{dataset_name}"
    logger.info(f"Загружаем датасет из коллекции {coll_name}.")
    coll = db[coll_name]
    docs = list(coll.find({}))
    if not docs:
        logger.warning(f"Датасет {coll_name} пуст.")
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    logger.info(f"Датасет {coll_name} загружен: {len(df)} записей.")
    return df


def sync_task_name(db: Database, task: Dict[str, Any]) -> None:
    """
    Синхронизирует поле task_name во всех очередях, связанных с задачей.
    Обновление производится без изменения статуса документов.
    """
    new_task_name = task.get("task_name")
    logger.info(f"Синхронизация поля task_name: новое значение {new_task_name}.")
    # Обновляем в основной очереди
    main_queue = f"queue_{new_task_name}"
    if collection_exists(db, main_queue):
        db[main_queue].update_many({}, {"$set": {"task_name": new_task_name}})
        logger.info(f"Поле task_name обновлено в коллекции {main_queue}.")
    # Обновляем в rta-очереди, если она существует
    rta_queue = f"queue_rta_{new_task_name}"
    if collection_exists(db, rta_queue):
        db[rta_queue].update_many({}, {"$set": {"task_name": new_task_name}})
        logger.info(f"Поле task_name обновлено в коллекции {rta_queue}.")


def sync_models(db: Database, task: Dict[str, Any]) -> None:
    """
    Синхронизирует модели для задачи:
      - Удаляет из коллекций queue_{task_name} и (при RtA) queue_rta_{task_name} записи, для которых поле model отсутствует
        в обновленном списке моделей.
      - Для каждой строки датасета и для каждой модели из обновленного списка, если запись с комбинацией
        (model, variables, prompt) отсутствует, создается новая запись со статусом pending.
    """
    logger.info(f"Начало синхронизации моделей для задачи: {task.get('task_name')}")
    task_name = task.get("task_name")
    dataset_name = task.get("dataset_name")
    metric = task.get("metric", "")
    new_models = set(task.get("models", []))
    queue_coll_name = f"queue_{task_name}"
    queue_coll = db[queue_coll_name]

    # Удаляем записи, где model не входит в актуальный список
    delete_result = queue_coll.delete_many({"model": {"$nin": list(new_models)}})
    if delete_result.deleted_count:
        logger.info(
            f"Удалено {delete_result.deleted_count} записей из {queue_coll_name} по удалённым моделям."
        )

    # Если метрика RtA — удаляем записи из соответствующей rta-коллекции
    if metric == "RtA":
        rta_coll_name = f"queue_rta_{task_name}"
        if collection_exists(db, rta_coll_name):
            rta_coll = db[rta_coll_name]
            delete_rta = rta_coll.delete_many(
                {"init_model": {"$nin": list(new_models)}}
            )
            if delete_rta.deleted_count:
                logger.info(
                    f"Удалено {delete_rta.deleted_count} записей из {rta_coll_name} по удалённым моделям."
                )

    # Собираем существующие ключи: (model, variables, prompt)
    existing_keys = set()
    for doc in queue_coll.find({}, {"model": 1, "variables": 1, "prompt": 1}):
        key = (
            doc.get("model"),
            tuple(sorted(doc.get("variables", {}).items())),
            doc.get("prompt"),
        )
        existing_keys.add(key)
    logger.debug(f"Найдено существующих записей: {len(existing_keys)}")

    df = get_dataset_head(db, dataset_name)
    if df.empty:
        logger.warning(
            f"Датасет '{dataset_name}' пуст. Пропускаем создание новых записей для моделей."
        )
        return

    new_inserts = []
    var_cols = task.get("variables_cols", [])
    base_prompt = task.get("prompt", "")
    regexp = task.get("regexp")
    target = task.get("target")
    rows = df.to_dict("records")
    for row in rows:
        variables = {col: row.get(col) for col in var_cols} if var_cols else {}
        for model in new_models:
            key = (model, tuple(sorted(variables.items())), base_prompt)
            if key in existing_keys:
                continue
            doc = {
                "dataset_id": row.get("_id"),
                "model": model,
                "variables": variables,
                "prompt": base_prompt,
                "metric": metric,
                "regexp": regexp,
                "status": "pending",
                "response": None,
                "task_name": task_name,
            }
            if metric == "RtA":
                rta_prompt = task.get("rta_prompt")
                rta_model = task.get("rta_model")
                if rta_prompt and rta_model:
                    doc["rta_prompt"] = rta_prompt
                    doc["rta_model"] = rta_model
                doc["target"] = target if isinstance(target, str) else metric
            elif metric == "include_exclude":
                include_col = task.get("include_column")
                exclude_col = task.get("exclude_column")
                if include_col and include_col in row:
                    value = row.get(include_col)
                    doc["include_list"] = [value] if isinstance(value, str) else value
                if exclude_col and exclude_col in row:
                    value = row.get(exclude_col)
                    doc["exclude_list"] = [value] if isinstance(value, str) else value
                doc["target"] = target if isinstance(target, str) else metric
            else:
                doc["target"] = row[target] if (target and target in row) else None
            new_inserts.append(doc)
    if new_inserts:
        try:
            result = queue_coll.insert_many(new_inserts, ordered=False)
            logger.info(
                f"Вставлено {len(result.inserted_ids)} новых записей в {queue_coll_name} для моделей."
            )
        except Exception as e:
            logger.error(f"Ошибка при вставке новых записей в {queue_coll_name}: {e}")
    logger.info(f"Завершена синхронизация моделей для задачи: {task_name}")


def sync_prompt(db: Database, task: Dict[str, Any]) -> None:
    """
    Обновляет поле prompt во всех документах основной очереди, переводя их в статус pending,
    только если новое значение отличается от текущего.
    Если метрика задачи RtA, то полностью удаляется коллекция queue_rta_{task_name}.
    """
    logger.info(f"Начало синхронизации prompt для задачи: {task.get('task_name')}")
    task_name = task.get("task_name")
    new_prompt = task.get("prompt", "")
    queue_coll_name = f"queue_{task_name}"
    queue_coll = db[queue_coll_name]

    update_result = queue_coll.update_many(
        {"prompt": {"$ne": new_prompt}},
        {"$set": {"prompt": new_prompt, "status": "pending", "task_name": task_name}},
    )
    if update_result.modified_count:
        logger.info(
            f"Обновлено {update_result.modified_count} записей в {queue_coll_name} с новым prompt."
        )
        if task.get("metric") == "RtA":
            rta_coll_name = f"queue_rta_{task_name}"
            if collection_exists(db, rta_coll_name):
                db.drop_collection(rta_coll_name)
                logger.info(
                    f"Коллекция {rta_coll_name} удалена из-за изменения prompt для задачи RtA."
                )
    logger.info(f"Завершена синхронизация prompt для задачи: {task_name}")


def sync_variables(db: Database, task: Dict[str, Any]) -> None:
    """
    Если в задаче заданы variables_cols, функция сравнивает список переменных из task с ключами поля
    variables в документах основной очереди (queue_{task_name}). Если они отличаются, удаляется коллекция.
    Если задача имеет метрику RtA, дополнительно удаляется коллекция queue_rta_{task_name}.
    """
    logger.info(f"Начало синхронизации variables для задачи: {task.get('task_name')}")
    task_name = task.get("task_name")
    var_cols = task.get("variables_cols", [])
    if not var_cols:
        logger.info("Нет variables_cols в задаче, пропускаем синхронизацию variables.")
        return

    queue_coll_name = f"queue_{task_name}"
    if collection_exists(db, queue_coll_name):
        doc = db[queue_coll_name].find_one({})
        if doc:
            current_keys = set(doc.get("variables", {}).keys())
            new_keys = set(var_cols)
            if current_keys != new_keys:
                db.drop_collection(queue_coll_name)
                logger.info(
                    f"Коллекция {queue_coll_name} удалена из-за изменения variables_cols: {current_keys} -> {new_keys}."
                )
                if task.get("metric") == "RtA":
                    rta_coll_name = f"queue_rta_{task_name}"
                    if collection_exists(db, rta_coll_name):
                        db.drop_collection(rta_coll_name)
                        logger.info(
                            f"Коллекция {rta_coll_name} удалена из-за изменения variables_cols для задачи."
                        )
        else:
            logger.info(
                f"Коллекция {queue_coll_name} пуста. Пропускаем проверку variables_cols."
            )
    else:
        logger.info(f"Коллекция {queue_coll_name} не существует, нечего удалять.")
    logger.info(f"Завершена синхронизация variables для задачи: {task_name}")


def sync_regexp_include_exclude(db: Database, task: Dict[str, Any]) -> None:
    """
    Обновляет поля regexp, target, а также include_list и exclude_list в основной очереди,
    только если новые значения отличаются от текущих.
    Если документ имеет статус extracted и были произведены изменения, его статус переводится в completed.
    Обновление в rta-очереди не производится, так как target для RtA всегда 1 или 0.
    """
    logger.info(
        f"Начало синхронизации regexp/include-exclude для задачи: {task.get('task_name')}"
    )
    task_name = task.get("task_name")
    new_regexp = task.get("regexp")
    include_col = task.get("include_column")
    exclude_col = task.get("exclude_column")
    dataset_name = task.get("dataset_name")
    queue_coll_name = f"queue_{task_name}"
    queue_coll = db[queue_coll_name]

    total_modified = 0

    update_result = queue_coll.update_many(
        {"regexp": {"$ne": new_regexp}},
        {"$set": {"regexp": new_regexp, "task_name": task_name}},
    )

    if update_result.modified_count:
        total_modified += update_result.modified_count
        logger.info(
            f"Обновлено {update_result.modified_count} записей в {queue_coll_name} с новым regexp и target."
        )

    if include_col or exclude_col:
        df = get_dataset_head(db, dataset_name)
        if not df.empty:
            for row in df.to_dict("records"):
                update_fields = {}
                if include_col and include_col in row:
                    new_include = (
                        [row.get(include_col)]
                        if isinstance(row.get(include_col), str)
                        else row.get(include_col)
                    )
                    update_fields["include_list"] = new_include
                if exclude_col and exclude_col in row:
                    new_exclude = (
                        [row.get(exclude_col)]
                        if isinstance(row.get(exclude_col), str)
                        else row.get(exclude_col)
                    )
                    update_fields["exclude_list"] = new_exclude
                if update_fields:
                    filter_query = {
                        "dataset_id": row.get("_id"),
                        "prompt": task.get("prompt", ""),
                        "$or": [
                            {
                                "include_list": {
                                    "$ne": update_fields.get("include_list")
                                }
                            },
                            {
                                "exclude_list": {
                                    "$ne": update_fields.get("exclude_list")
                                }
                            },
                        ],
                    }
                    update_res = queue_coll.update_many(
                        filter_query, {"$set": update_fields}
                    )
                    if update_res.modified_count:
                        total_modified += update_res.modified_count
                        logger.info(
                            f"Обновлены include/exclude поля для dataset_id {row.get('_id')} в {queue_coll_name}."
                        )
    if total_modified:
        status_update = queue_coll.update_many(
            {"status": "extracted"}, {"$set": {"status": "completed"}}
        )
        if status_update.modified_count:
            logger.info(
                f"Изменено статус {status_update.modified_count} записей в {queue_coll_name} с extracted на completed."
            )
    logger.info(
        f"Завершена синхронизация regexp/target/include-exclude для задачи: {task_name}"
    )


def sync_rta_fields(db: Database, task: Dict[str, Any]) -> None:
    """
    Обновляет поля rta_prompt и rta_model:
      - В основной очереди (queue_{task_name}) обновляются записи, если новые значения отличаются, с установкой статуса "completed".
      - Если коллекция rta-очереди (queue_rta_{task_name}) существует, она удаляется.
    """
    logger.info(f"Начало синхронизации rta-полей для задачи: {task.get('task_name')}")
    task_name = task.get("task_name")
    new_rta_prompt = task.get("rta_prompt")
    new_rta_model = task.get("rta_model")
    queue_coll_name = f"queue_{task_name}"
    queue_coll = db[queue_coll_name]

    update_result = queue_coll.update_many(
        {
            "$or": [
                {"rta_prompt": {"$ne": new_rta_prompt}},
                {"rta_model": {"$ne": new_rta_model}},
            ]
        },
        {
            "$set": {
                "rta_prompt": new_rta_prompt,
                "rta_model": new_rta_model,
                "status": "completed",
                "task_name": task_name,
            }
        },
    )
    if update_result.modified_count:
        logger.info(
            f"Обновлено {update_result.modified_count} записей в {queue_coll_name} с новыми rta_prompt и rta_model, статус изменен на completed."
        )

    rta_coll_name = f"queue_rta_{task_name}"
    if collection_exists(db, rta_coll_name):
        db.drop_collection(rta_coll_name)
        logger.info(
            f"Коллекция {rta_coll_name} удалена, так как rta поля были изменены."
        )
    logger.info(f"Завершена синхронизация rta-полей для задачи: {task_name}")


def sync_task(db: Database, task: Dict[str, Any]) -> None:
    """
    Синхронизирует очередь для одной задачи, последовательно обновляя task_name, модели, prompt, variables,
    regexp/target/include-exclude и rta-поля.
    """
    logger.info(f"==== Начало синхронизации задачи: {task.get('task_name')} ====")
    sync_task_name(db, task)
    sync_models(db, task)
    sync_prompt(db, task)
    sync_variables(db, task)
    sync_regexp_include_exclude(db, task)
    sync_rta_fields(db, task)
    logger.info(f"==== Завершена синхронизация задачи: {task.get('task_name')} ====")


def sync_all_tasks(db: Database) -> None:
    """
    Обходит все задачи из коллекции tasks и синхронизирует очереди для каждой.
    """
    logger.info("Начало синхронизации всех задач.")
    tasks_coll = db["tasks"]
    tasks = list(tasks_coll.find({}))
    if not tasks:
        logger.info("Нет задач для синхронизации.")
        return
    logger.info(f"Найдено {len(tasks)} задач для синхронизации.")
    for task in tasks:
        sync_task(db, task)
    logger.info("Синхронизация всех задач завершена.")


def main():
    db = get_db()
    interval = 10  # интервал проверки в секундах
    logger.info("Запуск цикла синхронизации задач.")
    while True:
        sync_all_tasks(db)
        time.sleep(interval)


if __name__ == "__main__":
    main()
