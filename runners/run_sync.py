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
    Creates a connection to MongoDB based on environment variables.
    """
    mongo_uri = (
        f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
    )
    client = MongoClient(mongo_uri)
    logger.info("The connection to MongoDB has been successfully established.")
    return client


def get_db() -> Database:
    client = get_mongo_client()
    db = client[MONGO_DB]
    logger.info(f"The database is being used: {MONGO_DB}")
    return db


def collection_exists(db: Database, coll_name: str) -> bool:
    """An auxiliary function for verifying the existence of a collection."""
    exists = coll_name in db.list_collection_names()
    logger.debug(f"Auxiliary function for checking the existence of a collection {coll_name}: {exists}")
    return exists


def get_dataset_head(db: Database, dataset_name: str) -> pd.DataFrame:
    """
    Returns a dataset from the dataset_<dataset_name> collection as a DataFrame.
    In this case, the _id identifier is not deleted – it is used to associate with queues.
    """
    coll_name = f"dataset_{dataset_name}"
    logger.info(f"Uploading a dataset from a collection {coll_name}.")
    coll = db[coll_name]
    docs = list(coll.find({}))
    if not docs:
        logger.warning(f"Dataset {coll_name} empty.")
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    logger.info(f"Dataset {coll_name} uploaded: {len(df)} records.")
    return df


def sync_task_name(db: Database, task: Dict[str, Any]) -> None:
    """
    Synchronizes the task_name field in all queues related to the task.
    The update is performed without changing the status of the documents.
    """
    new_task_name = task.get("task_name")
    logger.info(f"Synchronization of the task_name field: new value{new_task_name}.")
    # Updating in the main queue
    main_queue = f"queue_{new_task_name}"
    if collection_exists(db, main_queue):
        db[main_queue].update_many({}, {"$set": {"task_name": new_task_name}})
        logger.info(f"The task_name field has been updated in the collection {main_queue}.")
    # We update it in the rta queue, if it exists.
    rta_queue = f"queue_rta_{new_task_name}"
    if collection_exists(db, rta_queue):
        db[rta_queue].update_many({}, {"$set": {"task_name": new_task_name}})
        logger.info(f"The task_name field has been updated in the collection {rta_queue}.")


def sync_models(db: Database, task: Dict[str, Any]) -> None:
    """
    Synchronizes models for a task:
      - - Deletes from the collection queue_{task_name} and (if necessary) queue_rt_{task_name} entries for which the model field is missing
        in the updated list of models.
      - For each row of the dataset and for each model from the updated list, if an entry with a combination
        (model, variables, prompt) is missing, a new record with the pending status is being created.
    """
    logger.info(f"Starting synchronization of models for a task: {task.get('task_name')}")
    task_name = task.get("task_name")
    dataset_name = task.get("dataset_name")
    metric = task.get("metric", "")
    new_models = set(task.get("models", []))
    queue_coll_name = f"queue_{task_name}"
    queue_coll = db[queue_coll_name]

    # Deleting entries where the model is not included in the current list
    delete_result = queue_coll.delete_many({"model": {"$nin": list(new_models)}})
    if delete_result.deleted_count:
        logger.info(
            f"Deleted {delete_result.deleted_count} records from {queue_coll_name} by remote models."
        )

    # If the metric is RtA, we delete the records from the corresponding rta collection.
    if metric == "RtA":
        rta_coll_name = f"queue_rta_{task_name}"
        if collection_exists(db, rta_coll_name):
            rta_coll = db[rta_coll_name]
            delete_rta = rta_coll.delete_many(
                {"init_model": {"$nin": list(new_models)}}
            )
            if delete_rta.deleted_count:
                logger.info(
                    f"Deleted {delete_rta.deleted_count} records from {rta_coll_name} by remote models."
                )

    # Collecting existing keys: (model, variables, prompt)
    existing_keys = set()
    for doc in queue_coll.find({}, {"model": 1, "variables": 1, "prompt": 1}):
        key = (
            doc.get("model"),
            tuple(sorted(doc.get("variables", {}).items())),
            doc.get("prompt"),
        )
        existing_keys.add(key)
    logger.debug(f"Existing records found: {len(existing_keys)}")

    df = get_dataset_head(db, dataset_name)
    if df.empty:
        logger.warning(
            f"Dataset '{dataset_name}' empty. Skipping the creation of new records for models."
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
                f"Inserted {len(result.inserted_ids)} new entries in {queue_coll_name} for models."
            )
        except Exception as e:
            logger.error(f"Error when inserting new records in {queue_coll_name}: {e}")
    logger.info(f"Synchronization of models for the task is completed: {task_name}")


def sync_prompt(db: Database, task: Dict[str, Any]) -> None:
    """
    Updates the prompt field in all documents in the main queue, setting them to the pending status.,
    only if the new value differs from the current one.
    If the task metric is RtA, then the collection is completely deleted. queue_rta_{task_name}.
    """
    logger.info(f"Start syncing prompt for a task: {task.get('task_name')}")
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
            f"Update {update_result.modified_count} entries in {queue_coll_name} with a new prompt."
        )
        if task.get("metric") == "RtA":
            rta_coll_name = f"queue_rta_{task_name}"
            if collection_exists(db, rta_coll_name):
                db.drop_collection(rta_coll_name)
                logger.info(
                    f"Collection {rta_coll_name} deleted due to the prompt change for the RtA task."
                )
    logger.info(f"Prompt synchronization for the task is completed: {task_name}")


def sync_variables(db: Database, task: Dict[str, Any]) -> None:
    """
    If variables_cols are set in the task, the function compares the list of variables from the task with the keys of the field.
    variables in the documents of the main queue (queue_{task_name}). If they differ, the collection is deleted.
    If the task has the RtA metric, the queue_rta_{task_name} collection is additionally deleted.
    """
    logger.info(f"Starting synchronization of variables for a task: {task.get('task_name')}")
    task_name = task.get("task_name")
    var_cols = task.get("variables_cols", [])
    if not var_cols:
        logger.info("There are no variables_cols in the task, we skip synchronization variables.")
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
                    f"Collection {queue_coll_name} deleted due to a change variables_cols: {current_keys} -> {new_keys}."
                )
                if task.get("metric") == "RtA":
                    rta_coll_name = f"queue_rta_{task_name}"
                    if collection_exists(db, rta_coll_name):
                        db.drop_collection(rta_coll_name)
                        logger.info(
                            f"Collection {rta_coll_name} deleted due to a change variables_cols for the task."
                        )
        else:
            logger.info(
                f"Collection {queue_coll_name} empty. Skipping the check variables_cols."
            )
    else:
        logger.info(f"Collection {queue_coll_name} does not exist, there is nothing to delete.")
    logger.info(f"Synchronization of variables for the task is completed: {task_name}")


def sync_regexp_include_exclude(db: Database, task: Dict[str, Any]) -> None:
    """
    Updates the regexp, target, and include_list and exclude_list fields in the main queue.,
    only if the new values differ from the current ones.
    If a document has the extracted status and changes have been made, its status is converted to completed.
    There is no update in the rta queue, since the target for the RtA is always 1 or 0.
    """
    logger.info(
        f"Start of regexp/include-exclude synchronization for a task: {task.get('task_name')}"
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
            f"Update {update_result.modified_count} entries in {queue_coll_name} with the new regexp and target."
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
                            f"Updated include/exclude fields for dataset_id {row.get('_id')} in {queue_coll_name}."
                        )
    if total_modified:
        status_update = queue_coll.update_many(
            {"status": "extracted"}, {"$set": {"status": "completed"}}
        )
        if status_update.modified_count:
            logger.info(
                f"Status changed {status_update.modified_count} entries in {queue_coll_name} with extracted on completed."
            )
    logger.info(
        f"Synchronization of regexp/target/include-exclude for the task is completed: {task_name}"
    )


def sync_rta_fields(db: Database, task: Dict[str, Any]) -> None:
    """
    Updates the rta_prompt and rta_model fields:
      - In the main queue (queue_{task_name}), entries are updated if the new values differ, with the "completed" status set.
       If the collection of the rto queue (queue_rta_{task_name}) exists, it is deleted.
    """
    logger.info(f"The beginning of the synchronization of the rta fields for the task: {task.get('task_name')}")
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
            f"Update {update_result.modified_count} entries in {queue_coll_name} with the new rta_prompt and ru_model, the status has been changed to completed."
        )

    rta_coll_name = f"queue_rta_{task_name}"
    if collection_exists(db, rta_coll_name):
        db.drop_collection(rta_coll_name)
        logger.info(
            f"Collection {rta_coll_name} deleted because the rta fields were changed."
        )
    logger.info(f"Synchronization of the rta fields for the task has been completed: {task_name}")


def sync_task(db: Database, task: Dict[str, Any]) -> None:
    """
    Synchronizes the queue for a single task by sequentially updating task_name, models, prompt, variables,
    regexp/target/include-exclude and rta-fields.
    """
    logger.info(f"==== Starting task synchronization: {task.get('task_name')} ====")
    sync_task_name(db, task)
    sync_models(db, task)
    sync_prompt(db, task)
    sync_variables(db, task)
    sync_regexp_include_exclude(db, task)
    sync_rta_fields(db, task)
    logger.info(f"==== Task synchronization completed: {task.get('task_name')} ====")


def sync_all_tasks(db: Database) -> None:
    """
    Bypasses all tasks from the tasks collection and synchronizes queues for each one.
    """
    logger.info("The beginning of synchronization of all tasks.")
    tasks_coll = db["tasks"]
    tasks = list(tasks_coll.find({}))
    if not tasks:
        logger.info("There are no tasks to synchronize.")
        return
    logger.info(f"Found {len(tasks)} tasks to synchronize.")
    for task in tasks:
        sync_task(db, task)
    logger.info("Synchronization of all tasks is completed.")


def main():
    db = get_db()
    interval = 10  # verification interval in seconds
    logger.info("Starting the task synchronization cycle.")
    while True:
        sync_all_tasks(db)
        time.sleep(interval)


if __name__ == "__main__":
    main()
