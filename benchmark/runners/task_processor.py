import logging
import os
import time
from typing import Any, Dict, List

import pandas as pd
from pymongo import MongoClient
from pymongo.database import Database

from utils.constants import MONGO_HOST, MONGO_PASSWORD, MONGO_PORT, MONGO_USERNAME

MONGO_DB = os.environ.get("MONGO_DB", "TrustGen")

# Configuring logging
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
    logger.info("Successfully connected to MongoDB.")
    return client


def get_db() -> Database:
    client = get_mongo_client()
    return client[MONGO_DB]


def fetch_tasks(db: Database) -> List[Dict[str, Any]]:
    """
    Getting all the tasks from the tasks collection.
    """
    tasks_coll = db["tasks"]
    tasks = list(tasks_coll.find({}))
    return tasks


def get_dataset_head(db: Database, dataset_name: str, limit: int = None) -> pd.DataFrame:
    """
    Returns a dataset in DataFrame format from the dataset_<dataset_name> collection.
    You can limit the number of rows (limit).
    """
    coll_name = f"dataset_{dataset_name}"
    coll = db[coll_name]
    cursor = coll.find({})
    if limit:
        cursor = cursor.limit(limit)
    docs = list(cursor)
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])
    return df


def insert_queue_entries_for_task(db: Database, task: Dict[str, Any]) -> None:
    """
   For a task from the tasks collection:
      - Create entries in the queue (collection queue_<task_name>) for each row of the dataset and for each model.
      - If the record already exists (determined by the pair (line_index, model)), it is skipped.
    """
    task_type = task.get("task_type", "unknown")
    if task_type == "unknown":
        logger.warning(f"Task without task_type: {task}")
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
        logger.warning(f"The dataset for '{dataset_name}' is empty. There is nothing to process.")
        return

    queue_coll_name = f"queue_{task_name}"
    queue_coll = db[queue_coll_name]

    # Optimized selection of existing keys based only on the necessary models
    existing_keys = set()
    query = {"model": {"$in": models}}
    for entry in queue_coll.find(query, {"line_index": 1, "model": 1}):
        existing_keys.add((entry.get("line_index"), entry.get("model")))

    new_inserts = []
    rows = df.to_dict("records")
    for i, row in enumerate(rows):
        variables = {col: row.get(col, None) for col in var_cols}
        for model in models:
            key = (i, model)
            if key in existing_keys:
                continue  # the entry already exists – skip it.
            doc = {
                "task_type": task_type,
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
                f"Inserted {len(result.inserted_ids)}  new documents in '{queue_coll_name}'."
            )
        except Exception as e:
            logger.error(f"Error when inserting documents into '{queue_coll_name}': {e}")
    else:
        logger.info(f"There are no new documents to insert into '{queue_coll_name}'.")


def main():
    """
    Main cycle:
- Connect to the database.
      - Every N seconds, we go through the tasks from the tasks collection and call the queue record creation function for each one.
    """
    db = get_db()
    interval = 10  # interval in seconds

    try:
        while True:
            tasks = fetch_tasks(db)
            if tasks:
                for task in tasks:
                    logger.info(f"Task processing: {task['task_name']}")
                    insert_queue_entries_for_task(db, task)
            else:
                logger.info("There are no tasks to process.")
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Stopping the software process KeyboardInterrupt")


if __name__ == "__main__":
    main()
