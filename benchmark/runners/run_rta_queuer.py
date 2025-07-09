import logging
import os
import time
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database

from utils.constants import MONGO_HOST, MONGO_PASSWORD, MONGO_PORT, MONGO_USERNAME

# It is assumed that the environment variables for MONGO_USERNAME, MANGO_PASSWORD, MANGO_HOST, MANGO_SPORT, MONGO_DB are already set.
MONGO_DB = os.environ.get("MONGO_DB", "TrustGen")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_mongo_client() -> MongoClient:
    """
    Creating a connection to MongoDB.
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


def fetch_rta_tasks(db: Database):
    """
    Generator function: we go through all collections whose names start with "queue_"
    and select tasks with the "RtA" metric and the "completed" status.
    Returning the tuple (coll_name, task).
    """
    collections = [c for c in db.list_collection_names() if c.startswith("queue_")]
    for coll_name in collections:
        coll = db[coll_name]
        tasks = list(coll.find({"metric": "RtA", "status": "completed"}))
        for t in tasks:
            yield coll_name, t


def create_rta_queue_entry(db: Database, coll_name: str, task: Dict[str, Any]) -> None:
    """
    Transferring the task from the regular queue (queue_{task_name}) to the target rta queue (rta_queue_{task_name}).
    Logic:
      - We get task_name from the initial queue name and form rta_queue_{task_name}.
      - The following are copied to the target record:
    init_model = original model,
          init_prompt = original prompt,
          prompt = rta_prompt from the task,
          model = ru_model from the task.
      - A new variables field is formed, in which:
    "input" = filled in the original prompt with the variables substitution,
            "answer" = response.
       The required fields are checked: ru_model, rta_prompt and response.
      - If a duplicate is found (based on rta_model and already filled in variables fields),
        the record is not created, and the original one is marked as erroneous.
      - After successful transfer, the original task is updated – its status changes to 'transfered_to_rta'.
      - The "source_id" field is added to the new record for subsequent synchronization.
    """
    # Extracting task_name from coll_name: coll_name = "queue_{task_name}"
    task_name = coll_name.replace("queue_", "")
    rta_coll_name = f"queue_rta_{task_name}"
    rta_coll = db[rta_coll_name]

    # We get the necessary fields
    original_model = task["model"]  # The original model (init_model)
    rta_model = task.get("rta_model")
    if not rta_model:
        logger.warning("An RtA task without rta_model? Skip it.")
        db[coll_name].update_one(
            {"_id": task["_id"]},
            {"$set": {"status": "error", "error": "The RtA task without rta_model"}},
        )
        return

    original_prompt = task["prompt"]  # the original prompt (init_prompt)
    rta_prompt = task.get("rta_prompt")
    if not rta_prompt:
        logger.warning("An RtA task without rta_model? Skip it.")
        db[coll_name].update_one(
            {"_id": task["_id"]},
            {"$set": {"status": "error", "error": "The RtA task without rta_prompt"}},
        )
        return

    variables = task.get("variables", {})
    response = task.get("response", "")
    if response is None:
        logger.warning("An RtA task without a response? Skip it")
        db[coll_name].update_one(
            {"_id": task["_id"]},
            {"$set": {"status": "error", "error": "RtA task without response"}},
        )
        return

    # We form filled_input: we substitute variables in the original prompt
    try:
        filled_input = original_prompt.format(**variables)
    except Exception as e:
        logger.error(f"Formatting error prompt: {e}")
        db[coll_name].update_one(
            {"_id": task["_id"]},
            {"$set": {"status": "error", "error": "Formatting error prompt"}},
        )
        return

    new_variables = {"input": filled_input, "answer": response}

    # We check for a duplicate in the rta queue (by rta_model and filled in fields)
    existing = rta_coll.find_one(
        {
            "init_model": original_model,
            "variables": new_variables,
        }
    )
    if existing:
        logger.info("Duplicate found, do not add entry to rta_queue.")
        db[coll_name].update_one(
            {"_id": task["_id"]},
            {"$set": {"status": "error", "error": "Duplicate in rta_queue"}},
        )
        return

    # Creating a new document for the rta queue by adding the source_id field for subsequent synchronization
    doc = {
        "task_name": task.get("task_name"),
        "dataset_name": task.get("dataset_name"),
        "init_prompt": original_prompt,
        "init_model": original_model,
        "regexp": task.get("regexp"),
        "prompt": rta_prompt,
        "model": rta_model,
        "variables": new_variables,
        "status": "pending",  # a new record is awaiting processing
        "metric": "accuracy",  # according to the condition
        "target": task.get("target"),
        "source_id": task["_id"],  # link to the original entry in the regular queue
    }

    # We insert it into the rta queue
    rta_coll.insert_one(doc)
    logger.info(f"The RtA task has been added to {rta_coll_name}.")

    # Oupdating the original task – changing the status to 'transfered_to_rta'
    db[coll_name].update_one(
        {"_id": task["_id"]}, {"$set": {"status": "transfered_to_rta"}}
    )


def run_rta_transfer_loop(db: Database, interval: int = 10):
    """
    Endless loop:
      - We are looking for RtA tasks (metric=Ru, status=completed) in regular queues and transfer them to the rta queue.
      - Then we perform synchronization: we update the rta queues based on up-to-date data from regular queues.
      - If there are no tasks to transfer, we wait for the specified time.
    """
    while True:
        found_any = False
        for coll_name, task in fetch_rta_tasks(db):
            found_any = True
            create_rta_queue_entry(db, coll_name, task)

        if not found_any:
            logger.info("There are no RtA tasks to transfer. Expectation...")

        time.sleep(interval)


def main():
    db = get_db()
    run_rta_transfer_loop(db, interval=10)


if __name__ == "__main__":
    main()
