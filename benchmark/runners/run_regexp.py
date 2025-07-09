import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Union

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
    """Returns a database objectMongoDB."""
    client = get_mongo_client()
    return client[MONGO_DB]


def fetch_completed_tasks(db: Database):
    """
    We find all the tasks in the queue queues_* with the status 'completed' and the presence of the response field.
    We exclude the RtA metric, because it is processed by another script.

    Returns an iterator (coll_name, task).
    """
    collections = [c for c in db.list_collection_names() if c.startswith("queue_")]
    for coll_name in collections:
        coll = db[coll_name]
        # metric != 'RtA'
        tasks = list(
            coll.find(
                {
                    "status": "completed",
                    "response": {"$ne": None},
                    "metric": {"$ne": "RtA"},
                }
            )
        )
        for t in tasks:
            yield coll_name, t


def apply_regexp_to_response(response: Union[str, List[str]], regexp: str) -> Union[str, List[str]]:
    """
    Applying the regular schedule to the response.
    If the response is a list of strings, we apply a regular pattern to each element.
    If there is a match, we take the found value.
    If not, 'TFN'.
    Returns a string or a list of strings, depending on the type of response.
    """
    pattern = re.compile(regexp, re.DOTALL)
    
    if isinstance(response, list):
        # Processing the list of responses
        results = []
        for resp_item in response:
            match = pattern.search(resp_item)
            if match:
                # It is assumed that we take the first suitable group.
                if match.groups():
                    for g in match.groups():
                        if g is not None:
                            results.append(g)
                            break
                    else:
                        results.append(match.group(0))
                else:
                    results.append(match.group(0))
            else:
                results.append("TFN")
        return results
    else:
        # Processing one line (old logic)
        match = pattern.search(response)
        if match:
            # It is assumed that we take the first suitable group.
            if match.groups():
                for g in match.groups():
                    if g is not None:
                        return g
                return match.group(0)
            else:
                return match.group(0)
        else:
            return "TFN"


def apply_exact_match(response: str, target: Union[str, List[str]]) -> str:
    """
    For the exact_match metric:
    If the target is a list of rows, we check each one.
    If at least one is found in response, it is included in pred.
    If the target is a single row (not a list), we make it a list of one element.
    If nothing is found, pred='TFN'.
    """
    if isinstance(target, str):
        target = [target]  # Turning a row into a list

    found = []
    for t in target:
        if t in response:
            found.append(t)
    if not found:
        return "TFN"
    else:
        # We will return the list of found strings (or, for example, separated by commas).
        # For convenience, let's just have a list in the form of a string.
        return str(found)


def update_task_with_pred(db: Database, coll_name: str, task_id: Any, pred: Union[str, List[str]]):
    """
    We update the pred field in the issue and the status on extracted.
    """
    coll = db[coll_name]
    coll.update_one({"_id": task_id}, {"$set": {"pred": pred, "status": "extracted"}})
    
    # Logging information about the pred, depending on its type
    if isinstance(pred, list):
        pred_info = f"pred=[{', '.join(map(str, pred))}]"
    else:
        pred_info = f"pred={pred}"
    
    logger.info(
        f"Updated document {task_id} in {coll_name}: {pred_info}, status=extracted"
    )


def run_extraction_loop(db: Database, interval: int = 10):
    """
    Starting an endless queue polling cycle:
    - We find all the issues in the status completed (response != None) and metric != RtA
    - In relation to metric:
       1) exact_match: we use apply_exact_match
       2) include_exclude: just take the response in pred
       3) accuracy: we apply regexp to each element of the list response
       4) any others: use regexp (if available) -> apply_regexp_to_response
         if not, TFN
    - Change the status to extracted
    - Wait for interval seconds and repeat
    """
    while True:
        found_any = False
        for coll_name, task in fetch_completed_tasks(db):
            found_any = True
            task_id = task["_id"]
            response = task["response"]
            metric = task.get("metric", None)
            target = task.get("target", [])

            if metric == "exact_match":
                # exact_match logic
                pred = apply_exact_match(response, target)

            elif metric == "include_exclude":
                # By the condition "we just take the response and transfer it to pred"
                # The logic of the include/exclude check is performed by the following runner.
                pred = response

            elif metric == "accuracy":
                regexp = task.get("regexp", None)
                if not regexp:
                    logger.error(f"For the accuracy metric, regexp is not provided in the task {task_id} ({coll_name}) — the task was skipped")
                    continue
                pred = apply_regexp_to_response(response, regexp)

            else:
                regexp = task.get("regexp", None)
                if not regexp:
                    logger.error(f"For metrica {metric} regexp is not provided in the task {task_id} ({coll_name}) — the task was skipped")
                    continue
                pred = apply_regexp_to_response(response, regexp)

            update_task_with_pred(db, coll_name, task_id, pred)

        if not found_any:
            logger.info("There are no tasks to extract pred. Expectation...")
        time.sleep(interval)


def main():
    """
   Entry point:
    1) Connect to the database
    2) Start the processing cycle
    """
    db = get_db()
    run_extraction_loop(db, interval=60)


if __name__ == "__main__":
    main()
