import logging
from typing import Any, Dict, List

from pymongo import ReturnDocument
from pymongo.database import Database

logger = logging.getLogger(__name__)


def sync_task_once(db: Database, task_id: Any, new_models: List[str]) -> Dict[str, Any]:
    """
   Updates the list of models for a task with the specified task_id in the 'tasks' collection.
    :param db: database object (MongoDB Database)
    :param task_id: task id (_id)
    :param new_models: new list of models to write
    :return: updated task document or None if task is not found
    """
    collection = db["tasks"]
    updated_task = collection.find_one_and_update(
        {"_id": task_id},
        {"$set": {"models": new_models}},
        return_document=ReturnDocument.AFTER,
    )
    if updated_task:
        logger.info(f"Task {task_id} successfully updated: models = {new_models}")
    else:
        logger.warning(f"Task {task_id} not found or updated.")
    return updated_task
