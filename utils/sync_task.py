import logging
from typing import Any, Dict, List

from pymongo import ReturnDocument
from pymongo.database import Database

logger = logging.getLogger(__name__)


def sync_task_once(db: Database, task_id: Any, new_models: List[str]) -> Dict[str, Any]:
    """
    Обновляет список моделей у задачи с заданным task_id в коллекции 'tasks'.
    :param db: объект базы данных (MongoDB Database)
    :param task_id: идентификатор задачи (_id)
    :param new_models: новый список моделей для записи
    :return: обновлённый документ задачи или None, если задача не найдена
    """
    collection = db["tasks"]
    updated_task = collection.find_one_and_update(
        {"_id": task_id},
        {"$set": {"models": new_models}},
        return_document=ReturnDocument.AFTER,
    )
    if updated_task:
        logger.info(f"Задача {task_id} успешно обновлена: models = {new_models}")
    else:
        logger.warning(f"Задача {task_id} не найдена или не обновлена.")
    return updated_task
