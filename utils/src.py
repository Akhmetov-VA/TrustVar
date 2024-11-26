import uuid
from typing import Any, Dict, List, Optional

import pandas as pd
from pymongo.collection import Collection


def replace_curl(data: str) -> str:
    """Экранирует фигурные скобки в строке для корректной подстановки переменных.

    Args:
        data: Строка, в которой необходимо заменить фигурные скобки.

    Returns:
        Строка с экранированными фигурными скобками.
    """
    return data.replace("{", "{{").replace("}", "}}")


def add_task(
    collection: Collection,
    row: Dict[str, Any],
    job_id: str,
    model: str,
    prompt: str,
    variables: Dict[str, Any],
    target: Optional[Any] = None,
) -> Any:
    """Добавляет задачу в коллекцию MongoDB.

    Args:
        collection: Коллекция MongoDB для вставки задачи.
        row: Словарь с данными из DataFrame.
        job_id: Уникальный идентификатор задачи.
        model: Название модели.
        prompt: Шаблон запроса.
        variables: Переменные для подстановки в шаблон.
        target: Целевое значение для задачи (если есть).

    Returns:
        Идентификатор вставленной задачи.
    """
    task = {
        "job_id": job_id,
        "prompt": prompt,
        "variables": variables,
        "status": "pending",
        "model": model,
        "response": None,
        "target": target,
        **row,  # Добавляем данные из row
    }
    result = collection.insert_one(task)
    print(f"Added task with id: {result.inserted_id} and job_id: {job_id}")
    return result.inserted_id


def load_task_mongo(
    models: List[str],
    collection: Collection,
    prompts_data: Dict[str, List[str]],
    df_for_llm: pd.DataFrame,
    placeholder: str = "text",
    var_col: str = "prompt",
    target: Optional[Any] = None,
) -> None:
    """Загружает задачи в MongoDB из DataFrame.

    Args:
        models: Список названий моделей.
        collection: Коллекция MongoDB для вставки задач.
        prompts_data: Словарь с шаблонами запросов.
        df_for_llm: DataFrame с данными.
        placeholder: Название плейсхолдера в шаблоне.
        var_col: Название столбца в DataFrame для подстановки переменных.
        target: Целевое значение для задач (если есть).
    """
    job_id = str(uuid.uuid4())
    for model in models:
        for kind, prompts in prompts_data.items():
            for _, row in df_for_llm.iterrows():
                variables = {placeholder: replace_curl(row[var_col])}
                for prompt in prompts:
                    add_task(
                        collection,
                        row.to_dict(),
                        job_id,
                        model,
                        prompt,
                        variables,
                        target=target,
                    )
    print(f"All tasks for job_id {job_id} have been added.")
