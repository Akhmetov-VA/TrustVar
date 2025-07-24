import uuid
from typing import Any, Dict, List, Optional

import pandas as pd
from pymongo.collection import Collection


def replace_curl(data: str) -> str:
    """Escapes curly braces in a string for correct substitution of variables.

    Args:
        data: The line in which curly braces need to be replaced.

    Returns:
        A string with escaped curly braces.
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
    """Adds a task to the MongoDB collection.

    Args:
        collection: A MongoDB collection for inserting a task.
        row: A dictionary with data from the Data Frame.
        job_id: The unique identifier of the task.
        model: The name of the model.
        prompt: Request template.
        variables: Variables to be substituted into the template.
        target: The target value for the task (if any).

    Returns:
        ID of the inserted task.
    """
    task = {
        "job_id": job_id,
        "prompt": prompt,
        "variables": variables,
        "status": "pending",
        "model": model,
        "response": None,
        "target": target,
        **row,  # Adding data from row
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
    """Loads tasks to MongoDB from the DataFrame.

    Args:
        models: A list of model names.
        collection: A MongoDB collection for inserting tasks.
        prompt_data: A dictionary with query patterns.
        df_for_lm: DataFrame with data.
        placeholder: The name of the placeholder in the template.
        var_col: The name of the column in the DataFrame for variable substitution.
        target: The target value for the tasks (if any).
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


def filter_models(models: List[str], collection: Collection) -> List[str]:
    """Filters the list of models, excluding those that are already present in the collection..

    Args:
        models (List[str]): A list of model names.
        collection (Collection): The MongoDB collection to check.

    Returns:
        List[str]: A list of models that are not in the collection.
    """
    existing_models = set(collection.distinct("model"))
    models_to_add = [m for m in models if m not in existing_models]
    return models_to_add
