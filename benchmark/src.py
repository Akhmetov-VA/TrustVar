# Функция для добавления задачи в MongoDB
def add_task_name(
    collection,
    task_name_data,
    job_id,
    model,
    task_name,
    prompt,
    variabels,
):
    task_name = {
        "job_id": job_id,
        "prompt": prompt,
        "variables": variabels,
        "task_name": task_name,
        "status": "pending",
        "model": model,
        "response": None,
    }
    task_name.update(task_name_data)
    result = collection.insert_one(task_name)
    print(f"Added task_name with id: {result.inserted_id} and job_id: {job_id}")
    return result.inserted_id
