import uuid


def replace_curl(data: str):
    return data.replace("{", "{{").replace("}", "}}")


# Функция для добавления задачи в MongoDB
def add_task(collection, job_id, model, prompt, variabels, label=None):
    task = {
        "job_id": job_id,
        "prompt": prompt,
        "variables": variabels,
        "status": "pending",
        "model": model,
        "response": None,
        "label": label,
    }
    result = collection.insert_one(task)
    print(f"Added task with id: {result.inserted_id} and job_id: {job_id}")
    return result.inserted_id


# цикл добавления задач в Mongo
def load_task_mongo(
    models,
    collection,
    prompts_data,
    df_for_llm,
    task_name,
    placeholder="text",
    var_col="prompt",
):
    # Генерация уникального идентификатора задачи для текущего запуска
    job_id = str(uuid.uuid4())

    for model in models:
        for kind, prompts in prompts_data.items():
            for i in range(len(df_for_llm)):
                row = df_for_llm.iloc[i].to_dict()
                variables = {placeholder: replace_curl(row[var_col])}
                for prompt in prompts:
                    add_task(
                        collection,
                        row,
                        job_id,
                        model,
                        f"{task_name}_{kind}",
                        prompt,
                        variables,
                    )

    print(f"All task_names for job_id {job_id} have been added.")
