import logging
import time
from typing import Any, Dict, List
import string

import requests
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from utils.constants import (
    API_URL,
    AUGMENT_MODEL,
    AUGMENT_PROMPT,
    MONGO_HOST,
    MONGO_PASSWORD,
    MONGO_PORT,
    MONGO_USERNAME,
)


def configure_logging() -> None:
    """
    Настраивает логирование для отображения сообщений в консоли.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.info("Логирование успешно настроено.")


def get_mongo_client() -> MongoClient:
    """
    Создает подключение к MongoDB на основе переменных окружения.

    Returns:
        MongoClient: Экземпляр MongoDB клиента.
    """
    logging.info("Попытка подключения к MongoDB...")
    mongo_uri = (
        f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
    )
    try:
        client = MongoClient(mongo_uri)
        logging.info("Успешно подключились к MongoDB.")
        return client
    except Exception as e:
        logging.exception("Ошибка подключения к MongoDB.")
        raise e


def make_request(
    model: str, prompt: str, session: requests.Session, variables: dict = None
) -> Dict:
    """
    Отправляет POST-запрос к API с указанной моделью, промптом и переменными.
    """
    if variables is None:
        variables = {}
    logging.info(
        f"Отправка запроса к API для модели '{model}' с промптом: {prompt[:100]}..."
    )
    logging.debug(f"make_request input: model={model}, prompt={prompt}, variables={variables}")
    try:
        response = session.post(
            API_URL,
            json={
                "model": model,
                "stream": False,
                "prompt": prompt,
                "variables": variables,
            },
        )
        response.raise_for_status()
        logging.info(f"API raw response: {response.text}")
        logging.info(f"Успешный ответ от API для модели '{model}'.")
        if response.json() is None:
            raise Exception("null response")
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Ошибка при выполнении запроса к API для модели '{model}': {e}")
        raise e


def extract_text_from_response(response: Dict) -> str:
    """
    Извлекает текст из ответа API.
    
    Args:
        response (Dict): Ответ от API.
        
    Returns:
        str: Извлеченный текст или None, если не удалось извлечь.
    """
    if isinstance(response, dict):
        # Ищем стандартные ключи с текстом
        for key in ["response", "text", "content", "result", "output"]:
            if key in response and isinstance(response[key], str):
                return response[key]
        
        # Если не нашли стандартные ключи, берем первый строковый ключ
        for key, value in response.items():
            if isinstance(value, str):
                return value
        
        logging.error(f"Не удалось извлечь текст из ответа: {response}")
        return None
    elif isinstance(response, str):
        return response
    else:
        logging.error(f"Неожиданный формат ответа: {type(response)}")
        return None


def format_prompt_with_variables(prompt: str, variables: Dict[str, Any]) -> str:
    """
    Форматирует промпт с переменными. Если переменная не найдена, возвращает исходный prompt.
    """
    try:
        return prompt.format(**variables)
    except KeyError as e:
        logging.warning(f"Переменная {e} не найдена в промпте, используем исходный промпт")
        return prompt


def generate_answer_by_augmentations(
    dynamic_augments: List[str],
    model: str,
    prompt: str,
    variables: Dict[str, Any],
    session: requests.Session,
) -> List[Dict]:
    """
    Генерирует ответы по техникам аугментации:
    для каждой техники сначала получаем аугментированный текст,
    а затем подставляем его как новый prompt в основную модель.
    """
    logging.debug(f"generate_answer_by_augmentations input: dynamic_augments={dynamic_augments}, model={model}, prompt={prompt}, variables={variables}")
    responses = []
    
    for augment_technique in dynamic_augments:
        # Формируем промпт для модели-аугментатора
        augmenter_prompt = (
            AUGMENT_PROMPT
            + f"""[Техника]:\n            {augment_technique}\n            [Исходный текст]:\n            {prompt}\n            [Ответ]:"""
        )
        logging.debug(f"Augmenter prompt: {augmenter_prompt}")
        
        # 1) Запрашиваем аугментацию
        augmented_resp = make_request(AUGMENT_MODEL, augmenter_prompt, session, variables)
        
        # 2) Извлекаем аугментированный текст
        augmented_text = extract_text_from_response(augmented_resp)
        if augmented_text is None:
            logging.error(f"Не удалось извлечь текст для аугментации {augment_technique}")
            continue
        
        logging.info(
            f"Аугментированный текст (technique={augment_technique}): {augmented_text[:100]}..."
        )

        # 3) Подставляем переменные в аугментированный текст
        augmented_prompt_with_vars = format_prompt_with_variables(augmented_text, variables)
        
        # 4) Отправляем аугментированный промпт в основную модель
        final_resp = make_request(model, augmented_prompt_with_vars, session, variables)
        responses.append(final_resp)

    return responses


def process_ordinary_task(
    task: Dict, collection: Collection, session: requests.Session
) -> None:
    """
    Обрабатывает отдельную задачу, отправляя запрос к модели и обновляя статус задачи в базе данных.

    Args:
        task (Dict): Документ задачи из MongoDB.
        collection (Collection): Коллекция MongoDB, содержащая задачи.
        session (requests.Session): Сессия requests для повторного использования соединений.
    """
    task_id = task["_id"]
    logging.info(f"Начало обработки задачи с id: {task_id}")
    prompt = task["prompt"]
    model = task["model"]
    variables = task.get("variables", {})

    try:
        # Форматируем промпт с переменными
        formatted_prompt = format_prompt_with_variables(prompt, variables)
        
        # Отправляем запрос
        response = make_request(model, formatted_prompt, session, variables)
        
        collection.update_one(
            {"_id": task_id},
            {"$set": {"status": "completed", "response": response}},
        )
        logging.info(
            f"Задача с id: {task_id} успешно завершена и обновлена в базе данных."
        )
    except Exception as e:
        collection.update_one(
            {"_id": task_id},
            {"$set": {"status": "error", "error": str(e)}},
        )
        logging.error(f"Ошибка обработки задачи с id: {task_id}: {e}")


def process_augment_task(
    task: Dict, collection: Collection, session: requests.Session
) -> None:
    """
    Обрабатывает отдельную задачу с аугментацией, отправляя запрос к модели и обновляя статус задачи в базе данных.

    Args:
        task (Dict): Документ задачи из MongoDB.
        collection (Collection): Коллекция MongoDB, содержащая задачи.
        session (requests.Session): Сессия requests для повторного использования соединений.
    """
    task_id = task["_id"]
    logging.info(f"Начало обработки задачи с id: {task_id}")
    prompt = task["prompt"]
    model = task["model"]
    variables = task.get("variables", {})
    dynamic_augments = task.get("dynamic_augments", [])
    
    try:
        responses = generate_answer_by_augmentations(
            dynamic_augments, model, prompt, variables, session
        )

        collection.update_one(
            {"_id": task_id},
            {"$set": {"status": "completed", "response": responses}},
        )

        logging.info(f"Task: {task_id} augmented and updated in DB.")
    except Exception as e:
        collection.update_one(
            {"_id": task_id},
            {"$set": {"status": "error", "error": str(e)}},
        )
        logging.error(f"Ошибка обработки задачи с id: {task_id}: {e}")


def process_collection(
    db: Database, collection_name: str, session: requests.Session
) -> None:
    """
    Обрабатывает задачи в указанной коллекции.

    Args:
        db (Database): Экземпляр базы данных MongoDB.
        collection_name (str): Название коллекции.
        session (requests.Session): Сессия requests для повторного использования соединений.
    """
    logging.info(f"Начало обработки коллекции '{collection_name}'.")
    collection = db[collection_name]
    unique_models = collection.distinct("model")

    if not unique_models:
        logging.warning(
            f"В коллекции '{collection_name}' отсутствуют модели для обработки."
        )
        return

    logging.info(
        f"Найдено {len(unique_models)} уникальных моделей в коллекции '{collection_name}'."
    )
    for model in unique_models:
        logging.info(
            f"Обработка задач для модели '{model}' в коллекции '{collection_name}'."
        )
        while True:
            ordinary_task = collection.find_one_and_update(
                {"status": "pending", "model": model},
                {"$set": {"status": "processing"}},
                return_document=False,
            )

            if ordinary_task:
                logging.info(
                    f"Найдена задача с id: {ordinary_task['_id']} для обработки."
                )
                process_ordinary_task(ordinary_task, collection, session)
                continue

            augment_task = collection.find_one_and_update(
                {"status": "augmenting", "model": model},
                {"$set": {"status": "processing"}},
                return_document=False,
            )

            if augment_task:
                logging.info(
                    f"Найдена задача с id: {augment_task['_id']} для обработки."
                )
                process_augment_task(augment_task, collection, session)
                continue

            else:
                logging.info(
                    f"Нет ожидающих задач для модели '{model}' в коллекции '{collection_name}'."
                )
                break


def run_processing_loop(db: Database) -> None:
    """
    Запускает цикл обработки задач во всех коллекциях.

    Args:
        db (Database): Экземпляр базы данных MongoDB.
    """
    logging.info("Запуск основного цикла обработки задач.")
    session = requests.Session()

    try:
        collections_to_process = [
            col
            for col in db.list_collection_names()
            if col not in ["delete_me", "test"]
        ]

        logging.info(f"Найдено {len(collections_to_process)} коллекций для обработки.")
        for collection_name in collections_to_process:
            process_collection(db, collection_name, session)

        logging.info("Все коллекции обработаны. Ожидание новых задач...")
        time.sleep(5)
    except Exception as e:
        logging.exception(f"Ошибка в процессе обработки: {e}")


def main() -> None:
    """
    Основная функция для запуска обработки задач в MongoDB.
    """
    configure_logging()
    logging.info("Загрузка переменных окружения и инициализация подключения...")
    client = get_mongo_client()
    DB_NAME = "TrustGen"
    while True:
        db = client[DB_NAME]
        run_processing_loop(db)
        time.sleep(10)


if __name__ == "__main__":
    main()
