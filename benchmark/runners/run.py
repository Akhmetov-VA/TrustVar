import logging
import time
from typing import Any, Dict, List

import requests
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from utils.constants import (
    API_URL,
    AUGMENT_MODEL,
    CURRENT_AUGMENT_PROMPT,
    MONGO_HOST,
    MONGO_PASSWORD,
    MONGO_PORT,
    MONGO_USERNAME,
)


def configure_logging() -> None:
    """
    Configures logging for displaying messages in the console.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.info("Logging has been successfully configured.")


def get_mongo_client() -> MongoClient:
    """
    Creates a connection to MongoDB based on environment variables.

    Returns:
        MongoClient: An instance of the MongoDB client.
    """
    logging.info("Trying to connect to MongoDB...")
    mongo_uri = (
        f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
    )
    try:
        client = MongoClient(mongo_uri)
        logging.info("Successfully connected to MongoDB.")
        return client
    except Exception as e:
        logging.exception("Error connecting to MongoDB.")
        raise e


def make_request(
    model: str, prompt: str, session: requests.Session, variables: dict = None
) -> Dict:
    """
    Sends a POST request to the API with the specified model, prompt, and variables.
    """
    if variables is None:
        variables = {}
    logging.info(
        f"Sending an API request for a model '{model}' with promptness: {prompt[:100]}..."
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
        logging.info(f"Successful response from the API for the model '{model}'.")
        if response.json() is None:
            raise Exception("null response")
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error when making an API request for the model '{model}': {e}")
        raise e


def extract_text_from_response(response: Dict) -> str:
    """
    Extracts the text from the response API.
    
    Args:
        response (Dict): API response.
        
    Returns:
        str: Extracted text or None if it was not possible to extract.
    """
    if isinstance(response, dict):
        # Looking for standard keys with text
        for key in ["response", "text", "content", "result", "output"]:
            if key in response and isinstance(response[key], str):
                return response[key]
        
        # If the standard keys are not found, we take the first string key.
        for key, value in response.items():
            if isinstance(value, str):
                return value
        
        logging.error(f"Couldn't extract text from the response: {response}")
        return None
    elif isinstance(response, str):
        return response
    else:
        logging.error(f"Unexpected response format: {type(response)}")
        return None


def format_prompt_with_variables(prompt: str, variables: Dict[str, Any]) -> str:
    """
    Formats prompt with variables. If the variable is not found, returns the original prompt..
    """
    try:
        return prompt.format(**variables)
    except KeyError as e:
        logging.warning(f"The variable {e} was not found in the prompt, we use the original prompt")
        return prompt


def generate_answer_by_augmentations(
    dynamic_augments: List[str],
    model: str,
    prompt: str,
    variables: Dict[str, Any],
    session: requests.Session,
) -> List[Dict]:
    """
    Generates responses based on augmentation techniques:
    for each technique, we first get a reasoned text.,
    and then we insert it as a new prompt into the main model.
    """
    logging.debug(f"generate_answer_by_augmentations input: dynamic_augments={dynamic_augments}, model={model}, prompt={prompt}, variables={variables}")
    responses = []
    
    for augment_technique in dynamic_augments:
        # Creating a prompt for the augmentator model
        augmenter_prompt = (
            CURRENT_AUGMENT_PROMPT
            + f"""[Техника]:\n            {augment_technique}\n            [Исходный текст]:\n            {prompt}\n            [Ответ]:"""
        )
        logging.debug(f"Augmenter prompt: {augmenter_prompt}")
        
        # 1) Requesting an augmentation
        augmented_resp = make_request(AUGMENT_MODEL, augmenter_prompt, session, variables)
        
        # 2) Extracting the augmented text
        augmented_text = extract_text_from_response(augmented_resp)
        if augmented_text is None:
            logging.error(f"Couldn't extract text for augmentation {augment_technique}")
            continue
        
        logging.info(
            f"Augmented text (technique={augment_technique}): {augmented_text[:100]}..."
        )

        # 3) Substituting variables into the augmented text
        augmented_prompt_with_vars = format_prompt_with_variables(augmented_text, variables)
        
        # 4) We are sending the augmented prompt to the main model
        final_resp = make_request(model, augmented_prompt_with_vars, session, variables)
        responses.append(final_resp)

    return responses


def process_ordinary_task(
    task: Dict, collection: Collection, session: requests.Session
) -> None:
    """
    Processes a separate task by sending a request to the model and updating the task status in the database.

    Args:
        task (Dict): The task document is from MongoDB.
        collection (Collection): A MongoDB collection containing tasks.
        session (request.Session): The requests session is for connection reuse.
    """
    task_id = task["_id"]
    logging.info(f"Starting task processing with id: {task_id}")
    prompt = task["prompt"]
    model = task["model"]
    variables = task.get("variables", {})

    try:
        # Formatting the prompt with variables
        formatted_prompt = format_prompt_with_variables(prompt, variables)
        
        # Sending a request
        response = make_request(model, formatted_prompt, session, variables)
        
        collection.update_one(
            {"_id": task_id},
            {"$set": {"status": "completed", "response": response}},
        )
        logging.info(
            f"The task with id: {task_id} has been successfully completed and updated in the database."
        )
    except Exception as e:
        collection.update_one(
            {"_id": task_id},
            {"$set": {"status": "error", "error": str(e)}},
        )
        logging.error(f"Error processing an issue with an id: {task_id}: {e}")


def process_augment_task(
    task: Dict, collection: Collection, session: requests.Session
) -> None:
    """
    Processes a separate augmentation task by sending a request to the model and updating the task status in the database.

    Args:
        task (Dict): The task document is from MongoDB.
        collection: A MongoDB collection containing tasks.
        session (request.Session): The requests session is for connection reuse.
    """
    task_id = task["_id"]
    logging.info(f"Starting task processing with id: {task_id}")
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
        logging.error(f"Error processing an issue with an id: {task_id}: {e}")


def process_collection(
    db: Database, collection_name: str, session: requests.Session
) -> None:
    """
    Processes tasks in the specified collection.

    Args:
        db (Database): An instance of the MongoDB database.
        collection_name (str): The name of the collection.
        session (request.Session): The requests session is for connection reuse.
    """
    logging.info(f"Start of collection processing'{collection_name}'.")
    collection = db[collection_name]
    unique_models = collection.distinct("model")

    if not unique_models:
        logging.warning(
            f"In the collection'{collection_name}' there are no models for processing."
        )
        return

    logging.info(
        f"Found {len(unique_models)} unique models in the collection '{collection_name}'."
    )
    for model in unique_models:
        logging.info(
            f"Processing tasks for the model '{model}' in the collection '{collection_name}'."
        )
        while True:
            ordinary_task = collection.find_one_and_update(
                {"status": "pending", "model": model},
                {"$set": {"status": "processing"}},
                return_document=False,
            )

            if ordinary_task:
                logging.info(
                    f"An issue with the id was found: {ordinary_task['_id']} for processing."
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
                    f"An issue with the id was found: {augment_task['_id']} for processing."
                )
                process_augment_task(augment_task, collection, session)
                continue

            else:
                logging.info(
                    f"There are no pending tasks for the model '{model}' in the collection '{collection_name}'."
                )
                break


def run_processing_loop(db: Database) -> None:
    """
    Starts the task processing cycle in all collections.

    Args:
        db (Database): MongoDB Database Instance.
    """
    logging.info("Starting the main task processing cycle.")
    session = requests.Session()

    try:
        collections_to_process = [
            col
            for col in db.list_collection_names()
            if col not in ["delete_me", "test"]
        ]

        logging.info(f"Found {len(collections_to_process)} collections to process.")
        for collection_name in collections_to_process:
            process_collection(db, collection_name, session)

        logging.info("All collections have been processed. Waiting for new tasks...")
        time.sleep(5)
    except Exception as e:
        logging.exception(f"Error in the processing process: {e}")


def main() -> None:
    """
    The main function for starting task processing in MongoDB.
    """
    configure_logging()
    logging.info("Loading environment variables and initializing the connection...")
    client = get_mongo_client()
    DB_NAME = "TrustGen"
    while True:
        db = client[DB_NAME]
        run_processing_loop(db)
        time.sleep(10)


if __name__ == "__main__":
    main()
