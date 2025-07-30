import logging
import os
from typing import List

from dotenv import load_dotenv
from pymongo import MongoClient, collection
from pymongo.database import Database


def configure_logging() -> None:
    """
    Configures logging for displaying messages in the console.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.info("Logging has been successfully configured.")


def get_mongo_client() -> MongoClient:
    """
    Creates and returns a connection to MongoDB based on environment variables.

    Returns:
        MongoClient: A client for connecting to MongoDB.
    """
    load_dotenv()
    mongo_username = os.getenv("MONGO_INITDB_ROOT_USERNAME")
    mongo_password = os.getenv("MONGO_INITDB_ROOT_PASSWORD")
    mongo_host = os.getenv("MONGO_HOST")
    mongo_port = os.getenv("MONGO_INITDB_ROOT_PORT")

    mongo_uri = (
        f"mongodb://{mongo_username}:{mongo_password}@{mongo_host}:{mongo_port}/"
    )

    try:
        client = MongoClient(mongo_uri)
        client.admin.command("ping")
        logging.info("Successfully connected to MongoDB.")
        return client
    except Exception as e:
        logging.exception("Connection error to MongoDB.")
        raise e


def add_source_collection_to_rta(db: Database, rta_collection_name: str) -> None:
    """
    It goes through all the collections in the database and adds the name of the source collection
    to the `RtA` collection entries based on the matching of the task IDs.

    Args:
        db (Database): An instance of the MongoDB database.
        rta_collection_name (str): The name of the collection is `Rto'.
    """
    rta_collection = db[rta_collection_name]

    for collection_name in db.list_collection_names():
        # Skipping the 'RtA` collection (target collection)
        if collection_name == rta_collection_name:
            continue

        source_collection = db[collection_name]
        logging.info(
            f"Processing the collection '{collection_name}' for matching with 'RtA'."
        )

        # Collecting all IDs from the current collection
        task_ids = list(source_collection.find({}, {"_id": 1}))
        task_ids = [doc["_id"] for doc in task_ids]

        if not task_ids:
            logging.info(f"There are no records to process in the collection '{collection_name}'.")
            continue

        # We update all entries in the RtA corresponding to these identifiers.
        result = rta_collection.update_many(
            {"init_id": {"$in": task_ids}},  # Condition: init_id is included in the list of task_ids
            {"$set": {"dataset": collection_name}},  # Adding the dataset field
        )

        logging.info(
            f"Added the 'dataset' field for {result.modifier_count} tasks"
            f"in the 'RtA' collection from the '{collection_name}' collection."
        )


def main() -> None:
    """
    The main function for performing task processing in database collections.
    """
    try:
        # Configuring logging
        configure_logging()

        # DB name and `RtA` collection
        database_name = "TrustGen"
        rta_collection_name = "RtA"

        # Connetion to MongoDB
        client = get_mongo_client()
        db = client[database_name]

        # Adding the 'source_collection' field to the `RtA` collection
        add_source_collection_to_rta(db, rta_collection_name)

        logging.info("Processing is completed, all collections are mapped to 'RtA'.")
    except Exception as e:
        logging.exception(f"Error: {e}")


if __name__ == "__main__":
    main()
