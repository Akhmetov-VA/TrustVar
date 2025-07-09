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
    # Loading environment variables from a file .env
    load_dotenv()

    # Getting connection details from environment variables
    mongo_username = os.getenv("MONGO_INITDB_ROOT_USERNAME")
    mongo_password = os.getenv("MONGO_INITDB_ROOT_PASSWORD")
    mongo_host = os.getenv("MONGO_HOST")
    mongo_port = os.getenv("MONGO_INITDB_ROOT_PORT")

    # Creating a connection URI
    mongo_uri = (
        f"mongodb://{mongo_username}:{mongo_password}@{mongo_host}:{mongo_port}/"
    )

    try:
        client = MongoClient(mongo_uri)
        # Checking the connection
        client.admin.command("ping")
        logging.info("Successful connection to MongoDB.")
        return client
    except Exception as e:
        logging.exception("Connection error toMongoDB.")
        raise e


def revert_task_status(collection: collection.Collection) -> None:
    """
    Reverses the status of tasks in the collection from 'transferred' and 'measured' to 'completed'
    and deletes the 'pred' and 'metric' fields.

    Args:
        collection (Collection): The MongoDB collection where the operation is performed.

    Returns:
        None
    """
    try:
        # Defining the rollback statuses
        statuses_to_revert = ["transferred", "measured"]

        result = collection.update_many(
            {"status": {"$in": statuses_to_revert}},
            {"$set": {"status": "completed"}, "$unset": {"pred": "", "metric": ""}},
        )
        logging.info(
            f"Cancelled {result.modified_count} issues from{statuses_to_revert} in 'completed' in the collection '{collection.name}'."
        )
    except Exception as e:
        logging.error(f"Error processing the collection '{collection.name}': {e}")


def process_collections(db: Database) -> None:
    """
    Processes all collections in the database by updating task statuses.

    Args:
        db (Database): Database Instance MongoDB.

    Returns:
        None
    """
    for collection_name in db.list_collection_names():
        collection = db[collection_name]
        logging.info(f"Collection Processing '{collection_name}'")
        revert_task_status(collection)


def main() -> None:
    """
    The main function for performing task processing in database collections.
    """
    try:
        # Configuring logging
        configure_logging()

        # Database Name
        database_name = "TrustLLM_ru"

        # Connecting to MongoDB
        client = get_mongo_client()
        db = client[database_name]

        # Processing collections
        process_collections(db)

        logging.info("All specified collections have been processed.")
    except Exception as e:
        logging.exception(f"An error has occurred: {e}")


if __name__ == "__main__":
    main()
