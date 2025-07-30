import logging
import os

from dotenv import load_dotenv
from pymongo import MongoClient
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


def add_target_field_to_collection(db: Database, collection_name: str, value) -> None:
    """
    Adds a 'target' field with a value of 0 for all documents in the specified collection.

    Args:
        db (Database): MongoDB database instance.
        collection_name (str): The name of the collection to which the field is being added.
    """
    collection = db[collection_name]
    logging.info(f"Adding the 'target' field to the collection '{collection_name}'.")

    try:
        # We update all documents by adding the target field with the value 0
        result = collection.update_many(
            {},  # Condition: all documents
            {
                "$set": {"task_name": value}
            },  # Action: add the target field with the value 0
        )
        logging.info(
            f"Added the 'target' field with value 0 for {result.modified_count} documents in the '{collection_name}' collection."
        )
    except Exception as e:
        logging.error(
            f"Error when adding the 'target' field to the collection'{collection_name}': {e}"
        )


def main() -> None:
    """
    The main function is to add the 'target' field to the collection.
    """
    try:
        # Configuring logging
        configure_logging()

        # Database and collection name
        database_name = "TrustGen"
        collection_name = "queue_rta_Misuse_ru"
        value = "Misuse_ru"

        # MongoDB Connection
        client = get_mongo_client()
        db = client[database_name]

        # Addinf 'target' field
        add_target_field_to_collection(db, collection_name, value)

        logging.info("The addition of the 'target' field has been completed.")
    except Exception as e:
        logging.exception(f"Error: {e}")


if __name__ == "__main__":
    main()
