import os
from typing import List

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from utils.constants import COLLECTIONS_TO_PROCESS, MODELS


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

    # Creating a URI for connecting to MongoDB
    mongo_uri = f"mongodb://{mongo_username}:{mongo_password}@{mongo_host}:{mongo_port}"

    return MongoClient(mongo_uri)


def delete_pending_tasks(
    db: Database, collections_to_process: List[str], allowed_models: List[str]
) -> None:
    """
    Deletes entries with the 'pending' status and models that are not included in allowed_models,
    from the specified collections.

    Args:
        db (Database): An instance of the MongoDB database.
        collections_to_process (List[str]): A list of collections to process.
        allowed_models (List[str]): A list of acceptable models.

    Returns:
        None
    """
    # Creating a filter for deletion: the status is 'pending' and the model is not in allowed_models
    query = {"model": {"$nin": allowed_models}}

    for collection_name in collections_to_process:
        collection: Collection = db[collection_name]

        # Deleting all documents matching the request
        result = collection.delete_many(query)

        # Output of the number of deleted documents
        print(
            f"From the collection '{collection_name}' deleted {result.deleted_count} documents with the 'pending' status and invalid models."
        )


def main() -> None:
    """
    The main function is to delete certain records from specified collections.
    """
    # Connecting to MongoDB
    client = get_mongo_client()
    db = client["TrustGen"]

    # Deleting records
    delete_pending_tasks(db, COLLECTIONS_TO_PROCESS, MODELS)


if __name__ == "__main__":
    main()
