import os
from typing import List

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database


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
    mongo_uri = (
        f"mongodb://{mongo_username}:{mongo_password}@{mongo_host}:{mongo_port}/"
    )

    return MongoClient(mongo_uri)


def delete_pending_tasks(
    db: Database, excluded_collections: List[str], query: dict
) -> None:
    """
    Deletes tasks with the 'pending' status from all database collections,
    except for the exceptions listed in the list.

    Args:
        db (Database): Database Instance MongoDB.
        excluded_collections (List[str]): A list of collections that do not need to be cleaned.
        query (dict): A condition for searching for issues (for example, {"status": "pending"}).

    Returns:
        None
    """
    # Going through all the collections in the database
    for collection_name in db.list_collection_names():
        if collection_name in excluded_collections:
            continue  # Skipping collections from the exclusion list

        collection: Collection = db[collection_name]

        # Deleting all tasks matching a query
        result = collection.delete_many(query)

        # Output of the number of deleted tasks
        print(
            f"From the collection '{collection_name}' deleted {result.deleted_count} issues with the status'pending'."
        )


def main() -> None:
    """
    The main function is to remove issues with the 'pending' status from all collections.,
    except for the excluded ones.
    """
    # Connecting to MongoDB
    client = get_mongo_client()
    db = client["TrustGen"]

    # List of excluded collections
    excluded_collections = ["delete_me", "test"]

    # A condition for searching for issues with the status 'pending'
    query = {"status": "pending"}

    # Deleting issues
    delete_pending_tasks(db, excluded_collections, query)


if __name__ == "__main__":
    main()
