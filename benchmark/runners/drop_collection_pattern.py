import os
from typing import List

from dotenv import load_dotenv
from pymongo import MongoClient


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


def delete_collections_by_pattern(db, pattern: str) -> None:
    """
    Deletes collections from the MongoDB database whose names begin with the specified pattern.

    Args:
        db: An instance of the MongoDB database.
        pattern (str): The pattern that the names of the collections begin with.

    Returns:
        None
    """
    # Getting a list of all collections in the database
    collections = db.list_collection_names()

    # Filtering collections starting from a given pattern
    collections_to_delete = [col for col in collections if col.startswith(pattern)]

    # Deleting found collections
    for collection_name in collections_to_delete:
        db.drop_collection(collection_name)
        print(f"The collection '{collection_name}' has been successfully deleted.")
    print(f"All collections starting with '{pattern}' have been deleted.")


def main() -> None:
    """
    The main function is to delete collections starting from a specific pattern.
    """
    # A pattern for filtering collections
    pattern = "rubia_"

    # Connecting to MongoDB
    client = get_mongo_client()
    db = client["TrustGen"]

    # Deleting collections by pattern
    delete_collections_by_pattern(db, pattern)


if __name__ == "__main__":
    main()
