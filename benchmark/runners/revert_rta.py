import logging
import os

from pymongo import MongoClient
from pymongo.database import Database

from utils.constants import MONGO_HOST, MONGO_PASSWORD, MONGO_PORT, MONGO_USERNAME

# We use the MONGO_DB variable from the environment (or the default value)
MONGO_DB = os.environ.get("MONGO_DB", "TrustGen")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_mongo_client() -> MongoClient:
    """
    Creates a connection to MongoDB.
    """
    mongo_uri = (
        f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
    )
    client = MongoClient(mongo_uri)
    logger.info("Successfully connected to MongoDB.")
    return client


def get_db() -> Database:
    client = get_mongo_client()
    return client[MONGO_DB]


def reset_rta_tasks_to_completed(db: Database) -> None:
    """
    Passes through all the usual queues (collections starting with "queue_",
    but excluding collections of rta queues) and sets the status of documents with the metric "RtA"
    to the value "completed". This will allow the main migration process (fetch_rta_tasks)
    re-select these tasks.
    """
    # We get a list of collections starting with "queue_" and not containing "queue_rta_"
    collections = [
        c
        for c in db.list_collection_names()
        if c.startswith("queue_") and not c.startswith("queue_rta_")
    ]

    for coll_name in collections:
        coll = db[coll_name]
        # Updating documents with metric "RtA", which have a status not equal to "completed"
        result = coll.update_many(
            {"metric": "RtA", "status": {"$ne": "completed"}},
            {"$set": {"status": "completed"}},
        )
        logger.info(
            f"In the collection '{coll_name}' updated {result.modified_count} documents up to the status 'completed'."
        )


def main():
    db = get_db()
    reset_rta_tasks_to_completed(db)


if __name__ == "__main__":
    main()
