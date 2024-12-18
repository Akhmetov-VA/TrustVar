import logging
import time
from datetime import datetime

import pandas as pd
from pymongo import MongoClient

from utils.constants import (
    COLLECTIONS_TO_PROCESS,
    MONGO_HOST,
    MONGO_PASSWORD,
    MONGO_PORT,
    MONGO_USERNAME,
    RTA_MODEL,
    RTA_PROMPTS,
)
from utils.src import add_task


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.info("Logging configured successfully.")


def get_mongo_client() -> MongoClient:
    logging.info("Attempting to connect to MongoDB...")
    mongo_uri = (
        f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
    )
    try:
        client = MongoClient(mongo_uri)
        client.admin.command("ping")
        logging.info("Connected to MongoDB successfully.")
        return client
    except Exception as e:
        logging.exception("Failed to connect to MongoDB.")
        raise e


class RtAQueueAdder:
    def __init__(self, db):
        self.db = db
        self.rta_collection = db["RtA"]

    def add_tasks_to_rta_queue(self, collection_name: str) -> None:
        collection = self.db[collection_name]
        logging.info(
            f"Processing collection '{collection_name}' for RtA queue addition."
        )

        # Find tasks that are completed, have a response, and are not transferred yet
        tasks_cursor = collection.find(
            {
                "response": {"$ne": None},
                "transferred_to_rta": {"$ne": True},
            }
        )

        tasks_list = list(tasks_cursor)
        if not tasks_list:
            logging.info(f"No tasks to add from collection '{collection_name}'.")
            return

        df_for_llm = pd.DataFrame(tasks_list)
        df_for_llm["input"] = df_for_llm.apply(
            lambda x: x["prompt"].format(**x.get("variables", {})), axis=1
        )

        df_for_llm = df_for_llm.rename(
            {
                "model": "init_model",
                "response": "answer",
                "_id": "init_id",
            },
            axis=1,
        )

        df_for_llm["dataset"] = collection_name
        df_for_llm = df_for_llm[
            ["init_id", "job_id", "input", "init_model", "answer", "dataset"]
        ]

        # Get existing init_ids in RtA collection to avoid duplicates
        existing_tasks_cursor = self.rta_collection.find(
            {"init_id": {"$in": df_for_llm["init_id"].tolist()}}, {"init_id": 1}
        )
        existing_init_ids = {doc["init_id"] for doc in existing_tasks_cursor}

        df_for_llm = df_for_llm[~df_for_llm["init_id"].isin(existing_init_ids)]

        if df_for_llm.empty:
            logging.info(
                f"All tasks from collection '{collection_name}' are already in the RtA queue."
            )
            return

        for _, row in df_for_llm.iterrows():
            variables = {"input": row["input"], "answer": row["answer"]}
            for prompt in RTA_PROMPTS["check"]:
                add_task(
                    self.rta_collection,
                    row.to_dict(),
                    row["job_id"],
                    RTA_MODEL,
                    prompt,
                    variables,
                    target=1,
                )

        # Mark tasks as transferred to RtA
        collection.update_many(
            {"_id": {"$in": df_for_llm["init_id"].tolist()}},
            {"$set": {"transferred_to_rta": True}},
        )

        logging.info(
            f"Added {len(df_for_llm)} tasks from collection '{collection_name}' to the RtA queue."
        )


EXCLUDED_COLLECTIONS = {
    # "rubia_pro",
    # "rubia_anti",
    # "ethics_per",
    # "ethics_sit",
    # "SLAVA_only4",
    # "ruBia_short_12_11",
    # "ruhatespeech",
    # "LIBRA_4k",
    # "rublimp",
}


def main() -> None:
    configure_logging()
    client = get_mongo_client()
    db = client["TrustLLM_ru"]

    rta_queue_adder = RtAQueueAdder(db)

    while True:
        try:
            for collection_name in COLLECTIONS_TO_PROCESS:
                if collection_name in EXCLUDED_COLLECTIONS:
                    logging.info(f"Skipping collection: {collection_name}")
                    continue

                rta_queue_adder.add_tasks_to_rta_queue(collection_name)

            logging.info("Sleeping for 60 mins before next iteration.")
            time.sleep(60 * 60)
        except Exception as e:
            logging.exception(f"An error occurred: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()
