import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

import pandas as pd
from bson.objectid import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

# Loading environment variables from an .env file
load_dotenv()

# Configuring logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_numpy_objects(data):
    """Recursively convert numpy objects to Python native types."""
    if isinstance(data, np.ndarray):
        return data.tolist()  # Convert numpy array to list
    elif isinstance(data, dict):
        return {key: convert_numpy_objects(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_objects(item) for item in data]
    elif isinstance(data, np.generic):
        return data.item()  # Convert numpy scalars to Python scalars
    else:
        return data

class MongoDBConfig:
    """A class for managing MongoDB configurations."""

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[str] = None,
        database: str = "TrustGen",
    ):
        """
       When creating an instance of MongoDB Config, you can redefine the parameters or
       they will be taken from the environment variables.
        """
        self.username = username or os.getenv("MONGO_INITDB_ROOT_USERNAME")
        self.password = password or os.getenv("MONGO_INITDB_ROOT_PASSWORD")
        self.host = host or os.getenv("MONGO_HOST", "83.143.66.65")
        self.port = port or os.getenv("MONGO_PORT", "27363")
        self.database = database

    def get_uri(self) -> str:
        """
        Creating a URI for connecting to MongoDB.
        """
        print(f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/")
        return f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/"


class MongoDBClient:
    """A class for high-level work with MongoDB."""

    def __init__(self, config: Optional[MongoDBConfig] = None):
        """
        Initializing a MongoDB client using a preset or default config.
        """
        self.config = config or MongoDBConfig()
        try:
            self.client = MongoClient(self.config.get_uri())
            self.db = self.client[self.config.database]
            logger.info("Successfully connected to MongoDB.")
        except PyMongoError as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            raise

    # ---------------- Basic methods for working with collections ----------------

    def get_collection(self, collection_name: str) -> Collection:
        """Getting a collection by name."""
        return self.db[collection_name]

    def list_collections(self) -> List[str]:
        """Getting a list of all collections."""
        return self.db.list_collection_names()

    def list_collections_starting_with(self, prefix: str) -> List[str]:
        """Getting a list of collections starting with a specified prefix."""
        return [col for col in self.list_collections() if col.startswith(prefix)]

    def delete_collection(self, collection_name: str) -> None:
        """Deleting a collection by name."""
        try:
            self.db.drop_collection(collection_name)
            logger.info(f"Collection '{collection_name}' successfully deleted.")
        except PyMongoError as e:
            logger.error(f"Collection deletion error '{collection_name}': {e}")
            raise

    # ---------------- Methods for inserting and updating documents ----------------

    def insert_data(self, collection_name: str, data: List[Dict[str, Any]]) -> None:
        """
        Inserting multiple documents into the specified collection.
        It is used, for example, to download a dataset.
        """
        if not data:
            logger.warning("There is no data to insert.")
            return
        try:
            collection = self.get_collection(collection_name)
            collection.insert_many(data, ordered=False)
            logger.info(
                f"Inserted {len(data)} documents in the collection '{collection_name}'."
            )
        except PyMongoError as e:
            logger.error(f"Data insertion error in MongoDB: {e}")
            raise

    def insert_task(self, task_data: Dict[str, Any]):
        """
        Add a new task to the tasks collection.
        """
        coll = self.get_collection("tasks")
        coll.insert_one(task_data)

    def insert_prompt_for_dataset(self, dataset_name: str, prompt: str, name: str):
        """
        Insert a new prompt (prompt, name) into the prompt_{dataset_name} collection.
        """
        coll = self.get_collection(f"prompt_{dataset_name}")
        coll.insert_one({"name": name, "prompt": prompt})

    def insert_rta_prompt(self, prompt: str, name: str):
        """
        Insert a new RTA prompt in prompt_rta.
        """
        coll = self.get_collection("prompt_rta")
        coll.insert_one({"name": name, "prompt": prompt})

    def insert_regexp_for_metric(self, metric: str, pattern: str, name: str):
        """
        Insert a new regular expression (name, pattern) into the regexp_{metric} collection.
        """
        coll = self.get_collection(f"regexp_{metric}")
        coll.insert_one({"name": name, "pattern": pattern})

    def insert_dataset_records(self, dataset_name: str, df: pd.DataFrame):
        """
        Upload a dataset to the dataset_{dataset_name} collection.
        """
        coll_name = f"dataset_{dataset_name}"
        
        import numpy as np





        records = df.to_dict(orient="records")
        converted_records = [convert_numpy_objects(record) for record in records]
        if records:
            self.insert_data(coll_name, records)

    def insert_dataset_into_registry(self, doc: Dict[str, Any]):
        """
        Add information about the dataset to dataset_registry.
        It is expected that the doc already contains the fields:
            dataset_name, var_cols, metric, target_column, include_column, exclude_column ...
        """
        coll_name = "dataset_regestry"
        coll = self.get_collection(coll_name)
        coll.insert_one(doc)

    def update_task(self, task_id: Any, update_data: Dict[str, Any]):
        """
        Update an existing task in the tasks collection by _id.
        """
        coll = self.get_collection("tasks")
        if not isinstance(task_id, ObjectId):
            try:
                task_id = ObjectId(task_id)
            except Exception as e:
                logger.error(f"Invalid issue ID: {e}")
                raise
        coll.update_one({"_id": task_id}, {"$set": update_data})

    def update_tasks_status(self, collection_name: str, current_status: str, new_status: str) -> int:
        """
        Updating the status of issues in collection_name: current_status -> new_status.
        Returns the number of updated documents.
        """
        try:
            collection = self.get_collection(collection_name)
            result = collection.update_many(
                {"status": current_status}, {"$set": {"status": new_status}}
            )
            logger.info(
                f"Updated {result.modified_count} documents from the status '{current_status}' on '{new_status}'."
            )
            return result.modified_count
        except PyMongoError as e:
            logger.error(f"Issue status update error: {e}")
            raise

    # ---------------- Methods for calculating and obtaining documents ----------------

    def get_tasks_by_status(self, collection_name: str, status: str) -> List[Dict[str, Any]]:
        """
        Getting all tasks from collection_name that have status == status.
        """
        try:
            collection = self.get_collection(collection_name)
            tasks = list(collection.find({"status": status}))
            logger.info(f"Found {len(tasks)} issues with the status '{status}' in '{collection_name}'.")
            return tasks
        except PyMongoError as e:
            logger.error(f"Error receiving tasks by status: {e}")
            raise

    def count_tasks_by_status(self, collection_name: str, status: str) -> int:
        """
        Counting the number of tasks with the status status in collection_name.
        """
        try:
            collection = self.get_collection(collection_name)
            count = collection.count_documents({"status": status})
            logger.info(f"Number of issues with the status '{status}' in '{collection_name}': {count}.")
            return count
        except PyMongoError as e:
            logger.error(f"Error in calculating tasks by status: {e}")
            raise

    def count_total_tasks(self, collection_name: str) -> int:
        """
        Counting the total number of documents (tasks) in collection_name.
        """
        try:
            collection = self.get_collection(collection_name)
            count = collection.count_documents({})
            logger.info(
                f"Total number of issues in the collection '{collection_name}': {count}."
            )
            return count
        except PyMongoError as e:
            logger.error(f"Error in calculating the total number of tasks: {e}")
            raise

    # ---------------- Methods for reading data (tasks, datasets, prompts, regexp) ----------------

    def get_all_tasks(self) -> pd.DataFrame:
        """
        Returns all tasks from the tasks collection as a DataFrame.
        If there are no tasks, returns an empty Data Frame.
        """
        tasks_collection = self.get_collection("tasks")
        tasks = list(tasks_collection.find({}))
        if not tasks:
            return pd.DataFrame()
        return pd.DataFrame(tasks)

    def get_all_datasets(self) -> List[str]:
        """
        Retrieves a list of all datasets (names),
        based on collections starting with 'dataset_'.
        """
        collections = self.list_collections()
        dataset_colls = [col for col in collections if col.startswith("dataset_")]
        datasets = [col.replace("dataset_", "") for col in dataset_colls]
        return datasets

    def get_dataset_head(self, dataset_name: str, limit: int = 10) -> pd.DataFrame:
        """
        Returns the first 'limit' rows of the dataset dataset_{dataset_name} as a DataFrame.
        """
        coll = self.get_collection(f"dataset_{dataset_name}")
        docs = list(coll.find({}).limit(limit))
        if not docs:
            return pd.DataFrame()
        df = pd.DataFrame(docs)
        if "_id" in df.columns:
            df = df.drop(columns=["_id"])
        return df

    def get_dataset_registry_info(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a dataset (from the 'dataset_registry')
        by its dataset_name.
        """
        coll = self.get_collection("dataset_regestry")
        doc = coll.find_one({"dataset_name": dataset_name})
        return doc

    def get_prompt_docs_for_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """
        Get the full list (documents) of promptes for dataset_{dataset_name},
        in other words, the prompt_{dataset_name} collection.
        """
        coll_name = f"prompt_{dataset_name}"
        if coll_name not in self.list_collections():
            return []
        coll = self.get_collection(coll_name)
        return list(coll.find({}))

    def get_rta_prompt_docs(self) -> List[Dict[str, Any]]:
        """
        Get the full list of RTA promptes (documents) from the prompt_rta collection (if available).
        """
        if "prompt_rta" not in self.list_collections():
            return []
        coll = self.get_collection("prompt_rta")
        return list(coll.find({}))

    def get_prompts_for_dataset(self, dataset_name: str) -> List[str]:
        """
        Return a list of product names for the specified dataset.
        """
        prompts = self.get_prompt_docs_for_dataset(dataset_name)
        return [p["name"] for p in prompts if "name" in p]

    def get_rta_prompts(self) -> List[str]:
        """
        Return the list of RTA prompt names from prompt_rta.
        """
        rta_prompts = self.get_rta_prompt_docs()
        return [rp["name"] for rp in rta_prompts if "name" in rp]

    def get_regexp_docs_for_metric(self, metric: str) -> List[Dict[str, Any]]:
        """
        Get the full list of documents (name, pattern) from regexp_{metric}.
        """
        coll_name = f"regexp_{metric}"
        if coll_name not in self.list_collections():
            return []
        coll = self.get_collection(coll_name)
        return list(coll.find({}))

    def get_regexp_for_metric(self, metric: str) -> List[str]:
        """
        Get a list of the names of the controls (name) for a given metric (regexp_{metric}).
        """
        docs = self.get_regexp_docs_for_metric(metric)
        return [d["name"] for d in docs if "name" in d]

    def validate_regex(self, pattern: str) -> bool:
        """
        Check the correctness of the regular expression.
        """
        try:
            re.compile(pattern)
            return True
        except re.error:
            return False

    def list_metrics(self) -> List[str]:
        """
        Get a list of metrics based on collections starting with regexp_ or results_.
        """
        regexp_cols = self.list_collections_starting_with("regexp_")
        result_cols = self.list_collections_starting_with("results_")
        metrics_regexp = [c.replace("regexp_", "") for c in regexp_cols]
        metrics_results = [c.replace("results_", "") for c in result_cols]
        metrics = list(set(metrics_regexp + metrics_results))
        return metrics
