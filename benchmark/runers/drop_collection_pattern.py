import os

from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables from .env file
load_dotenv()

# Retrieve connection details from environment variables
MONGO_USERNAME = os.getenv("MONGO_INITDB_ROOT_USERNAME")
MONGO_PASSWORD = os.getenv("MONGO_INITDB_ROOT_PASSWORD")
MONGO_HOST = os.getenv("MONGO_HOST")
MONGO_PORT = os.getenv("MONGO_INITDB_ROOT_PORT")

# Construct MongoDB URI
mongo_uri = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"

pattern = "rubia_"

# Connect to MongoDB
client = MongoClient(mongo_uri)
db = client["TrustLLM_ru"]

# List all collections in the database
collections = db.list_collection_names()

# Filter collections that start with 'ethics_'
collections_to_delete = [col for col in collections if col.startswith(pattern)]

# Delete the collections
for collection_name in collections_to_delete:
    db.drop_collection(collection_name)
    print(f"Collection '{collection_name}' has been deleted.")

print(f"All collections starting with {pattern} have been deleted.")
