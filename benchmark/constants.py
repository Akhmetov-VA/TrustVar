import os

from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()

MONGO_USERNAME = os.getenv("MONGO_INITDB_ROOT_USERNAME")
MONGO_PASSWORD = os.getenv("MONGO_INITDB_ROOT_PASSWORD")
MONGO_HOST = os.getenv("MONGO_HOST")
MONGO_PORT = os.getenv("MONGO_INITDB_ROOT_PORT")

MODELS = [
    # "gemma2:27b-instruct-q4_0",
    "gemma2:9b-instruct-q4_0",
    "ilyagusev/saiga_llama3",
    "llama2:13b",
    "llama3.1:8b-instruct-q4_0",
    # "llama3:70b-instruct-q4_0",
    "llama3:8b-instruct-q4_0",
    "mistral:7b-instruct-v0.3-q4_0",
    "mixtral:8x7b-instruct-v0.1-q4_0",
    "phi3:14b-medium-4k-instruct-q4_0",
    "qwen:7b",
    # "qwen2:72b-instruct-q4_0",
    "qwen2:7b-instruct-q4_0",
    "solar:10.7b-instruct-v1-q4_0",
    "wavecut/vikhr:7b-instruct_0.4-Q4_1",
    "yi:6b",
    "yi:9b",
]
