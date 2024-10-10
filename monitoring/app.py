import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# Получение данных для подключения из переменных окружения
MONGO_USERNAME = os.getenv("MONGO_INITDB_ROOT_USERNAME")
MONGO_PASSWORD = os.getenv("MONGO_INITDB_ROOT_PASSWORD")
MONGO_HOST = os.getenv("MONGO_HOST")
MONGO_PORT = os.getenv("MONGO_INITDB_ROOT_PORT")

# Формирование URI для подключения к MongoDB
mongo_uri = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"

# Подключение к MongoDB
client = MongoClient(mongo_uri)

# Выбор базы данных
db = client.TrustLLM_ru

st.title("Дашборд состояния экспериментов")

# Получение списка коллекций для обработки, исключая определенные
collections_to_process = [
    col for col in db.list_collection_names() if col not in ["delete_me", "test"]
]


# Функция для загрузки данных
def load_data():
    data = []
    for collection_name in collections_to_process:
        collection = db[collection_name]
        total_tasks = collection.count_documents({})
        pending_tasks = collection.count_documents({"status": "pending"})
        completed_tasks = collection.count_documents({"status": "completed"})
        failed_tasks = collection.count_documents({"status": "failed"})

        # Определение статуса коллекции
        if failed_tasks > 0:
            status = "Ошибка"
        elif pending_tasks > 0:
            status = "В процессе"
        else:
            status = "Завершено"

        # Добавление данных в список
        data.append(
            {
                "Коллекция": collection_name,
                "Всего задач": total_tasks,
                "Выполнено": completed_tasks,
                "В ожидании": pending_tasks,
                "С ошибками": failed_tasks,
                "Статус": status,
            }
        )
    return pd.DataFrame(data)


# Добавляем кнопку для обновления данных
if st.button("Обновить данные"):
    df = load_data()
else:
    df = load_data()


# Функция для цветового выделения статуса
def highlight_status(s):
    if s == "Ошибка":
        return "background-color: red; color: white;"
    elif s == "В процессе":
        return "background-color: orange; color: white;"
    elif s == "Завершено":
        return "background-color: green; color: white;"
    else:
        return ""


# Применение стилей к DataFrame
df_style = df.style.map(highlight_status, subset=["Статус"])

# Отображение таблицы
st.write(df_style.to_html(), unsafe_allow_html=True)

# Отображение задач с ошибками
if df["С ошибками"].sum() > 0:
    st.header("Задачи с ошибками")
    for collection_name in collections_to_process:
        collection = db[collection_name]
        failed_tasks = list(collection.find({"status": "failed"}))
        if len(failed_tasks) > 0:
            st.subheader(f"Коллекция: {collection_name}")
            for i, task in enumerate(failed_tasks):
                task_id = task.get("_id")
                error_message = task.get("error", "Нет информации об ошибке")
                st.write(f"**ID задачи:** {task_id}")
                st.write(f"**Ошибка:** {error_message}")
                if i >= 2:
                    break  # Показываем не более 3 задач с ошибками в каждой коллекции
            st.write("---")
