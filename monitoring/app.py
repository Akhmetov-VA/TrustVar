import os
from collections import Counter

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# Получение данных для подключения из .env файла
MONGO_USERNAME = os.getenv("MONGO_INITDB_ROOT_USERNAME")
MONGO_PASSWORD = os.getenv("MONGO_INITDB_ROOT_PASSWORD")
MONGO_HOST = os.getenv("MONGO_HOST")
MONGO_PORT = os.getenv("MONGO_INITDB_ROOT_PORT")


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


# Формирование URI для подключения к MongoDB
mongo_uri = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"

# Подключение к MongoDB
client = MongoClient(mongo_uri)

# Выбор базы данных
db = client.TrustLLM_ru

st.title("Дашборд состояния экспериментов")

### Добавляем визуализацию по таблице 'results' ###

# Получаем данные из коллекции 'results'
results_collection = db["results"]
results_data = list(results_collection.find())

if results_data:
    # Преобразуем данные в DataFrame
    results_df = pd.DataFrame(results_data)

    # Удаляем служебное поле '_id' если оно есть
    if "_id" in results_df.columns:
        results_df = results_df.drop(columns=["_id"])

    # Определяем доступные значения для фильтрации
    datasets = results_df["dataset"].unique()
    models = results_df["model"].unique()

    # Добавляем виджеты для фильтрации
    selected_datasets = st.multiselect(
        "Выберите датасеты", options=datasets, default=datasets
    )
    selected_models = st.multiselect("Выберите модели", options=models, default=models)

    # Применяем фильтры к данным
    filtered_df = results_df[
        (results_df["dataset"].isin(selected_datasets))
        & (results_df["model"].isin(selected_models))
    ]

    # Группируем данные и вычисляем среднее значение 'value' для каждой пары 'dataset'-'model'
    pivot_table = filtered_df.pivot_table(
        index="model", columns="dataset", values="value", aggfunc="mean"
    )

    st.subheader("Таблица метрик по датасетам и моделям")
    st.dataframe(pivot_table)

    # Визуализация данных
    st.subheader("Визуализация метрик")
    st.bar_chart(pivot_table)
else:
    st.info("Данные в коллекции 'results' отсутствуют.")

# Получение списка коллекций для обработки, исключая определенные
collections_to_process = [
    col
    for col in db.list_collection_names()
    if col not in ["delete_me", "test", "results"]
]


# Добавляем кнопку для обновления данных
if st.button("Обновить таблицу"):
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
df_style = df.style.applymap(highlight_status, subset=["Статус"])

# Отображение таблицы
st.write(df_style)

# Отображение уникальных сообщений об ошибках в раскрывающейся вкладке
if df["С ошибками"].sum() > 0:
    st.header("Уникальные сообщения об ошибках")
    with st.expander("Показать ошибки"):
        for collection_name in collections_to_process:
            collection = db[collection_name]
            failed_tasks = list(collection.find({"status": "failed"}))
            if len(failed_tasks) > 0:
                error_messages = [
                    task.get("error", "Нет информации об ошибке")
                    for task in failed_tasks
                ]
                error_counts = Counter(error_messages)
                st.subheader(f"Коллекция: {collection_name}")
                for error_message, count in error_counts.items():
                    st.write(f"**Ошибка:** {error_message} | **Количество:** {count}")
                st.write("---")

        if st.button("Перезапустить задачи с ошибками"):
            for collection_name in collections_to_process:
                collection = db[collection_name]
                # Обновляем статус задач с ошибками на 'pending'
                result = collection.update_many(
                    {"status": "failed"}, {"$set": {"status": "pending"}}
                )
                if result.modified_count > 0:
                    st.write(
                        f"В коллекции '{collection_name}' перезапущено {result.modified_count} задач."
                    )
        else:
            st.write("Нажмите кнопку выше, чтобы перезапустить все задачи с ошибками.")


### Отображение CSV-файла с самыми сложными вопросами для jailbreak ###

st.header("Самые сложные вопросы для моделей")

# Указываем путь к CSV-файлу
csv_file_path = "/home/vadim/work/TrustLLM_ru/data/new_data/top20_jailbreak.csv"

try:
    # Загружаем данные из CSV-файла
    jailbreak_df = pd.read_csv(csv_file_path, index_col=0)

    # Отображаем данные в раскрывающемся элементе
    with st.expander("Показать вопросы для Jailbreak"):
        st.dataframe(jailbreak_df)
except Exception as e:
    st.error(f"Ошибка при загрузке CSV-файла: {e}")
