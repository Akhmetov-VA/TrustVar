import datetime
import os
import uuid
from collections import Counter

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# Список моделей для выбора
MODELS = [
    # "gemma2:27b-instruct-q4_0",
    "gemma2:9b-instruct-q4_0",
    "ilyagusev/saiga_llama3",
    "llama2:13b",
    "llama3.1:8b-instruct_q4_0",
    # "llama3:70b-instruct-q4_0",
    "llama3:8b-instruct_q4_0",
    "mistral:7b-instruct-v0.3-q4_0",
    "mixtral:8x7b-instruct-v0.1-q4_0",
    "phi3:14b-medium-4k-instruct_q4_0",
    "qwen:7b",
    "qwen2:72b-instruct_q4_0",
    "qwen2.5:72b-instruct_q4_0",
    "qwen2:7b-instruct_q4_0",
    "solar:10.7b-instruct-v1-q4_0",
    "wavecut/vikhr:7b-instruct_0.4-Q4_1",
    "yi:6b",
    "yi:9b",
]


class MongoDBClient:
    """Класс для работы с MongoDB."""

    def __init__(self):
        # Получение данных для подключения из .env файла
        MONGO_USERNAME = os.getenv("MONGO_INITDB_ROOT_USERNAME")
        MONGO_PASSWORD = os.getenv("MONGO_INITDB_ROOT_PASSWORD")
        MONGO_HOST = os.getenv("MONGO_HOST")
        MONGO_PORT = os.getenv("MONGO_INITDB_ROOT_PORT")

        # Формирование URI для подключения к MongoDB
        mongo_uri = (
            f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/"
        )

        # Подключение к MongoDB
        self.client = MongoClient(mongo_uri)

        # Выбор базы данных
        self.db = self.client.TrustLLM_ru

    def get_collection(self, collection_name):
        return self.db[collection_name]

    def list_collections(self):
        return self.db.list_collection_names()

    def insert_data(self, collection_name, data):
        collection = self.get_collection(collection_name)
        if data:
            collection.insert_many(data)

    def update_tasks_status(self, collection_name, current_status, new_status):
        collection = self.get_collection(collection_name)
        result = collection.update_many(
            {"status": current_status}, {"$set": {"status": new_status}}
        )
        return result.modified_count

    def get_tasks_by_status(self, collection_name, status):
        collection = self.get_collection(collection_name)
        return list(collection.find({"status": status}))

    def count_tasks_by_status(self, collection_name, status):
        collection = self.get_collection(collection_name)
        return collection.count_documents({"status": status})

    def count_total_tasks(self, collection_name):
        collection = self.get_collection(collection_name)
        return collection.count_documents({})

    # Метод для удаления коллекции
    def delete_collection(self, collection_name):
        self.db.drop_collection(collection_name)


class ExperimentManager:
    """Класс для управления экспериментами."""

    def __init__(self, db_client):
        self.db_client = db_client
        self.prefix = "experiment_"
        self.unique_queries_collection = "unique_queries"

    def get_experiment_collections(self):
        return [
            col
            for col in self.db_client.list_collections()
            if col.startswith(self.prefix)
        ]

    def get_experiment_status(self, collection_name):
        total_tasks = self.db_client.count_total_tasks(collection_name)
        pending_tasks = self.db_client.count_tasks_by_status(collection_name, "pending")
        completed_tasks = self.db_client.count_tasks_by_status(
            collection_name, "completed"
        )
        failed_tasks = self.db_client.count_tasks_by_status(collection_name, "failed")

        # Определение статуса
        if failed_tasks > 0:
            status = "Ошибка"
        elif pending_tasks > 0:
            status = "В процессе"
        else:
            status = "Завершено"

        return {
            "total_tasks": total_tasks,
            "pending_tasks": pending_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "status": status,
        }

    def restart_failed_tasks(self, collection_name):
        return self.db_client.update_tasks_status(collection_name, "failed", "pending")

    def insert_experiment_data(self, collection_name, data_df, models):
        data_records = data_df.to_dict("records")
        job_id = str(uuid.uuid4())
        current_date = datetime.datetime.utcnow()
        # Установка начального статуса 'pending' и добавление модели
        records_to_insert = []
        for model in models:
            for record in data_records:
                new_record = record.copy()
                new_record["status"] = "pending"
                new_record["model"] = model
                new_record["job_id"] = job_id
                new_record["task_name"] = collection_name
                new_record["date"] = current_date
                # Обеспечиваем наличие поля 'variables'
                if "variables" not in new_record:
                    new_record["variables"] = {}
                # Обеспечиваем наличие поля 'prompt'
                if "prompt" not in new_record:
                    st.error("В записи отсутствует поле 'prompt'.")
                    return
                records_to_insert.append(new_record)
        self.db_client.insert_data(collection_name, records_to_insert)

    def insert_unique_queries(self, queries, models):
        job_id = str(uuid.uuid4())
        current_date = datetime.datetime.utcnow()
        records_to_insert = []
        for model in models:
            for query in queries:
                record = {
                    "prompt": query,
                    "variables": {},
                    "status": "pending",
                    "model": model,
                    "job_id": job_id,
                    "task_name": self.unique_queries_collection,
                    "date": current_date,
                }
                records_to_insert.append(record)
        self.db_client.insert_data(self.unique_queries_collection, records_to_insert)

    def get_unique_queries_status(self):
        return self.get_experiment_status(self.unique_queries_collection)

    def delete_unique_queries(self):
        self.db_client.delete_collection(self.unique_queries_collection)

    # Добавляем метод для удаления экспериментов
    def delete_experiment(self, collection_name):
        self.db_client.delete_collection(collection_name)


class Dashboard:
    """Класс для отображения дашборда в Streamlit."""

    def __init__(self):
        self.db_client = MongoDBClient()
        self.experiment_manager = ExperimentManager(self.db_client)
        st.set_page_config(page_title="Дашборд экспериментов", layout="wide")
        st.title("Дашборд состояния экспериментов")
        self.tabs = st.tabs(
            ["Дашборд", "Загрузка данных", "Единичные запросы", "Метрики"]
        )

    def run(self):
        self.show_dashboard_tab()
        self.show_experiments_tab()
        self.show_unique_queries_tab()
        self.show_metrics_tab()

    def show_dashboard_tab(self):
        with self.tabs[0]:
            collections_to_process = [
                col
                for col in self.db_client.list_collections()
                if col not in ["delete_me", "test", "results"]
            ]

            df = self.load_data(collections_to_process)

            # Добавляем кнопку для обновления данных
            if st.button("Обновить таблицу"):
                df = self.load_data(collections_to_process)

            # Применение стилей к DataFrame
            df = df.sort_values("Коллекция").reset_index(drop=True)
            df_style = df.style.applymap(self.highlight_status, subset=["Статус"])

            # Отображение таблицы
            st.write(df_style)

            # Отображение уникальных сообщений об ошибках
            if df["С ошибками"].sum() > 0:
                self.show_errors(collections_to_process)

            # Добавляем возможность отобразить данные выбранной коллекции
            st.header("Просмотр данных коллекции")
            if collections_to_process:
                selected_collection = st.selectbox(
                    "Выберите коллекцию", collections_to_process
                )

                if selected_collection:
                    collection = self.db_client.get_collection(selected_collection)
                    data = list(collection.find())
                    df_collection = pd.DataFrame(data)

                    # Удаление поля '_id'
                    if "_id" in df_collection.columns:
                        df_collection = df_collection.drop(columns=["_id"])

                    # Фильтрация по моделям
                    if "model" in df_collection.columns:
                        models_in_data = df_collection["model"].unique()
                        filter_models = st.multiselect(
                            "Фильтровать по моделям",
                            options=models_in_data,
                            default=models_in_data,
                            key=f"dashboard_filter_models_{selected_collection}",
                        )
                        df_filtered = df_collection[
                            df_collection["model"].isin(filter_models)
                        ]
                    else:
                        df_filtered = df_collection

                    st.dataframe(df_filtered)

                    # Скачивание результатов
                    csv = df_filtered.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Скачать результаты в CSV",
                        data=csv,
                        file_name=f"{selected_collection}_results.csv",
                        mime="text/csv",
                    )
            else:
                st.info("Нет доступных коллекций для просмотра.")

    def load_data(self, collections):
        data = []
        for collection_name in collections:
            total_tasks = self.db_client.count_total_tasks(collection_name)
            pending_tasks = self.db_client.count_tasks_by_status(
                collection_name, "pending"
            )
            completed_tasks = self.db_client.count_tasks_by_status(
                collection_name, "completed"
            )
            failed_tasks = self.db_client.count_tasks_by_status(
                collection_name, "failed"
            )

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

    @staticmethod
    def highlight_status(s):
        if s == "Ошибка":
            return "background-color: red; color: white;"
        elif s == "В процессе":
            return "background-color: orange; color: white;"
        elif s == "Завершено":
            return "background-color: green; color: white;"
        else:
            return ""

    def show_errors(self, collections):
        st.header("Уникальные сообщения об ошибках")
        with st.expander("Показать ошибки"):
            for collection_name in collections:
                failed_tasks = self.db_client.get_tasks_by_status(
                    collection_name, "failed"
                )
                if len(failed_tasks) > 0:
                    error_messages = [
                        task.get("error", "Нет информации об ошибке")
                        for task in failed_tasks
                    ]
                    error_counts = Counter(error_messages)
                    st.subheader(f"Коллекция: {collection_name}")
                    for error_message, count in error_counts.items():
                        st.write(
                            f"**Ошибка:** {error_message} | **Количество:** {count}"
                        )
                    st.write("---")

            if st.button("Перезапустить задачи с ошибками"):
                for collection_name in collections:
                    modified_count = self.db_client.update_tasks_status(
                        collection_name, "failed", "pending"
                    )
                    if modified_count > 0:
                        st.write(
                            f"В коллекции '{collection_name}' перезапущено {modified_count} задач."
                        )
            else:
                st.write(
                    "Нажмите кнопку выше, чтобы перезапустить все задачи с ошибками."
                )

    def show_experiments_tab(self):
        with self.tabs[1]:
            st.header("Загрузка данных для экспериментов")

            # Выбор моделей
            selected_models = st.multiselect(
                "Выберите модели для обработки",
                options=MODELS,
                default=[],
                key="experiment_model_selection",
            )

            if not selected_models:
                st.warning("Пожалуйста, выберите хотя бы одну модель.")
            else:
                # Загрузка файла
                uploaded_file = st.file_uploader(
                    "Загрузите CSV или Excel файл", type=["csv", "xlsx"]
                )

                if uploaded_file is not None:
                    self.handle_file_upload(uploaded_file, selected_models)
                else:
                    st.info("Пожалуйста, загрузите файл для обработки.")

            st.header("Результаты экспериментов")

            experiment_collections = (
                self.experiment_manager.get_experiment_collections()
            )

            if experiment_collections:
                self.show_experiment_results(experiment_collections)
            else:
                st.info("Нет загруженных экспериментов.")

    def handle_file_upload(self, uploaded_file, selected_models):
        try:
            # Чтение файла в DataFrame
            if uploaded_file.name.endswith(".csv"):
                data_df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                data_df = pd.read_excel(uploaded_file)
            else:
                st.error("Неподдерживаемый формат файла.")
                return

            # Проверка наличия колонки 'prompt'
            if "prompt" in data_df.columns:
                st.success("Файл успешно загружен.")
                st.dataframe(data_df.head())

                # Ввод названия для запуска
                run_name = st.text_input("Введите название для этого запуска")

                if run_name:
                    collection_name = self.experiment_manager.prefix + run_name

                    if st.button("Загрузить данные в MongoDB"):
                        self.experiment_manager.insert_experiment_data(
                            collection_name, data_df, selected_models
                        )
                        st.success(f"Данные загружены в коллекцию '{collection_name}'.")
                else:
                    st.warning("Пожалуйста, введите название для этого запуска.")
            else:
                st.error("В загруженном файле отсутствует колонка 'prompt'.")
        except Exception as e:
            st.error(f"Ошибка при обработке файла: {e}")

    def show_experiment_results(self, experiment_collections):
        # Выбор эксперимента
        selected_experiment = st.selectbox(
            "Выберите эксперимент", experiment_collections
        )

        # Получение статуса эксперимента
        status_info = self.experiment_manager.get_experiment_status(selected_experiment)

        st.write(f"**Всего задач:** {status_info['total_tasks']}")
        st.write(f"**Выполнено:** {status_info['completed_tasks']}")
        st.write(f"**В ожидании:** {status_info['pending_tasks']}")
        st.write(f"**С ошибками:** {status_info['failed_tasks']}")
        st.write(f"**Статус:** {status_info['status']}")

        with st.expander("Показать данные"):
            collection = self.db_client.get_collection(selected_experiment)
            data = list(collection.find())
            df = pd.DataFrame(data)

            # Удаление поля '_id'
            if "_id" in df.columns:
                df = df.drop(columns=["_id"])

            # Фильтрация по моделям
            models_in_data = df["model"].unique()
            filter_models = st.multiselect(
                "Фильтровать по моделям",
                options=models_in_data,
                default=models_in_data,
                key=f"experiment_filter_models_{selected_experiment}",
            )
            df_filtered = df[df["model"].isin(filter_models)]

            st.dataframe(df_filtered)

            # Скачивание результатов
            csv = df_filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Скачать результаты в CSV",
                data=csv,
                file_name=f"{selected_experiment}_results.csv",
                mime="text/csv",
            )

            # Добавляем кнопку для удаления эксперимента
            if st.button("Удалить эксперимент"):
                self.experiment_manager.delete_experiment(selected_experiment)
                st.success(f"Эксперимент '{selected_experiment}' удален.")
                # Обновляем страницу без использования st.experimental_rerun()
                st.experimental_set_query_params()
                st.stop()

        # Возможность перезапустить задачи с ошибками
        if status_info["failed_tasks"] > 0:
            if st.button(
                "Перезапустить задачи с ошибками",
                key=f"restart_experiment_{selected_experiment}",
            ):
                modified_count = self.experiment_manager.restart_failed_tasks(
                    selected_experiment
                )
                st.success(f"Перезапущено {modified_count} задач с ошибками.")

    def show_unique_queries_tab(self):
        with self.tabs[2]:
            st.header("Обработка единичных текстовых запросов")

            # Выбор моделей
            selected_models = st.multiselect(
                "Выберите модели для обработки",
                options=MODELS,
                default=[],
                key="unique_queries_model_selection",
            )

            if not selected_models:
                st.warning("Пожалуйста, выберите хотя бы одну модель.")
            else:
                # Поле для ввода текстовых запросов
                queries_input = st.text_area(
                    "Введите текстовые запросы (по одному на строку)"
                )

                if queries_input:
                    queries = [
                        q.strip() for q in queries_input.split("\n") if q.strip()
                    ]

                    if st.button("Отправить запросы на обработку"):
                        self.experiment_manager.insert_unique_queries(
                            queries, selected_models
                        )
                        st.success("Запросы отправлены на обработку.")
                else:
                    st.info("Пожалуйста, введите текстовые запросы для обработки.")

            # Отображение результатов
            st.header("Результаты единичных запросов")

            status_info = self.experiment_manager.get_unique_queries_status()

            if status_info["total_tasks"] > 0:
                st.write(f"**Всего задач:** {status_info['total_tasks']}")
                st.write(f"**Выполнено:** {status_info['completed_tasks']}")
                st.write(f"**В ожидании:** {status_info['pending_tasks']}")
                st.write(f"**С ошибками:** {status_info['failed_tasks']}")
                st.write(f"**Статус:** {status_info['status']}")

                with st.expander("Показать данные"):
                    collection = self.db_client.get_collection(
                        self.experiment_manager.unique_queries_collection
                    )
                    data = list(collection.find())
                    df = pd.DataFrame(data)

                    # Удаление поля '_id'
                    if "_id" in df.columns:
                        df = df.drop(columns=["_id"])

                    # Фильтрация по моделям
                    models_in_data = df["model"].unique()
                    filter_models = st.multiselect(
                        "Фильтровать по моделям",
                        options=models_in_data,
                        default=models_in_data,
                        key="unique_queries_filter_models",
                    )
                    df_filtered = df[df["model"].isin(filter_models)]

                    st.dataframe(df_filtered)

                    # Скачивание результатов
                    csv = df_filtered.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Скачать результаты в CSV",
                        data=csv,
                        file_name="unique_queries_results.csv",
                        mime="text/csv",
                    )

                    # Добавляем кнопку для удаления единичных экспериментов
                    if st.button("Удалить единичные эксперименты"):
                        self.experiment_manager.delete_unique_queries()
                        st.success("Единичные эксперименты удалены.")
                        # Обновляем страницу без использования st.experimental_rerun()
                        st.experimental_set_query_params()
                        st.stop()

                # Возможность перезапустить задачи с ошибками
                if status_info["failed_tasks"] > 0:
                    if st.button(
                        "Перезапустить задачи с ошибками",
                        key="restart_unique_queries",
                    ):
                        modified_count = self.experiment_manager.restart_failed_tasks(
                            self.experiment_manager.unique_queries_collection
                        )
                        st.success(f"Перезапущено {modified_count} задач с ошибками.")
            else:
                st.info("Нет обработанных единичных запросов.")

    def show_metrics_tab(self):
        with self.tabs[3]:
            st.header("Метрики моделей")

            # Визуализация метрик по таблице 'results'
            results_collection = self.db_client.get_collection("results")
            results_data = list(results_collection.find())

            if results_data:
                self.visualize_metrics(results_data)
            else:
                st.info("Данные в коллекции 'results' отсутствуют.")

            # Отображение CSV-файла с самыми сложными вопросами для jailbreak
            self.show_jailbreak_questions()

    def visualize_metrics(self, results_data):
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
        selected_models = st.multiselect(
            "Выберите модели", options=models, default=models
        )

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

    def show_jailbreak_questions(self):
        st.subheader("Самые сложные вопросы для моделей")

        # Указываем путь к CSV-файлу
        csv_file_path = "/home/vadim/work/TrustLLM_ru/data/new_data/top20_jailbreak.csv"

        try:
            # Загружаем данные из CSV-файла
            jailbreak_df = pd.read_csv(csv_file_path, index_col=0)

            # Отображаем данные
            st.dataframe(jailbreak_df)
        except Exception as e:
            st.error(f"Ошибка при загрузке CSV-файла: {e}")


if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.run()
