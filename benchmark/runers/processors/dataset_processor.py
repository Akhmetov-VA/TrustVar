import logging
from abc import ABC, abstractmethod

import pandas as pd
from pymongo import UpdateOne


class ProcessorMeta(type):
    registry = []

    def __init__(cls, name, bases, dct):
        if name != "DatasetProcessor" and issubclass(cls, DatasetProcessor):
            ProcessorMeta.registry.append(cls)
        super().__init__(name, bases, dct)


class DatasetProcessor(ABC, metaclass=ProcessorMeta):
    """
    Абстрактный базовый класс для обработки датасетов.
    """

    COLLECTION_NAME = ""

    def __init__(self, db_client, collection_results, collection_top_questions):
        self.db_client = db_client
        self.collection_results = db_client.get_collection(collection_results)
        self.collection_top_questions = db_client.get_collection(
            collection_top_questions
        )

    def process_dataset(self):
        """
        Шаблонный метод, описывающий общий процесс обработки датасета.
        """
        df, collection = self.load_tasks({"status": "completed"})
        if df is None:
            return

        df = self.extract_results(df)
        self.update_tasks_in_collection(df, collection, ["res"])
        self.generate_top_questions(
            df, self.COLLECTION_NAME, ["prompt"], "res == 0", 20
        )
        metrics = self.calculate_metrics(df)
        self.save_metrics(metrics)

    def load_tasks(self, query_filter):
        """
        Загружает задачи из коллекции с заданным фильтром.
        """
        collection = self.db_client.get_collection(self.COLLECTION_NAME)
        tasks_list = list(collection.find(query_filter))
        if not tasks_list:
            logging.info(f"Нет завершенных задач в коллекции '{self.COLLECTION_NAME}'.")
            return None, None
        df = pd.DataFrame(tasks_list)
        if df.empty:
            logging.info(
                f"Данных для обработки в коллекции '{self.COLLECTION_NAME}' нет."
            )
            return None, None
        return df, collection

    def update_tasks_in_collection(self, df, collection, fields_to_update):
        """
        Обновляет документы в коллекции, устанавливая указанные поля и статус 'measured'.
        """
        operations = [
            UpdateOne(
                {"_id": row["_id"]},
                {
                    "$set": {
                        **{field: row[field] for field in fields_to_update},
                        "status": "measured",
                    }
                },
            )
            for _, row in df.iterrows()
        ]
        if operations:
            collection.bulk_write(operations)
            logging.info(
                f"Обновлено {len(operations)} документов в коллекции '{collection.name}'."
            )

    def generate_top_questions(
        self, df, dataset_name, group_by_columns, filter_condition=None, top_n=20
    ):
        """
        Генерирует топ вопросов и сохраняет их в коллекцию 'top_questions'.
        """
        filtered_df = df.query(filter_condition) if filter_condition else df
        if filtered_df.empty:
            logging.info(
                f"Нет данных для генерации топ вопросов в датасете '{dataset_name}'."
            )
            return

        top_questions = (
            filtered_df.groupby(group_by_columns)
            .size()
            .reset_index(name="count")
            .sort_values(by="count", ascending=False)
            .head(top_n)
        )

        self.collection_top_questions.update_one(
            {"dataset": dataset_name},
            {"$set": {"top_questions": top_questions.to_dict("records")}},
            upsert=True,
        )
        logging.info(f"Топ {top_n} вопросов для датасета '{dataset_name}' сохранены.")

    def save_metrics(self, metrics):
        """
        Сохраняет метрики в коллекцию 'results'.
        """
        if not metrics:
            logging.info("Нет новых метрик для сохранения.")
            return
        operations = [
            UpdateOne(
                {"dataset": record["dataset"], "model": record["model"]},
                {"$set": record},
                upsert=True,
            )
            for record in metrics
        ]
        self.collection_results.bulk_write(operations)
        logging.info(f"Метрики для датасета '{metrics[0]['dataset']}' сохранены.")

    def revert_status(self):
        """
        Отменяет статус 'measured' на 'completed' для всех коллекций, кроме служебных.
        """
        query = {"status": "measured"}
        for collection_name in self.db_client.db.list_collection_names():
            if collection_name in ["delete_me", "test", "results", "top_questions"]:
                continue
            collection = self.db_client.get_collection(collection_name)
            result = collection.update_many(query, {"$set": {"status": "completed"}})
            logging.info(
                f"Обновлено {result.modified_count} документов в коллекции '{collection_name}'."
            )

    @abstractmethod
    def extract_results(self, df):
        """
        Абстрактный метод для извлечения результатов. Должен быть реализован в подклассе.
        """
        pass

    @abstractmethod
    def calculate_metrics(self, df):
        """
        Абстрактный метод для расчета метрик. Должен быть реализован в подклассе.
        """
        pass
