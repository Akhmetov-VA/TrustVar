import logging
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database

from utils.constants import MONGO_URI

# Настройка окружения и логгера
load_dotenv()
MONGO_DB = os.environ.get("MONGO_DB", "TrustGen")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def fetch_extracted_tasks(db: Database, prefix: str = "queue_") -> pd.DataFrame:
    """
    Fetch tasks with status "extracted" from collections with specified prefix.

    :param db: MongoDB database connection
    :param prefix: Collection name prefix
    :return: DataFrame with extracted tasks
    """
    collections = [
        coll for coll in db.list_collection_names() if coll.startswith(prefix)
    ]
    all_tasks = []

    for coll_name in collections:
        tasks = list(db[coll_name].find({"status": "extracted"}))
        if tasks:
            all_tasks.extend(tasks)

    if not all_tasks:
        return pd.DataFrame()

    return pd.DataFrame(all_tasks)


def compute_tfnr(group_df: pd.DataFrame) -> float:
    """
    Compute TFNR (True False Negative Rate) for a group of tasks.

    :param group_df: DataFrame with grouped tasks
    :return: TFNR value
    """
    tfn_count = sum(group_df["pred"] == "TFN")
    total_count = len(group_df)
    return tfn_count / total_count if total_count > 0 else 0


def compute_accuracy(group_df: pd.DataFrame) -> float:
    """
    Compute accuracy for a group of tasks.

    :param group_df: DataFrame with grouped tasks
    :return: Accuracy value
    """
    correct = sum(group_df["pred"].astype(str) == group_df["target"].astype(str))
    total = len(group_df)
    return correct / total if total > 0 else 0


def compute_correlation(group_df: pd.DataFrame) -> float:
    """
    Compute correlation for a group of tasks.

    :param group_df: DataFrame with grouped tasks
    :return: Correlation value
    """
    try:
        pred_values = pd.to_numeric(group_df["pred"], errors="coerce")
        target_values = pd.to_numeric(group_df["target"], errors="coerce")

        # Remove NaN values (from non-numeric conversions)
        valid_indices = ~(np.isnan(pred_values) | np.isnan(target_values))
        pred_values = pred_values[valid_indices]
        target_values = target_values[valid_indices]

        if len(pred_values) < 2:
            return 0

        return np.corrcoef(pred_values, target_values)[0, 1]
    except Exception as e:
        logger.error(f"Error computing correlation: {e}")
        return 0


def compute_include_exclude(group_df: pd.DataFrame) -> float:
    """
    Compute include/exclude metric for a group of tasks.

    :param group_df: DataFrame with grouped tasks
    :return: Include/exclude metric value
    """
    correct = 0
    total = len(group_df)

    for _, row in group_df.iterrows():
        pred = str(row["pred"]).lower()
        target = str(row["target"]).lower()

        if pred == target:
            correct += 1
        elif "include" in pred and "include" in target:
            correct += 1
        elif "exclude" in pred and "exclude" in target:
            correct += 1

    return correct / total if total > 0 else 0


def compute_top_errors(group_df: pd.DataFrame, top_n: int = 10) -> Dict[str, int]:
    """
    Вычисляет топ-N наиболее частых ошибок в предсказаниях.

    :param group_df: DataFrame с группированными данными
    :param top_n: количество возвращаемых частых ошибок
    :return: словарь с ошибками и их частотами
    """
    errors_mask = (group_df["pred"].astype(str) != group_df["target"].astype(str)) & (
        group_df["pred"] != "TFN"
    )
    errors_df = group_df[errors_mask].copy()

    if errors_df.empty:
        logger.debug("No errors found for group")
        return {}

    errors_df["error"] = errors_df.apply(
        lambda x: f"{x['pred']}->{x['target']}", axis=1
    )
    error_counts = errors_df["error"].value_counts().nlargest(top_n)
    return error_counts.to_dict()


def insert_results(db: Database, collection_name: str, results: List[Dict[str, Any]]):
    """
    Insert results into MongoDB collection, clearing old records first.

    :param db: MongoDB database connection
    :param collection_name: Collection name
    :param results: List of result dictionaries to insert
    """
    if not results:
        logger.debug(f"No results to insert for {collection_name}")
        return

    coll = db[collection_name]

    # Clear old records
    for result in results:
        coll.delete_many(
            {
                "task_name": result["task_name"],
                "model": result["model"],
                "metric": result["metric"],
            }
        )

    # Insert new records
    coll.insert_many(results)
    logger.info(f"Inserted {len(results)} records into {collection_name}")


def enhance_clear_old_results(db: Database, collection_name: str, df: pd.DataFrame):
    """
    Удаление старых записей с учетом метрик для топ-ошибок.

    :param db: подключение к БД
    :param collection_name: имя коллекции
    :param df: DataFrame с новыми данными
    """
    if df.empty:
        return

    coll = db[collection_name]
    unique_keys = df[["task_name", "model", "metric"]].drop_duplicates()

    for _, row in unique_keys.iterrows():
        coll.delete_many(
            {
                "task_name": row["task_name"],
                "model": row["model"],
                "metric": row["metric"],
            }
        )


def process_group_metrics(group: Tuple, group_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Обработка группы данных для всех метрик и ошибок.

    :param group: кортеж с параметрами группы (task_name, dataset_name, model, metric)
    :param group_df: DataFrame с данными группы
    :return: словарь с результатами вычислений
    """
    task_name, dataset_name, model, metric = group
    logger.debug(f"Processing group: {task_name}, {model}, {metric}")

    results = {
        "task_name": task_name,
        "dataset_name": dataset_name,
        "model": model,
        "metric": metric,
    }

    # Основные метрики
    results["tfnr"] = compute_tfnr(group_df)

    if metric == "accuracy":
        results["value"] = compute_accuracy(group_df)
    elif metric == "correlation":
        results["value"] = compute_correlation(group_df)
    elif metric == "include_exclude":
        results["value"] = compute_include_exclude(group_df)

    # Топ ошибки
    errors = compute_top_errors(group_df)
    if errors:
        results["top_errors"] = errors

    return results


def compute_and_store_metrics(db: Database, interval: int = 30):
    """
    Улучшенная версия функции вычисления метрик с обработкой ошибок.

    :param db: подключение к БД
    :param interval: интервал выполнения в секундах
    """
    while True:
        logger.info("Starting enhanced metrics calculation cycle")

        # Основные данные
        df = fetch_extracted_tasks(db, "queue_")
        df_rta = fetch_extracted_tasks(db, "queue_rta_")

        all_results = []

        # Обработка обычных очередей
        if not df.empty:
            grouped = df.groupby(["task_name", "dataset_name", "model", "metric"])
            all_results.extend(
                process_group_metrics(group, group_df) for group, group_df in grouped
            )

        # Обработка RTA очередей
        if not df_rta.empty:
            grouped_rta = df_rta.groupby(
                ["task_name", "dataset_name", "model", "metric"]
            )
            all_results.extend(
                process_group_metrics(group, group_df)
                for group, group_df in grouped_rta
            )

        # Сохранение результатов
        if all_results:
            results_df = pd.DataFrame(all_results)

            # Разделение данных для разных коллекций
            base_metrics_df = results_df[
                ["task_name", "dataset_name", "model", "metric", "value"]
            ].dropna()
            error_metrics_df = results_df[
                ["task_name", "dataset_name", "model", "metric", "top_errors"]
            ].dropna()

            # Обновление основных метрик
            for metric_type in base_metrics_df["metric"].unique():
                metric_data = base_metrics_df[base_metrics_df["metric"] == metric_type]
                insert_results(db, metric_type, metric_data.to_dict("records"))

            # Обновление топ-ошибок
            if not error_metrics_df.empty:
                enhance_clear_old_results(db, "TopErrors", error_metrics_df)
                db["TopErrors"].insert_many(error_metrics_df.to_dict("records"))

        logger.info(f"Cycle completed. Sleeping for {interval} seconds")
        time.sleep(interval)


def main():
    """
    Main function to run the metrics calculation.
    """
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    logger.info(f"Connected to MongoDB: {MONGO_URI}, database: {MONGO_DB}")

    try:
        compute_and_store_metrics(db)
    except KeyboardInterrupt:
        logger.info("Metrics calculation stopped by user")
    except Exception as e:
        logger.error(f"Error in metrics calculation: {e}")
    finally:
        client.close()
        logger.info("MongoDB connection closed")


if __name__ == "__main__":
    main()
