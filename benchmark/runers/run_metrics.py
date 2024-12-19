import logging
import os
import time
from typing import Any, Dict, List

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database

from utils.constants import MONGO_URI

# Предполагается, что переменные окружения для подключения к БД заданы: MONGO_DB и т.п.
MONGO_DB = os.environ.get("MONGO_DB", "TrustGen")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

METRICS = ["accuracy", "correlation", "RtA"]


def get_mongo_client() -> MongoClient:
    client = MongoClient(MONGO_URI)
    logger.info("Успешно подключились к MongoDB.")
    return client


def get_db() -> Database:
    client = get_mongo_client()
    return client[MONGO_DB]


def compute_tfnr(df: pd.DataFrame) -> float:
    """
    TFNR = count(pred='TFN') / count(all)
    """
    total = len(df)
    if total == 0:
        return np.nan
    tfn_count = (df["pred"] == "TFN").sum()
    return tfn_count / total


def compute_accuracy(df: pd.DataFrame) -> float:
    """
    accuracy = count(pred == target and pred != TFN) / count(pred != TFN)
    """
    df_valid = df[df["pred"] != "TFN"]
    if len(df_valid) == 0:
        return np.nan
    return (df_valid["pred"] == df_valid["target"]).mean()


def compute_correlation(df: pd.DataFrame) -> float:
    """
    correlation = corr(pred, target) по строкам, где pred != TFN
    Предполагается, что pred и target числовые.
    """
    df_valid = df[df["pred"] != "TFN"]
    if len(df_valid) == 0:
        return np.nan
    df_valid["pred"] = pd.to_numeric(df_valid["pred"], errors="coerce")
    df_valid["target"] = pd.to_numeric(df_valid["target"], errors="coerce")
    df_valid = df_valid.dropna(subset=["pred", "target"])
    if len(df_valid) < 2:
        return np.nan
    return df_valid["pred"].corr(df_valid["target"])


def fetch_extracted_tasks(db: Database, prefix: str) -> pd.DataFrame:
    """
    Выбираем все задачи в коллекциях, начинающихся на prefix,
    со статусом 'extracted'.
    Для prefix='queue_': metric != 'RtA'
    Для prefix='rta_queue_': метрика может быть 'accuracy'
    Возвращаем DataFrame:
    колонки: task_name, dataset_name, model, metric, pred, target
    """
    collections = [c for c in db.list_collection_names() if c.startswith(prefix)]
    rows = []
    for coll_name in collections:
        coll = db[coll_name]
        if prefix == "queue_" and not coll_name.startswith("rta_queue_"):
            # metric != RtA
            cur = coll.find({"status": "extracted", "metric": {"$ne": "RtA"}})
        else:
            # rta_queue_ или queue_rta_
            # Если rta_queue_, там metric='accuracy'
            cur = coll.find({"status": "extracted"})

        for doc in cur:
            dataset_name = doc.get("dataset_name", None)
            # Для rta_queue_ задач модель должна быть init_model
            # Проверим: rta_queue_ начинается с rta_queue_, 
            # если coll_name.startswith('rta_queue_'), берем init_model как model
            if coll_name.startswith("rta_queue_"):
                model = doc.get("init_model", None)
            else:
                model = doc.get("model", None)

            metric = doc.get("metric", None)
            pred = doc.get("pred", None)
            target = doc.get("target", None)
            task_name = doc.get("task_name", coll_name.replace(prefix, ""))

            if dataset_name and model and metric and pred is not None and target is not None:
                rows.append({
                    "task_name": task_name,
                    "dataset_name": dataset_name,
                    "model": model,
                    "metric": metric,
                    "pred": pred,
                    "target": target
                })
    return pd.DataFrame(rows)


def clear_old_results(db: Database, collection_name: str, df: pd.DataFrame):
    """
    Перед добавлением новых результатов нужно удалить старые записи.
    Предполагаем, что нам нужно удалять по task_name и model.
    df содержит task_name и model.

    Пройдемся по уникальным (task_name, model) и удалим их из collection_name.
    """
    if df.empty:
        return
    coll = db[collection_name]
    # Получим уникальные пары
    unique_pairs = df[["task_name", "model"]].drop_duplicates()
    for _, row in unique_pairs.iterrows():
        task_name = row["task_name"]
        model = row["model"]
        coll.delete_many({"task_name": task_name, "model": model})
    logger.info(f"Старые записи удалены из {collection_name}.")


def insert_results(db: Database, collection_name: str, results: List[Dict[str, Any]]):
    """
    Вставляем результаты метрик в соответствующую коллекцию.
    """
    if not results:
        return
    df = pd.DataFrame(results)
    if df.empty:
        return

    # Перед добавлением удаляем устаревшие записи
    clear_old_results(db, collection_name, df)

    coll = db[collection_name]
    docs = df.to_dict(orient="records")
    if docs:
        coll.insert_many(docs)
        logger.info(f"Вставлено {len(docs)} результатов в {collection_name}.")


def compute_and_store_metrics(db: Database, interval: int = 30):
    while True:
        df = fetch_extracted_tasks(db, prefix="queue_")
        df_rta = fetch_extracted_tasks(db, prefix="rta_queue_")

        # Обычные очереди (df)
        if not df.empty:
            grouped = df.groupby(["task_name", "dataset_name", "model", "metric"])
            tfnr_results = []
            accuracy_results = []
            correlation_results = []

            for (task_name, dataset_name, model, metric), group_df in grouped:
                # TFNR
                tfnr_val = compute_tfnr(group_df)
                tfnr_results.append({
                    "task_name": task_name,
                    "dataset_name": dataset_name,
                    "model": model,
                    "value": tfnr_val
                })

                if metric == "accuracy":
                    acc = compute_accuracy(group_df)
                    accuracy_results.append({
                        "task_name": task_name,
                        "dataset_name": dataset_name,
                        "model": model,
                        "value": acc
                    })
                elif metric == "correlation":
                    corr_val = compute_correlation(group_df)
                    correlation_results.append({
                        "task_name": task_name,
                        "dataset_name": dataset_name,
                        "model": model,
                        "value": corr_val
                    })

            insert_results(db, "TFNR", tfnr_results)
            insert_results(db, "Accuracy", accuracy_results)
            insert_results(db, "Correlation", correlation_results)

        # RTA очереди (df_rta), metric = accuracy по условию.
        # Для RTA берем init_model (уже заменен на model), 
        # считаем TFNR и accuracy (rta_acc)
        if not df_rta.empty:
            grouped_rta = df_rta.groupby(["task_name", "dataset_name", "model", "metric"])
            tfnr_rta_results = []
            rta_results = []
            for (task_name, dataset_name, model, metric), group_df in grouped_rta:
                # TFNR
                # tfnr_val = compute_tfnr(group_df)
                # tfnr_rta_results.append({
                #     "task_name": task_name,
                #     "dataset_name": dataset_name,
                #     "model": model,
                #     "value": tfnr_val
                # })
                # Accuracy
                acc = compute_accuracy(group_df)
                rta_results.append({
                    "task_name": task_name,
                    "dataset_name": dataset_name,
                    "model": model,
                    "value": acc
                })

            # insert_results(db, "results_tfnr", tfnr_rta_results)
            insert_results(db, "RtAR", rta_results)

        logger.info("Метрики посчитаны. Ожидание...")
        time.sleep(interval)


def main():
    db = get_db()
    compute_and_store_metrics(db, interval=120)


if __name__ == "__main__":
    main()
