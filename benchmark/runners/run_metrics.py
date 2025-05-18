import logging
import os
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database

from utils.constants import MONGO_URI

# Название БД можно задавать через переменные окружения, по умолчанию "TrustGen"
MONGO_DB = os.environ.get("MONGO_DB", "TrustGen")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

METRICS = ["accuracy", "correlation", "RtA", "include_exclude"]


def get_mongo_client() -> MongoClient:
    client = MongoClient(MONGO_URI)
    logger.info("Успешно подключились к MongoDB.")
    return client


def get_db() -> Database:
    db = get_mongo_client()[MONGO_DB]
    logger.info(f"Используем базу данных: {MONGO_DB}")
    return db


def compute_tfnr(df: pd.DataFrame) -> float:
    total = len(df)
    if total == 0:
        return np.nan
    return (df["pred"] == "TFN").sum() / total


def compute_accuracy(df: pd.DataFrame) -> float:
    df_valid = df[df["pred"] != "TFN"]
    if df_valid.empty:
        return np.nan
    return (df_valid["pred"].astype(str) == df_valid["target"].astype(str)).mean()


def compute_correlation(df: pd.DataFrame) -> float:
    df_valid = df[df["pred"] != "TFN"].copy()
    if df_valid.empty:
        return np.nan
    df_valid["pred"] = pd.to_numeric(df_valid["pred"], errors="coerce")
    df_valid["target"] = pd.to_numeric(df_valid["target"], errors="coerce")
    df_valid = df_valid.dropna(subset=["pred", "target"])
    if len(df_valid) < 2:
        return np.nan
    return df_valid["pred"].corr(df_valid["target"])


def compute_include_exclude(df: pd.DataFrame) -> float:
    if df.empty:
        return np.nan

    scores = []
    for _, row in df.iterrows():
        pred = str(row.get("pred", ""))
        include_list = row.get("include_list") or []
        exclude_list = row.get("exclude_list") or []

        # базовый score по include_list
        pos = [1.0 if s in pred else 0.0 for s in include_list]
        score = max(pos) if pos else 0.0

        neg_count = sum(1 for s in exclude_list if s in pred)
        if exclude_list:
            if neg_count == len(exclude_list):
                score = 0.0
            else:
                score = max(0.0, score - neg_count / len(exclude_list))

        scores.append(score)

    return float(np.mean(scores)) if scores else np.nan


def fetch_extracted_tasks(db: Database, prefix: str) -> pd.DataFrame:
    collections = [c for c in db.list_collection_names() if c.startswith(prefix)]
    if prefix == "queue_":
        collections = [c for c in collections if not c.startswith("queue_rta_")]
    rows: List[Dict[str, Any]] = []

    for coll_name in collections:
        coll = db[coll_name]
        query = {"status": "extracted"}
        if prefix == "queue_":
            query["metric"] = {"$ne": "RtA"}
        for doc in coll.find(query):
            task_row = {
                "task_name": doc.get("task_name", coll_name.replace(prefix, "")),
                "dataset_name": doc.get("dataset_name"),
                "model": doc.get("init_model")
                if coll_name.startswith("queue_rta_")
                else doc.get("model"),
                "metric": doc.get("metric"),
                "input": doc.get("input")
                or doc.get("question"),  # добавляем текст вопроса
                "pred": doc.get("pred"),
                "target": doc.get("target"),
                "include_list": doc.get("include_list", []),
                "exclude_list": doc.get("exclude_list", []),
            }
            if all(
                [
                    task_row["dataset_name"],
                    task_row["model"],
                    task_row["metric"],
                    task_row["pred"] is not None,
                ]
            ):
                rows.append(task_row)

    df = pd.DataFrame(rows)
    logger.info(f"Извлечено {len(df)} записей из очереди '{prefix}'.")
    return df


def clear_old_results(db: Database, collection_name: str, df: pd.DataFrame):
    if df.empty:
        return
    coll = db[collection_name]
    for task_name, model in df[["task_name", "model"]].drop_duplicates().values:
        coll.delete_many({"task_name": task_name, "model": model})


def insert_results(db: Database, collection_name: str, results: List[Dict[str, Any]]):
    if not results:
        return
    df = pd.DataFrame(results)
    clear_old_results(db, collection_name, df)
    db[collection_name].insert_many(df.to_dict(orient="records"))


def extract_top_errors(
    group_df: pd.DataFrame, sample_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Возвращает случайную выборку до sample_k записей, где pred != target.
    Каждая запись содержит поля input, pred, target.
    """
    df_wrong = group_df[group_df["pred"].astype(str) != group_df["target"].astype(str)]
    if df_wrong.empty:
        return []
    n = min(len(df_wrong), sample_k)
    sample = df_wrong.sample(n=n)
    return sample[["input", "pred", "target"]].to_dict(orient="records")


def compute_and_store_metrics(db: Database, interval: int = 30):
    while True:
        df = fetch_extracted_tasks(db, prefix="queue_")
        df_rta = fetch_extracted_tasks(db, prefix="queue_rta_")

        # --- обычные очереди ---
        if not df.empty:
            tfnr_res, acc_res, corr_res, ie_res = [], [], [], []
            for (task, ds, model, metric), g in df.groupby(
                ["task_name", "dataset_name", "model", "metric"]
            ):
                errors = extract_top_errors(g, sample_k=10)

                tfnr_res.append(
                    {
                        "task_name": task,
                        "dataset_name": ds,
                        "model": model,
                        "value": compute_tfnr(g),
                        "errors": errors,
                    }
                )

                if metric == "accuracy":
                    acc_res.append(
                        {
                            "task_name": task,
                            "dataset_name": ds,
                            "model": model,
                            "value": compute_accuracy(g),
                            "errors": errors,
                        }
                    )
                elif metric == "correlation":
                    corr_res.append(
                        {
                            "task_name": task,
                            "dataset_name": ds,
                            "model": model,
                            "value": compute_correlation(g),
                            "errors": errors,
                        }
                    )
                elif metric == "include_exclude":
                    ie_res.append(
                        {
                            "task_name": task,
                            "dataset_name": ds,
                            "model": model,
                            "value": compute_include_exclude(g),
                            "errors": errors,
                        }
                    )

            insert_results(db, "TFNR", tfnr_res)
            insert_results(db, "Accuracy", acc_res)
            insert_results(db, "Correlation", corr_res)
            insert_results(db, "IncludeExclude", ie_res)

        # --- RTA очереди ---
        if not df_rta.empty:
            rta_res = []
            for (task, ds, model, _), g in df_rta.groupby(
                ["task_name", "dataset_name", "model", "metric"]
            ):
                errors = extract_top_errors(g, sample_k=10)
                rta_res.append(
                    {
                        "task_name": task,
                        "dataset_name": ds,
                        "model": model,
                        "value": compute_accuracy(g),
                        "errors": errors,
                    }
                )
            insert_results(db, "RtAR", rta_res)

        time.sleep(interval)


def main():
    db = get_db()
    compute_and_store_metrics(db, interval=120)


if __name__ == "__main__":
    main()
