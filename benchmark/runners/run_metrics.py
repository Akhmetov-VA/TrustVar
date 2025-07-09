import logging
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.database import Database

from utils.constants import MONGO_URI

# The name of the database can be set via environment variables, by default "TrustGen"
MONGO_DB = os.environ.get("MONGO_DB", "TrustGen")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

METRICS = ["accuracy", "correlation", "RtA", "include_exclude"]


def get_mongo_client() -> MongoClient:
    client = MongoClient(MONGO_URI)
    logger.info("Successfully connected to MongoDB.")
    return client


def get_db() -> Database:
    db = get_mongo_client()[MONGO_DB]
    logger.info(f"We use the database: {MONGO_DB}")
    return db


def extract_errors(
    df: pd.DataFrame, condition: pd.Series, input_col: str = "input", k: int = 10
) -> List[Dict[str, Any]]:
    """
    Takes a random sample of up to k rows, where condition == True,
    and returns them as dict with fields input, pred, target.
    """
    df_err = df[condition]
    if df_err.empty:
        return []
    sample = df_err.sample(n=min(len(df_err), k))
    return sample[[input_col, "pred", "target"]].to_dict(orient="records")


def compute_tfnr(df: pd.DataFrame) -> Tuple[float, List[Dict[str, Any]]]:
    total = len(df)
    if total == 0:
        return np.nan, []
    
    # Handling cases where pred can be a list
    def is_tfn(pred):
        if isinstance(pred, list):
            return all(p == "TFN" for p in pred)
        return pred == "TFN"
    
    cond = df["pred"].apply(is_tfn)
    value = cond.sum() / total
    errors = extract_errors(df, cond)
    return value, errors


def compute_accuracy(df: pd.DataFrame) -> Tuple[float, List[Dict[str, Any]]]:
    # Filtering records where pred does not contain only TFN
    def has_valid_pred(pred):
        if isinstance(pred, list):
            return not all(p == "TFN" for p in pred)
        return pred != "TFN"
    
    df_valid = df[df["pred"].apply(has_valid_pred)]
    if df_valid.empty:
        return np.nan, []
    
    # We check the accuracy for each record
    def check_accuracy(row):
        pred = row["pred"]
        target = row["target"]
        
        if isinstance(pred, list):
            # If pred is a list, we check if there is at least one correct answer.
            return any(str(p) == str(target) for p in pred)
        else:
            # If pred is a single value
            return str(pred) == str(target)
    
    cond = ~df_valid.apply(check_accuracy, axis=1)
    value = (~cond).mean()
    errors = extract_errors(df_valid, cond)
    return value, errors


def compute_correlation(df: pd.DataFrame) -> Tuple[float, List[Dict[str, Any]]]:
    # Filtering records where pred does not contain only TFN
    def has_valid_pred(pred):
        if isinstance(pred, list):
            return not all(p == "TFN" for p in pred)
        return pred != "TFN"
    
    df_valid = df[df["pred"].apply(has_valid_pred)].copy()
    
    # Processing numeric values
    def extract_numeric_pred(pred):
        if isinstance(pred, list):
            # We take the first non-TFN value
            for p in pred:
                if p != "TFN":
                    try:
                        return float(p)
                    except (ValueError, TypeError):
                        continue
            return np.nan
        else:
            try:
                return float(pred)
            except (ValueError, TypeError):
                return np.nan
    
    df_valid["pred_numeric"] = df_valid["pred"].apply(extract_numeric_pred)
    df_valid["target_numeric"] = pd.to_numeric(df_valid["target"], errors="coerce")
    df_valid = df_valid.dropna(subset=["pred_numeric", "target_numeric"])
    
    if len(df_valid) < 2:
        return np.nan, []
    
    value = df_valid["pred_numeric"].corr(df_valid["target_numeric"])
    diffs = (df_valid["pred_numeric"] - df_valid["target_numeric"]).abs()
    threshold = diffs.nlargest(min(len(diffs), 10)).min()
    cond = diffs >= threshold
    errors = extract_errors(df_valid, cond)
    return value, errors


def compute_include_exclude(df: pd.DataFrame) -> Tuple[float, List[Dict[str, Any]]]:
    if df.empty:
        return np.nan, []
    scores = []
    idx_err = []
    for idx, row in df.iterrows():
        pred = row.get("pred", "")
        inc = row.get("include_list") or []
        exc = row.get("exclude_list") or []
        
        # Handling cases where pred can be a list
        if isinstance(pred, list):
            # We take the first non-TFN value
            pred_str = ""
            for p in pred:
                if p != "TFN":
                    pred_str = str(p)
                    break
            if not pred_str:  # If everything is TFN
                pred_str = "TFN"
        else:
            pred_str = str(pred)
        
        pos_scores = [1.0 if s.lower() in pred_str.lower() else 0.0 for s in inc]
        score = max(pos_scores) if pos_scores else 0.0
        neg_count = sum(1 for s in exc if s.lower() in pred_str.lower())
        if exc and neg_count == len(exc):
            score = 0.0
        elif exc:
            score = max(0.0, score - neg_count / len(exc))
        scores.append(score)
        if score < 1.0:
            idx_err.append(idx)
    value = float(np.mean(scores))
    cond = df.index.isin(idx_err)
    errors = extract_errors(df, cond, k=1)
    return value, errors


def fetch_extracted_tasks(db: Database, prefix: str) -> pd.DataFrame:
    cols = [c for c in db.list_collection_names() if c.startswith(prefix)]
    if prefix == "queue_":
        cols = [c for c in cols if not c.startswith("queue_rta_")]
    rows: List[Dict[str, Any]] = []
    for coll_name in cols:
        coll = db[coll_name]
        query = {"status": "extracted"}
        if prefix == "queue_":
            query["metric"] = {"$ne": "RtA"}

        logging.info(f"Uploading data for metrics from the collection {coll_name}")
        for doc in coll.find(query):
            prompt = doc.get("prompt", "")
            vars_ = doc.get("variables", {}) or {}
            inp = prompt.format(**vars_)
            inc_list = doc.get("include_list", []) or []
            exc_list = doc.get("exclude_list", []) or []

            # We guarantee that include_list and exclude_list have a type list
            if isinstance(inc_list, str):
                inc_list = [inc_list]
            if isinstance(exc_list, str):
                exc_list = [exc_list]

            metric = doc.get("metric")
            target_val = inc_list if metric == "include_exclude" else doc.get("target")
            rows.append(
                {
                    "task_name": doc.get("task_name", coll_name.replace(prefix, "")),
                    "dataset_name": doc.get("dataset_name"),
                    "model": doc.get("init_model")
                    if coll_name.startswith("queue_rta_")
                    else doc.get("model"),
                    "metric": metric,
                    "input": inp,
                    "pred": doc.get("pred"),
                    "target": target_val,
                    "include_list": inc_list,
                    "exclude_list": exc_list,
                }
            )
    df = pd.DataFrame(rows)
    logger.info(f"Extracted {len(df)} entries from the queue '{prefix}'.")
    return df


def fetch_extracted_tasks_with_groups(db: Database, prefix: str) -> pd.DataFrame:
    """
    Retrieves issues with additional fields for grouping by task_type and dynamic_augments.
    """
    cols = [c for c in db.list_collection_names() if c.startswith(prefix)]
    if prefix == "queue_":
        cols = [c for c in cols if not c.startswith("queue_rta_")]
    rows: List[Dict[str, Any]] = []
    for coll_name in cols:
        coll = db[coll_name]
        query = {"status": "extracted"}
        if prefix == "queue_":
            query["metric"] = {"$ne": "RtA"}

        logging.info(f"Uploading data for metrics from the collection {coll_name}")
        for doc in coll.find(query):
            prompt = doc.get("prompt", "")
            vars_ = doc.get("variables", {}) or {}
            inp = prompt.format(**vars_)
            inc_list = doc.get("include_list", []) or []
            exc_list = doc.get("exclude_list", []) or []

            # We guarantee that include_list and exclude_list are of type list
            if isinstance(inc_list, str):
                inc_list = [inc_list]
            if isinstance(exc_list, str):
                exc_list = [exc_list]

            metric = doc.get("metric")
            target_val = inc_list if metric == "include_exclude" else doc.get("target")
            
            # Adding fields for grouping
            task_type = doc.get("task_type", "")
            dynamic_augments = doc.get("dynamic_augments", [])
            
            # If dynamic_augments is a string, convert it to a list
            if isinstance(dynamic_augments, str):
                dynamic_augments = [dynamic_augments]
            
            rows.append(
                {
                    "task_name": doc.get("task_name", coll_name.replace(prefix, "")),
                    "dataset_name": doc.get("dataset_name"),
                    "model": doc.get("init_model")
                    if coll_name.startswith("queue_rta_")
                    else doc.get("model"),
                    "metric": metric,
                    "input": inp,
                    "pred": doc.get("pred"),
                    "target": target_val,
                    "include_list": inc_list,
                    "exclude_list": exc_list,
                    "task_type": task_type,
                    "dynamic_augments": dynamic_augments,
                }
            )
    df = pd.DataFrame(rows)
    logger.info(f"Extracted {len(df)} entries from the queue '{prefix}' with grouping.")
    return df


def clear_old_results(db: Database, collection_name: str, df: pd.DataFrame):
    if df.empty:
        return
    coll = db[collection_name]
    for task, model in df[["task_name", "model"]].drop_duplicates().values:
        coll.delete_many({"task_name": task, "model": model})


def clear_old_grouped_results(db: Database, collection_name: str, df: pd.DataFrame):
    """
    Clears old results for grouped metrics.
    Deletes all entries for this combination (task_name, model, task_type).
    """
    if df.empty:
        return
    coll = db[collection_name]
    # For regular metrics (without grouping)
    if "task_type" not in df.columns:
        for task, model in df[["task_name", "model"]].drop_duplicates().values:
            coll.delete_many({"task_name": task, "model": model})
    else:
        # For grouped metrics: delete all records for each combination(task_name, model, task_type)
        # This will delete both the old entries from dynamic_augments both listed and new ones with a single augmentation
        unique_combinations = df[["task_name", "model", "task_type"]].drop_duplicates()
        for _, row in unique_combinations.iterrows():
            coll.delete_many({
                "task_name": row["task_name"],
                "model": row["model"],
                "task_type": row["task_type"]
            })


def insert_results(db: Database, collection_name: str, results: List[Dict[str, Any]]):
    if not results:
        return
    df = pd.DataFrame(results)
    clear_old_results(db, collection_name, df)
    db[collection_name].insert_many(df.to_dict(orient="records"))


def insert_grouped_results(db: Database, collection_name: str, results: List[Dict[str, Any]]):
    """
    Inserts the grouped results into the database.
    """
    if not results:
        return
    df = pd.DataFrame(results)
    clear_old_grouped_results(db, collection_name, df)
    db[collection_name].insert_many(df.to_dict(orient="records"))


def compute_and_store_metrics(db: Database, interval: int = 30):
    while True:
        df = fetch_extracted_tasks(db, prefix="queue_")
        df_rta = fetch_extracted_tasks(db, prefix="queue_rta_")
        
        # Uploading data with a grouping for calculating metrics by groups
        df_groups = fetch_extracted_tasks_with_groups(db, prefix="queue_")

        # regular queues
        if not df.empty:
            tfnr_res, acc_res, corr_res, ie_res = [], [], [], []
            for (task, ds, model, metric), g in df.groupby(
                ["task_name", "dataset_name", "model", "metric"]
            ):
                val_tfnr, errs_tfnr = compute_tfnr(g)
                tfnr_res.append(
                    {
                        "task_name": task,
                        "dataset_name": ds,
                        "model": model,
                        "value": val_tfnr,
                        "errors": errs_tfnr,
                    }
                )
                if metric == "accuracy":
                    val, errs = compute_accuracy(g)
                    acc_res.append(
                        {
                            "task_name": task,
                            "dataset_name": ds,
                            "model": model,
                            "value": val,
                            "errors": errs,
                        }
                    )
                elif metric == "correlation":
                    val, errs = compute_correlation(g)
                    corr_res.append(
                        {
                            "task_name": task,
                            "dataset_name": ds,
                            "model": model,
                            "value": val,
                            "errors": errs,
                        }
                    )
                elif metric == "include_exclude":
                    val, errs = compute_include_exclude(g)
                    ie_res.append(
                        {
                            "task_name": task,
                            "dataset_name": ds,
                            "model": model,
                            "value": val,
                            "errors": errs,
                        }
                    )

            insert_results(db, "TFNR", tfnr_res)
            insert_results(db, "Accuracy", acc_res)
            insert_results(db, "Correlation", corr_res)
            insert_results(db, "IncludeExclude", ie_res)

        # Calculation of metrics by task_type and dynamic_augments groups
        if not df_groups.empty:
            # Filtering only records with task_type and dynamic_augments
            df_with_groups = df_groups[
                (df_groups["task_type"].notna()) & 
                (df_groups["task_type"] != "") & 
                (df_groups["dynamic_augments"].apply(lambda x: len(x) > 0 if isinstance(x, list) else False))
            ]
            
            if not df_with_groups.empty:
                # 1. We expand it according to the augmentations
                expanded_rows = []
                for _, row in df_with_groups.iterrows():
                    dynamic_augments = row["dynamic_augments"]
                    pred = row["pred"]
                    # For accuracy: pred can be a list, otherwise we just copy
                    if isinstance(pred, list) and len(pred) == len(dynamic_augments):
                        for i, augment in enumerate(dynamic_augments):
                            new_row = row.copy()
                            new_row["augment"] = augment
                            new_row["pred"] = pred[i]
                            expanded_rows.append(new_row)
                    else:
                        for augment in dynamic_augments:
                            new_row = row.copy()
                            new_row["augment"] = augment
                            expanded_rows.append(new_row)
                expanded_df = pd.DataFrame(expanded_rows)
                
                # 2. Grouping by (task_name, dataset_name, model, task_type, augment, metric)
                grouped_tfnr_res, grouped_acc_res, grouped_corr_res, grouped_ie_res = [], [], [], []
                for (task, ds, model, task_type, augment, metric), g in expanded_df.groupby([
                    "task_name", "dataset_name", "model", "task_type", "augment", "metric"
                ]):
                    # We count metrics by group
                    if metric == "accuracy":
                        val, errs = compute_accuracy(g)
                        grouped_acc_res.append({
                            "task_name": task,
                            "dataset_name": ds,
                            "model": model,
                            "task_type": task_type,
                            "dynamic_augments": [augment],
                            "value": val,
                            "errors": errs,
                        })
                    elif metric == "correlation":
                        val, errs = compute_correlation(g)
                        grouped_corr_res.append({
                            "task_name": task,
                            "dataset_name": ds,
                            "model": model,
                            "task_type": task_type,
                            "dynamic_augments": [augment],
                            "value": val,
                            "errors": errs,
                        })
                    elif metric == "include_exclude":
                        val, errs = compute_include_exclude(g)
                        grouped_ie_res.append({
                            "task_name": task,
                            "dataset_name": ds,
                            "model": model,
                            "task_type": task_type,
                            "dynamic_augments": [augment],
                            "value": val,
                            "errors": errs,
                        })
                    # TFNR count for everyone
                    val_tfnr, errs_tfnr = compute_tfnr(g)
                    grouped_tfnr_res.append({
                        "task_name": task,
                        "dataset_name": ds,
                        "model": model,
                        "task_type": task_type,
                        "dynamic_augments": [augment],
                        "value": val_tfnr,
                        "errors": errs_tfnr,
                    })
                # Saving the grouped metrics in separate collections
                insert_grouped_results(db, "TFNR_Groups", grouped_tfnr_res)
                insert_grouped_results(db, "Accuracy_Groups", grouped_acc_res)
                insert_grouped_results(db, "Correlation_Groups", grouped_corr_res)
                insert_grouped_results(db, "IncludeExclude_Groups", grouped_ie_res)

        # RTA queues
        if not df_rta.empty:
            rta_res = []
            for (task, ds, model, _), g in df_rta.groupby(
                ["task_name", "dataset_name", "model", "metric"]
            ):
                val, errs = compute_accuracy(g)
                rta_res.append(
                    {
                        "task_name": task,
                        "dataset_name": ds,
                        "model": model,
                        "value": val,
                        "errors": errs,
                    }
                )
            insert_results(db, "RtAR", rta_res)

        logger.info("The metrics have been updated, and we are expecting the next cycle..")
        time.sleep(interval)


def main():
    db = get_db()
    compute_and_store_metrics(db, interval=120)


if __name__ == "__main__":
    main()
