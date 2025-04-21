# metrics.py
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from utils.db_client import MongoDBClient, MongoDBConfig

# Инициализация клиента БД
config = MongoDBConfig(database="TrustGen")
db_client = MongoDBClient(config)


def visualize_metrics(results_data: List[Dict[str, Any]], collection_name: str):
    results_df = pd.DataFrame(results_data)
    if "_id" in results_df.columns:
        results_df = results_df.drop(columns=["_id"])
    required_cols = {"task_name", "model", "value"}
    if not required_cols.issubset(results_df.columns):
        st.error("В данных отсутствуют необходимые поля (task_name, model, value).")
        return

    tasks = results_df["task_name"].unique()
    models = results_df["model"].unique()

    selected_tasks = st.multiselect(
        "Выберите задачу(и):",
        options=tasks,
        default=list(tasks),
        key=f"metrics_tasks_{collection_name}",
    )
    selected_models = st.multiselect(
        "Выберите модели:",
        options=models,
        default=list(models),
        key=f"metrics_models_{collection_name}",
    )

    filtered_df = results_df[
        (results_df["task_name"].isin(selected_tasks))
        & (results_df["model"].isin(selected_models))
    ]
    if filtered_df.empty:
        st.info("Нет данных для отображения с выбранными фильтрами.")
        return

    pivot_table = filtered_df.pivot_table(
        index="model", columns="task_name", values="value", aggfunc="mean"
    )
    st.subheader("Таблица метрик по задачам и моделям")
    st.dataframe(pivot_table)
    st.subheader("Визуализация метрик")
    st.bar_chart(pivot_table)


def render_metrics_tab():
    st.header("Метрики моделей")
    results_collections = ["RtAR", "TFNR", "Accuracy", "Correlation", "IncludeExclude"]
    if results_collections:
        selected_results_collection = st.selectbox(
            "Выберите коллекцию с метриками",
            options=results_collections,
            key="metrics_collection_selection",
        )
        results_collection = db_client.get_collection(selected_results_collection)
        results_data = list(results_collection.find())
        if results_data:
            visualize_metrics(results_data, selected_results_collection)
        else:
            st.info(f"Данные в коллекции '{selected_results_collection}' отсутствуют.")
    else:
        st.info("Нет доступных коллекций с метриками.")
