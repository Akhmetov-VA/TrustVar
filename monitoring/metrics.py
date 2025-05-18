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

    # ДОПОЛНЕНИЕ: Expander с выбором двух метрик и сравнением
    with st.expander("Сравнение двух метрик на графике"):
        task_options = set()
        data_per_collection = {}

        for collection_name in results_collections:
            collection = db_client.get_collection(collection_name)
            records = list(collection.find())
            if not records:
                continue
            df = pd.DataFrame(records)
            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)
            if {"task_name", "model", "value"}.issubset(df.columns):
                task_options.update(df["task_name"].unique())
                data_per_collection[collection_name] = df

        task_options = sorted(task_options)
        selected_tasks = st.multiselect(
            "Выберите две задачи для сравнения:",
            task_options,
            max_selections=2,
            key="compare_task_names",
        )

        if len(selected_tasks) == 2:
            df_all = pd.concat(
                [
                    df[df["task_name"].isin(selected_tasks)][
                        ["task_name", "model", "value"]
                    ]
                    for df in data_per_collection.values()
                ]
            )
            pivot = df_all.pivot_table(
                index="model", columns="task_name", values="value"
            )

            if pivot.shape[1] == 2:
                st.subheader("Scatter Plot: Сравнение метрик")
                st.dataframe(pivot)

                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(pivot[selected_tasks[0]], pivot[selected_tasks[1]])

                for model, row in pivot.iterrows():
                    ax.text(
                        row[selected_tasks[0]] + 0.002,
                        row[selected_tasks[1]],
                        model,
                        fontsize=7,
                    )

                ax.set_xlabel(selected_tasks[0])
                ax.set_ylabel(selected_tasks[1])
                ax.set_title("Сравнение моделей по выбранным метрикам")
                st.pyplot(fig)
            else:
                st.warning("Недостаточно данных по обеим выбранным задачам.")
