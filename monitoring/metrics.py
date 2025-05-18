from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # 🔹 Для интерактивной heatmap
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

    # 🔽 Интерактивное сравнение и корреляция
    with st.expander("Сравнение метрик и корреляции между задачами"):
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

        # --- Сравнение двух задач ---
        selected_tasks = st.multiselect(
            "Выберите две задачи для scatter-графика:",
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
            ).dropna()

            if pivot.shape[1] == 2:
                st.subheader("Интерактивный график: сравнение метрик")
                st.dataframe(pivot)

                fig = px.scatter(
                    pivot,
                    x=selected_tasks[0],
                    y=selected_tasks[1],
                    text=pivot.index,
                    labels={
                        selected_tasks[0]: selected_tasks[0],
                        selected_tasks[1]: selected_tasks[1],
                    },
                    title="Сравнение моделей по выбранным метрикам",
                )
                fig.update_traces(textposition="top center")
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Недостаточно данных по обеим выбранным задачам.")

        # --- Корреляция между множеством задач ---
        selected_corr_tasks = st.multiselect(
            "Выберите задачи для анализа корреляции:",
            task_options,
            key="correlation_tasks",
        )

        if len(selected_corr_tasks) >= 2:
            df_corr = pd.concat(
                [
                    df[df["task_name"].isin(selected_corr_tasks)][
                        ["task_name", "model", "value"]
                    ]
                    for df in data_per_collection.values()
                ]
            )
            pivot_corr = df_corr.pivot_table(
                index="model", columns="task_name", values="value"
            ).dropna()

            if not pivot_corr.empty:
                st.subheader("Корреляционная матрица задач")
                st.dataframe(pivot_corr.corr().round(2))

                # 🔹 Построение интерактивной heatmap
                corr_matrix = pivot_corr.corr()
                fig = go.Figure(
                    data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale="RdBu",
                        zmin=-1,
                        zmax=1,
                        colorbar=dict(title="Корреляция"),
                        hovertemplate="Задачи: %{y} и %{x}<br>Значение: %{z:.2f}<extra></extra>",
                    )
                )
                fig.update_layout(
                    title="Интерактивная корреляционная матрица задач",
                    xaxis=dict(title=""),
                    yaxis=dict(title="", autorange="reversed"),
                    height=600,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Недостаточно данных для построения корреляционной матрицы.")
