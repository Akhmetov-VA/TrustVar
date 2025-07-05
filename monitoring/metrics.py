import json
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # 🔹 Для интерактивной heatmap
import streamlit as st
import numpy as np
# import logging
from utils.db_client import MongoDBClient, MongoDBConfig


# Настройка логирования
# logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
# logger = logging.getLogger(__name__)

# Инициализация клиента БД
config = MongoDBConfig(database="TrustGen")
db_client = MongoDBClient(config)


def calculate_coefficient_of_variation(values: List[float]) -> float:
    """Вычисляет коэффициент вариации (CV = std/mean * 100%)."""
    if not values or len(values) < 2:
        return np.nan
    mean_val = np.mean(values)
    if mean_val == 0:
        return np.nan
    std_val = np.std(values)
    return (std_val / mean_val) * 100


def visualize_grouped_metrics(results_data: List[Dict[str, Any]], collection_name: str):
    """Визуализация метрик по группам task_type и dynamic_augments."""
    results_df = pd.DataFrame(results_data)
    if "_id" in results_df.columns:
        results_df = results_df.drop(columns=["_id"])
    
    required_cols = {"task_name", "model", "value", "task_type", "dynamic_augments"}
    if not required_cols.issubset(results_df.columns):
        st.error("В данных отсутствуют необходимые поля для группированных метрик.")
        return

    # Фильтруем только задачи типа "Compare model behaviour"
    compare_df = results_df[results_df["task_type"] == "Compare model behaviour"].copy()
    
    if compare_df.empty:
        st.info("Нет данных для задач типа 'Compare model behaviour'.")
        return

    # Преобразуем списки dynamic_augments в строки для удобства отображения
    compare_df["augments_str"] = compare_df["dynamic_augments"].apply(
        lambda x: " + ".join(sorted(x)) if isinstance(x, list) else str(x)
    )

    # Выборка по задачам и моделям
    tasks = compare_df["task_name"].unique()
    models = compare_df["model"].unique()
    augments = compare_df["augments_str"].unique()
    
    selected_tasks = st.multiselect(
        "Выберите задачу(и):",
        options=tasks,
        default=list(tasks),
        key=f"grouped_metrics_tasks_{collection_name}",
    )
    selected_models = st.multiselect(
        "Выберите модели:",
        options=models,
        default=list(models),
        key=f"grouped_metrics_models_{collection_name}",
    )
    selected_augments = st.multiselect(
        "Выберите аугментации:",
        options=augments,
        default=list(augments),
        key=f"grouped_metrics_augments_{collection_name}",
    )

    filtered_df = compare_df[
        (compare_df["task_name"].isin(selected_tasks))
        & (compare_df["model"].isin(selected_models))
        & (compare_df["augments_str"].isin(selected_augments))
    ]
    
    if filtered_df.empty:
        st.info("Нет данных для отображения с выбранными фильтрами.")
        return

    # 1. Таблица метрик по аугментациям
    st.subheader("Метрики по аугментациям")
    pivot_augments = filtered_df.pivot_table(
        index=["model", "task_name"], 
        columns="augments_str", 
        values="value", 
        aggfunc="mean"
    )
    st.dataframe(pivot_augments.round(3))

    # 2. График сравнения аугментаций
    st.subheader("Сравнение влияния аугментаций на метрики")
    
    # Группируем по модели и задаче для построения графика
    fig_data = filtered_df.groupby(["model", "task_name", "augments_str"])["value"].mean().reset_index()
    
    if not fig_data.empty:
        fig = px.bar(
            fig_data,
            x="augments_str",
            y="value",
            color="model",
            facet_col="task_name",
            title="Влияние аугментаций на производительность моделей",
            labels={"value": f"Метрика ({collection_name})", "augments_str": "Аугментации"}
        )
        fig.update_xaxes(tickangle=45)
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    # 3. Коэффициент вариации для оценки устойчивости
    st.subheader("Коэффициент вариации (устойчивость к аугментациям)")
    
    # Вычисляем CV для каждой модели и задачи
    cv_data = []
    for (model, task), group in filtered_df.groupby(["model", "task_name"]):
        values = group["value"].tolist()
        cv = calculate_coefficient_of_variation(values)
        cv_data.append({
            "model": model,
            "task_name": task,
            "cv": cv,
            "mean_value": np.mean(values),
            "std_value": np.std(values),
            "min_value": np.min(values),
            "max_value": np.max(values),
            "num_augments": len(values)
        })
    
    cv_df = pd.DataFrame(cv_data)
    
    if not cv_df.empty:
        # Сортируем по CV (меньше CV = более устойчивая модель)
        cv_df = cv_df.sort_values("cv")
        
        st.write("**Интерпретация CV:**")
        st.write("- CV < 10%: очень устойчивая модель")
        st.write("- CV 10-20%: устойчивая модель") 
        st.write("- CV 20-30%: умеренно устойчивая модель")
        st.write("- CV > 30%: неустойчивая модель")
        
        st.dataframe(cv_df.round(3))
        
        # График CV
        fig_cv = px.bar(
            cv_df,
            x="model",
            y="cv",
            color="task_name",
            title="Коэффициент вариации по моделям и задачам (меньше = устойчивее)",
            labels={"cv": "Коэффициент вариации (%)", "model": "Модель"}
        )
        fig_cv.update_layout(height=500)
        st.plotly_chart(fig_cv, use_container_width=True)
        
        # Heatmap CV
        cv_pivot = cv_df.pivot_table(
            index="model", 
            columns="task_name", 
            values="cv", 
            aggfunc="mean"
        )
        
        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=cv_pivot.values,
                x=cv_pivot.columns,
                y=cv_pivot.index,
                colorscale="RdYlGn_r",  # Зеленый = устойчивая, красный = неустойчивая
                zmin=0,
                zmax=cv_pivot.values.max(),
                colorbar=dict(title="CV (%)"),
                hovertemplate="Модель: %{y}<br>Задача: %{x}<br>CV: %{z:.1f}%<extra></extra>",
            )
        )
        fig_heatmap.update_layout(
            title="Тепловая карта коэффициента вариации (устойчивость к аугментациям)",
            xaxis=dict(title="Задача"),
            yaxis=dict(title="Модель"),
            height=500,
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # 4. Детальный анализ по каждой аугментации
    with st.expander("Детальный анализ по аугментациям"):
        for augment in selected_augments:
            st.write(f"**Аугментация: {augment}**")
            augment_data = filtered_df[filtered_df["augments_str"] == augment]
            
            if not augment_data.empty:
                # Сравнение моделей для данной аугментации
                fig_augment = px.bar(
                    augment_data,
                    x="model",
                    y="value",
                    color="task_name",
                    title=f"Производительность моделей при аугментации: {augment}",
                    labels={"value": f"Метрика ({collection_name})"}
                )
                fig_augment.update_layout(height=400)
                st.plotly_chart(fig_augment, use_container_width=True)
                
                # Таблица значений
                pivot_augment = augment_data.pivot_table(
                    index="model", 
                    columns="task_name", 
                    values="value", 
                    aggfunc="mean"
                )
                st.dataframe(pivot_augment.round(3))


def visualize_metrics(results_data: List[Dict[str, Any]], collection_name: str):
    results_df = pd.DataFrame(results_data)
    if "_id" in results_df.columns:
        results_df = results_df.drop(columns=["_id"])
    required_cols = {"task_name", "model", "value"}
    if not required_cols.issubset(results_df.columns):
        st.error("В данных отсутствуют необходимые поля (task_name, model, value).")
        return

    # logger.info(results_df.columns)
    # logger.info(results_df.shape)
    # logger.info(results_df.head())
    # logger.info(f' Tsks {len(results_df["task_name"].unique())}')
    # logger.info(f' Models {len(results_df["model"].unique())}')

    # выборка по задачам и моделям
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

    # табличное и графическое представление метрик
    pivot_table = filtered_df.pivot_table(
        index="model", columns="task_name", values="value", aggfunc="mean"
    )
    st.subheader("Таблица метрик по задачам и моделям")
    st.dataframe(pivot_table)
    st.subheader("Визуализация метрик")
    st.bar_chart(pivot_table)

    # 🔽 Новый expander: показать ошибки в виде DataFrame
    with st.expander("Просмотр топ-10 ошибок по выбранным задачам и моделям"):
        if "errors" not in filtered_df.columns:
            st.info("Для этой метрики нет сохранённых ошибок.")
        else:
            df_err = (
                filtered_df[["task_name", "model", "errors"]]
                .dropna(subset=["errors"])
                .drop_duplicates(subset=["task_name", "model"])
            )
            if df_err.empty:
                st.info("Ошибок не найдено.")
            else:
                df_err["errors"] = df_err["errors"].apply(
                    lambda errs: json.dumps(errs, ensure_ascii=False, indent=2)
                )
                df_to_show = df_err.set_index(["task_name", "model"])
                st.dataframe(df_to_show)


def render_grouped_metrics_tab():
    """Рендерит вкладку с группированными метриками."""
    st.header("Анализ метрик по группам (Compare model behaviour)")
    
    # Коллекции с группированными метриками
    grouped_collections = ["Accuracy_Groups", "Correlation_Groups", "IncludeExclude_Groups"]
    available_collections = []
    
    for coll in grouped_collections:
        try:
            collection = db_client.get_collection(coll)
            if collection.count_documents({}) > 0:
                available_collections.append(coll)
        except:
            continue
    
    if not available_collections:
        st.info("Нет доступных коллекций с группированными метриками.")
        return
    
    selected_collection = st.selectbox(
        "Выберите коллекцию с группированными метриками:",
        options=available_collections,
        key="grouped_metrics_collection_selection",
    )
    
    results_collection = db_client.get_collection(selected_collection)
    results_data = list(results_collection.find())
    
    if results_data:
        visualize_grouped_metrics(results_data, selected_collection)
    else:
        st.info(f"Данные в коллекции '{selected_collection}' отсутствуют.")


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
    with st.expander("Сравнение метрик и корреляция между задачами"):
        task_options = set()
        data_per_collection: Dict[str, pd.DataFrame] = {}
        for coll in results_collections:
            if coll == "TFNR":
                continue
            recs = list(db_client.get_collection(coll).find())
            if not recs:
                continue
            df = pd.DataFrame(recs)
            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)
            if {"task_name", "model", "value"}.issubset(df.columns):
                task_options.update(df["task_name"].unique())
                data_per_collection[coll] = df
        task_options = sorted(task_options)

        # --- scatter plot для двух задач ---
        sel = st.multiselect(
            "Выберите две задачи для scatter-графика:",
            task_options,
            max_selections=3,
            key="compare_task_names",
        )
        if len(sel) >= 2:
            df_all = pd.concat(
                [
                    df[df["task_name"].isin(sel)][["task_name", "model", "value"]]
                    for df in data_per_collection.values()
                ][:2]
            )
            pivot = df_all.pivot_table(
                index="model", columns="task_name", values="value"
            ).dropna()
            if pivot.shape[1] == 2:
                st.subheader("Интерактивный график: сравнение метрик")
                st.dataframe(pivot)
                fig = px.scatter(
                    pivot,
                    x=sel[0],
                    y=sel[1],
                    text=pivot.index,
                    labels={sel[0]: sel[0], sel[1]: sel[1]},
                    title="Сравнение моделей по выбранным метрикам",
                )
                fig.update_traces(textposition="top center")
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Недостаточно данных для scatter-графика.")

        # --- интерактивная корреляция для списка задач ---
        corr_sel = st.multiselect(
            "Выберите задачи для анализа корреляции:",
            task_options,
            key="correlation_tasks",
        )
        if len(corr_sel) >= 2:
            df_corr = pd.concat(
                [
                    df[df["task_name"].isin(corr_sel)][["task_name", "model", "value"]]
                    for df in data_per_collection.values()
                ]
            )
            pivot_corr = df_corr.pivot_table(
                index="model", columns="task_name", values="value"
            ).dropna()
            if not pivot_corr.empty:
                st.subheader("Корреляционная матрица задач")
                st.dataframe(pivot_corr.corr().round(2))
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
