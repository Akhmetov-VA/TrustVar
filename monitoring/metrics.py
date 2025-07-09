import json
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # 🔹 For an interactive heatmap
import streamlit as st
import numpy as np
# import logging
from utils.db_client import MongoDBClient, MongoDBConfig


# Configuring logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
# logger = logging.getLogger(__name__)

# Initializing the database client
config = MongoDBConfig(database="TrustGen")
db_client = MongoDBClient(config)


def calculate_coefficient_of_variation(values: List[float]) -> float:
    """Calculates the coefficient of variation (CV = std/mean * 100%)."""
    if not values or len(values) < 2:
        return np.nan
    mean_val = np.mean(values)
    if mean_val == 0:
        return np.nan
    std_val = np.std(values)
    return (std_val / mean_val) * 100


def visualize_grouped_metrics(results_data: List[Dict[str, Any]], collection_name: str):
    """Visualization of metrics by groups task_type and dynamic_augments."""
    results_df = pd.DataFrame(results_data)
    if "_id" in results_df.columns:
        results_df = results_df.drop(columns=["_id"])
    
    required_cols = {"task_name", "model", "value", "task_type", "dynamic_augments"}
    if not required_cols.issubset(results_df.columns):
        st.error("The required fields for grouped metrics are missing in the data.")
        return

    # We only filter tasks like "Compare model behaviour"
    compare_df = results_df[results_df["task_type"] == "Compare model behaviour"].copy()
    
    if compare_df.empty:
        st.info("There is no data for tasks like 'Compare model behaviour'.")
        return

    # Expanding the dynamic_augments lists into separate lines
    expanded_rows = []
    for _, row in compare_df.iterrows():
        dynamic_augments = row["dynamic_augments"]
        if isinstance(dynamic_augments, list):
            for augment in dynamic_augments:
                new_row = row.copy()
                new_row["augment"] = augment
                expanded_rows.append(new_row)
        else:
            new_row = row.copy()
            new_row["augment"] = str(dynamic_augments)
            expanded_rows.append(new_row)
    
    expanded_df = pd.DataFrame(expanded_rows)
    
    if expanded_df.empty:
        st.info("There is no data to display after the augmentations are deployed.")
        return

    # Selection by tasks, models, and augmentations
    tasks = expanded_df["task_name"].unique()
    models = expanded_df["model"].unique()
    augments = expanded_df["augment"].unique()
    
    selected_tasks = st.multiselect(
        "Select the task(s):",
        options=tasks,
        default=list(tasks),
        key=f"grouped_metrics_tasks_{collection_name}",
    )
    selected_models = st.multiselect(
        "Select models:",
        options=models,
        default=list(models),
        key=f"grouped_metrics_models_{collection_name}",
    )
    selected_augments = st.multiselect(
        "Choose Augmentation:",
        options=augments,
        default=list(augments),
        key=f"grouped_metrics_augments_{collection_name}",
    )

    filtered_df = expanded_df[
        (expanded_df["task_name"].isin(selected_tasks))
        & (expanded_df["model"].isin(selected_models))
        & (expanded_df["augment"].isin(selected_augments))
    ]
    
    if filtered_df.empty:
        st.info("There is no data to display with the selected filters..")
        return

    # After creating results_df (or filtered_df), add augment processing
    if "augment" not in filtered_df.columns and "dynamic_augments" in filtered_df.columns:
        filtered_df["augment"] = filtered_df["dynamic_augments"].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) == 1 else str(x)
        )

    # 1. The table of metrics for augmentations
    st.subheader("Augmentation metrics")
    pivot_augments = filtered_df.pivot_table(
        index=["model", "task_name"], 
        columns="augment", 
        values="value", 
        aggfunc="mean"
    )
    st.dataframe(pivot_augments.round(3))

    # 2. Augmentation comparison chart
    st.subheader("Comparing the impact of augmentation on metrics")
    
    # Grouping by model and tasks for plotting
    fig_data = filtered_df.groupby(["model", "task_name", "augment"])["value"].mean().reset_index()
    
    if not fig_data.empty:
        fig = px.bar(
            fig_data,
            x="augment",
            y="value",
            color="model",
            facet_col="task_name",
            title="The impact of augmentation on model performance",
            labels={"value": f"Metric ({collection_name})", "augment": "Augmentation"}
        )
        fig.update_xaxes(tickangle=45)
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    # 3. Gossamer (Radar Chart) for each model
    st.subheader("Cobweb (Radar Chart) - augmentation performance")
    
    # Choosing one model for a spider web
    selected_model_for_radar = st.selectbox(
        "Choose a model for a spider web:",
        options=selected_models,
        key=f"radar_model_{collection_name}"
    )
    
    radar_data = filtered_df[
        (filtered_df["model"] == selected_model_for_radar) &
        (filtered_df["task_name"].isin(selected_tasks))
    ]
    
    if not radar_data.empty:
        # Creating a web for each task
        for task in selected_tasks:
            task_data = radar_data[radar_data["task_name"] == task]
            if not task_data.empty:
                # Grouping by augmentation
                task_pivot = task_data.groupby("augment")["value"].mean().reset_index()
                if len(task_pivot) >= 3:  # You need at least 3 points for a spider web
                    # Sorting augmentations for consistent display
                    task_pivot = task_pivot.sort_values("augment")
                    augment_names = [short_augment_name(a) for a in task_pivot["augment"].tolist()]
                    values = task_pivot["value"].tolist()
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=augment_names,
                        fill='toself',
                        name=f'{task}',
                        line_color='blue',
                        line_width=2
                    ))
                    max_value = max(values) if values else 1.0
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, max_value * 1.1],
                                tickfont=dict(size=10)
                            )
                        ),
                        showlegend=True,
                        title=f"Gossamer for modeling {selected_model_for_radar} - task {task}",
                        height=500
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                else:
                    st.info(f"Not enough data for a spider web for the task {task} (You need at least 3 augmentations)")

    # 4. Coefficient of variation for stability assessment
    st.subheader("Coefficient of variation (resistance to augmentation)")
    
    # We calculate the CV for each model and task
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
        # We sort by CM (less CM = more stable model)
        cv_df = cv_df.sort_values("cv")
        
        st.write("**Interpretation CV:**")
        st.write("- CV < 10%: a very stable model")
        st.write("- CV 10-20%: a sustainable model") 
        st.write("- CV 20-30%: a moderately stable model")
        st.write("- CV > 30%: an unstable model")
        
        st.dataframe(cv_df.round(3))
        
        # Chart CV
        fig_cv = px.bar(
            cv_df,
            x="model",
            y="cv",
            color="task_name",
            title="Coefficient of variation for models and tasks (less = more stable)",
            labels={"cv": "Coefficient of variation (%)", "model": "Model"}
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
        
        # We check that the array is not empty before calling max()
        if cv_pivot.size > 0 and not cv_pivot.isna().all().all():
            max_cv = cv_pivot.values.max()
        else:
            max_cv = 100  # Default value
        
        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=cv_pivot.values,
                x=cv_pivot.columns,
                y=cv_pivot.index,
                colorscale="RdYlGn_r",  # Green = steady, red = unstable
                zmin=0,
                zmax=max_cv,
                colorbar=dict(title="CV (%)"),
                hovertemplate="Model: %{y}<br>Task: %{x}<br>CV: %{z:.1f}%<extra></extra>",
            )
        )
        fig_heatmap.update_layout(
            title="Heat map of the coefficient of variation (resistance to augmentation)",
            xaxis=dict(title="Task"),
            yaxis=dict(title="Model"),
            height=500,
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # 5. Detailed analysis for each augmentation
    with st.expander("Detailed augmentation analysis"):
        for augment in selected_augments:
            st.write(f"**Augmentation: {augment}**")
            augment_data = filtered_df[filtered_df["augment"] == augment]
            
            if not augment_data.empty:
                # Comparison of models for this augmentation
                fig_augment = px.bar(
                    augment_data,
                    x="model",
                    y="value",
                    color="task_name",
                    title=f"Model performance during augmentation: {augment}",
                    labels={"value": f"Metric ({collection_name})"}
                )
                fig_augment.update_layout(height=400)
                st.plotly_chart(fig_augment, use_container_width=True)
                
                # Table of values
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
        st.error("The required fields are missing in the data (task_name, model, value).")
        return

    # logger.info(results_df.columns)
    # logger.info(results_df.shape)
    # logger.info(results_df.head())
    # logger.info(f' Tsks {len(results_df["task_name"].unique())}')
    # logger.info(f' Models {len(results_df["model"].unique())}')

    # selection by tasks and models
    tasks = results_df["task_name"].unique()
    models = results_df["model"].unique()
    selected_tasks = st.multiselect(
        "Select the task(s):",
        options=tasks,
        default=list(tasks),
        key=f"metrics_tasks_{collection_name}",
    )
    selected_models = st.multiselect(
        "Select models:",
        options=models,
        default=list(models),
        key=f"metrics_models_{collection_name}",
    )
    filtered_df = results_df[
        (results_df["task_name"].isin(selected_tasks))
        & (results_df["model"].isin(selected_models))
    ]
    if filtered_df.empty:
        st.info("There is no data to display with the selected filters..")
        return

    # After creating results_df (or filtered_df), add processing augment
    if "augment" not in filtered_df.columns and "dynamic_augments" in filtered_df.columns:
        filtered_df["augment"] = filtered_df["dynamic_augments"].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) == 1 else str(x)
        )

    # tabular and graphical representation of metrics
    pivot_table = filtered_df.pivot_table(
        index="model", columns="task_name", values="value", aggfunc="mean"
    )
    st.subheader("The metric table by tasks and models")
    st.dataframe(pivot_table)
    st.subheader("Visualization of metrics")
    st.bar_chart(pivot_table)

    # 🔽 New expander: show errors as a DataFrame
    with st.expander("View the top 10 errors for selected tasks and models"):
        if "errors" not in filtered_df.columns:
            st.info("There are no saved errors for this metric.")
        else:
            df_err = (
                filtered_df[["task_name", "model", "errors"]]
                .dropna(subset=["errors"])
                .drop_duplicates(subset=["task_name", "model"])
            )
            if df_err.empty:
                st.info("No errors found.")
            else:
                df_err["errors"] = df_err["errors"].apply(
                    lambda errs: json.dumps(errs, ensure_ascii=False, indent=2)
                )
                df_to_show = df_err.set_index(["task_name", "model"])
                st.dataframe(df_to_show)


def short_augment_name(name):
    mapping = {
        "Synonymy": "Syn",
        "Stylistic change": "Style",
        "Reorder words/phrases": "Reorder",
        "Shorten sentence length": "Shorten",
        "Increase sentence length": "Length+",
        "Paraphrasing": "Paraph"
    }
    return mapping.get(name, str(name)[:8])


def render_metrics_tab():
    st.header("Model metrics")
    
    # Switch between metric types
    metric_type = st.radio(
        "Select the type of metric analysis:",
        ["Common metrics", "Group analysis(Compare model behaviour)"],
        key="metrics_type_selection"
    )
    
    if metric_type == "Common metrics":
        # Original logic for common metrics
        results_collections = ["RtAR", "TFNR", "Accuracy", "Correlation", "IncludeExclude"]
        if results_collections:
            selected_results_collection = st.selectbox(
                "Select a collection with metrics",
                options=results_collections,
                key="metrics_collection_selection",
            )
            results_collection = db_client.get_collection(selected_results_collection)
            results_data = list(results_collection.find())
            if results_data:
                visualize_metrics(results_data, selected_results_collection)
            else:
                st.info(f"Data in the collection '{selected_results_collection}' missing.")
        else:
            st.info("There are no available collections with metrics.")

        # 🔽 Interactive comparison and correlation
        with st.expander("Comparison of metrics and correlation between tasks"):
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

            # --- scatter plot for two tasks ---
            sel = st.multiselect(
                "Select two tasks for the scatter chart:",
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
                    st.subheader("Interactive graph: comparison of metrics")
                    st.dataframe(pivot)
                    fig = px.scatter(
                        pivot,
                        x=sel[0],
                        y=sel[1],
                        text=pivot.index,
                        labels={sel[0]: sel[0], sel[1]: sel[1]},
                        title="Comparison of models by selected metrics",
                    )
                    fig.update_traces(textposition="top center")
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Insufficient data for the scatter chart.")

            # --- interactive correlation for the task list ---
            corr_sel = st.multiselect(
                "Select tasks for correlation analysis:",
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
                    st.subheader("Correlation matrix of tasks")
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
                            colorbar=dict(title="Correlation "),
                            hovertemplate="Tasks: %{y} and %{x}<br>Meaning: %{z:.2f}<extra></extra>",
                        )
                    )
                    fig.update_layout(
                        title="Interactive correlation matrix of tasks",
                        xaxis=dict(title=""),
                        yaxis=dict(title="", autorange="reversed"),
                        height=600,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("There is not enough data to build a correlation matrix.")
    
    else:
        # Logic for grouped metrics
        # # Collections with grouped metrics
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
            st.info("There are no collections available with grouped metrics.")
            return
        
        selected_collection = st.selectbox(
            "Select a collection with grouped metrics:",
            options=available_collections,
            key="grouped_metrics_collection_selection",
        )
        
        results_collection = db_client.get_collection(selected_collection)
        results_data = list(results_collection.find())
        
        if results_data:
            visualize_grouped_metrics(results_data, selected_collection)
        else:
            st.info(f"Data in the collection '{selected_collection}' missing.")
