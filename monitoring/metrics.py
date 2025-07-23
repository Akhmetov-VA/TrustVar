import json
import logging
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # 🔹 For an interactive heatmap
import streamlit as st
import numpy as np
from plotly.subplots import make_subplots
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import iqr

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
    return (np.std(values) / mean_val) * 100


def calculate_corrected_cv(values: List[float]) -> float:
    """Calculates the corrected coefficient of variation for small sample sizes."""
    if not values or len(values) < 2:
        return np.nan
    
    arr = np.array(values)
    n = arr.size
    mean_val = arr.mean()
    std_val = arr.std(ddof=0)  # population standard deviation
    
    if mean_val == 0:
        return np.nan
    
    cv = std_val / mean_val
    # Everitt's correction for small sample bias
    return (1 + 1/(4*n)) * cv * 100


def calculate_iqr_cv(values: List[float]) -> float:
    """Calculates IQR-based coefficient of variation: (Q3-Q1) / midhinge."""
    if not values or len(values) < 2:
        return np.nan
    
    arr = np.array(values)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    midhinge = (q1 + q3) / 2
    
    if midhinge == 0:
        return np.nan
    
    return ((q3 - q1) / midhinge) * 100


def calculate_jsd_divergence(values: List[float]) -> float:
    """Calculates Jensen-Shannon Divergence for measuring distribution heterogeneity."""
    if not values or len(values) < 2:
        return np.nan
    
    arr = np.array(values)
    # Normalize to create probability distribution
    arr_sum = arr.sum()
    if arr_sum == 0:
        return np.nan
    
    P = arr / arr_sum
    # Calculate mean distribution
    P_mean = P.mean()
    
    # Calculate JSD
    jsd = jensenshannon(P, [P_mean] * len(P)) ** 2
    return jsd * 100  # Scale for better visualization


def calculate_confidence_interval(values: List[float], confidence: float = 0.95) -> tuple:
    """Calculate confidence interval for a list of values."""
    if not values or len(values) < 2:
        return np.nan, np.nan
    
    mean_val = np.mean(values)
    std_err = np.std(values) / np.sqrt(len(values))
    
    # Using t-distribution for small samples, normal for large samples
    if len(values) < 30:
        t_value = stats.t.ppf((1 + confidence) / 2, len(values) - 1)
        margin = t_value * std_err
    else:
        # For large samples, use normal distribution
        z_value = stats.norm.ppf((1 + confidence) / 2)
        margin = z_value * std_err
    
    return mean_val - margin, mean_val + margin


def compute_dispersion_indices(values: List[float]) -> Dict[str, float]:
    """Compute all dispersion indices for a set of values."""
    return {
        "cv": calculate_coefficient_of_variation(values),
        "cv_corrected": calculate_corrected_cv(values),
        "iqr_cv": calculate_iqr_cv(values),
        "jsd": calculate_jsd_divergence(values)
    }


def visualize_task_centric_metrics(results_data: List[Dict[str, Any]], collection_name: str):
    """Task-centric visualization of metrics for Compare model behaviour tasks."""
    st.write(f"Starting visualize_task_centric_metrics for {collection_name}")
    st.write(f"Number of documents: {len(results_data)}")
    if results_data:
        st.write(f"First document keys: {list(results_data[0].keys())}")
    
    results_df = pd.DataFrame(results_data)
    if "_id" in results_df.columns:
        results_df = results_df.drop(columns=["_id"])
    
    required_cols = {"task_name", "model", "value", "task_type", "dynamic_augments"}
    if not required_cols.issubset(results_df.columns):
        st.error("The required fields for grouped metrics are missing in the data.")
        return

    # Filter only "Compare model behaviour" tasks
    compare_df = results_df[results_df["task_type"] == "Compare model behaviour"].copy()
    
    if compare_df.empty:
        st.info("There is no data for tasks like 'Compare model behaviour'.")
        return

    # Process dynamic_augments - each record already contains one augmentation
    expanded_rows = []
    for _, row in compare_df.iterrows():
        dynamic_augments = row["dynamic_augments"]
        # Handle both list and single string cases
        if isinstance(dynamic_augments, list):
            if len(dynamic_augments) == 1:
                # Single augmentation in list
                augment = dynamic_augments[0]
            else:
                # Multiple augmentations - this shouldn't happen in current data format
                continue
        else:
            # Single string augmentation
            augment = str(dynamic_augments)
        
        new_row = row.copy()
        new_row["augment"] = augment
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
        key=f"task_centric_tasks_{collection_name}",
    )
    selected_models = st.multiselect(
        "Select models:",
        options=models,
        default=list(models),
        key=f"task_centric_models_{collection_name}",
    )
    selected_augments = st.multiselect(
        "Choose Augmentation:",
        options=augments,
        default=list(augments),
        key=f"task_centric_augments_{collection_name}",
    )
    
    filtered_df = expanded_df[
        (expanded_df["task_name"].isin(selected_tasks))
        & (expanded_df["model"].isin(selected_models))
        & (expanded_df["augment"].isin(selected_augments))
    ]
    
    if filtered_df.empty:
        st.info("There is no data to display with the selected filters.")
        return

    st.subheader("Task-Centric Analysis: Augmentation Impact on Task Stability")
    
    # 1. Task Stability Radar Chart with Multiple Metrics
    st.subheader("1. Task Stability Radar Chart (Multiple Metrics)")
    
    # Calculate stability metrics for each task
    task_stability_data = []
    for task in selected_tasks:
        task_data = filtered_df[filtered_df["task_name"] == task]
        if not task_data.empty:
            # Calculate dispersion indices for each augmentation
            augment_metrics = {}
            for augment in selected_augments:
                augment_data = task_data[task_data["augment"] == augment]
                if not augment_data.empty:
                    values = augment_data["value"].tolist()
                    dispersion_indices = compute_dispersion_indices(values)
                    augment_metrics[augment] = dispersion_indices
            
            task_stability_data.append({
                "task_name": task,
                "augment_metrics": augment_metrics
            })
    
    if task_stability_data:
        # Create radar charts for different metrics
        metric_options = ["cv", "cv_corrected", "iqr_cv", "jsd"]
        selected_metric = st.selectbox(
            "Select dispersion metric for radar chart:",
            options=metric_options,
            index=0,
            key=f"radar_metric_{collection_name}"
        )
        
        fig_radar = go.Figure()
        
        for task_info in task_stability_data:
            task_name = task_info["task_name"]
            augment_metrics = task_info["augment_metrics"]
            
            if augment_metrics:
                augment_names = list(augment_metrics.keys())
                metric_values = [augment_metrics[aug][selected_metric] for aug in augment_names]
                
                # Filter out NaN values
                valid_indices = [i for i, v in enumerate(metric_values) if not np.isnan(v)]
                if valid_indices:
                    valid_augments = [augment_names[i] for i in valid_indices]
                    valid_values = [metric_values[i] for i in valid_indices]
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=valid_values,
                        theta=valid_augments,
                        fill='toself',
                        name=task_name,
                        line=dict(width=2)
                    ))
        
        # Calculate max value for proper scaling
        max_val = 0
        for task_info in task_stability_data:
            if task_info["augment_metrics"]:
                for augment_metrics in task_info["augment_metrics"].values():
                    val = augment_metrics.get(selected_metric, 0)
                    if not np.isnan(val):
                        max_val = max(max_val, val)
        
        metric_names = {
            "cv": "Coefficient of Variation",
            "cv_corrected": "Corrected CV",
            "iqr_cv": "IQR-based CV", 
            "jsd": "Jensen-Shannon Divergence"
        }
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max_val * 1.1 if max_val > 0 else 100]
                )),
            showlegend=True,
            title=f"Task Stability: {metric_names[selected_metric]} (Lower = More Stable)",
            height=600
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # 2. Task Performance Heatmap
    st.subheader("2. Task Performance Heatmap")
    
    # Calculate mean performance for each task-augmentation combination
    performance_data = []
    for task in selected_tasks:
        for augment in selected_augments:
            task_augment_data = filtered_df[
                (filtered_df["task_name"] == task) & 
                (filtered_df["augment"] == augment)
            ]
            if not task_augment_data.empty:
                mean_performance = task_augment_data["value"].mean()
                performance_data.append({
                    "task_name": task,
                    "augment": augment,
                    "mean_performance": mean_performance
                })
    
    if performance_data:
        perf_df = pd.DataFrame(performance_data)
        perf_pivot = perf_df.pivot_table(
            index="task_name", 
            columns="augment", 
            values="mean_performance"
        )
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=perf_pivot.values,
            x=perf_pivot.columns,
            y=perf_pivot.index,
            colorscale="RdYlGn",
            colorbar=dict(title=f"Performance ({collection_name})"),
            hovertemplate="Task: %{y}<br>Augmentation: %{x}<br>Performance: %{z:.3f}<extra></extra>",
        ))
        fig_heatmap.update_layout(
            title="Task Performance Across Augmentations",
            xaxis=dict(title="Augmentation"),
            yaxis=dict(title="Task"),
            height=500
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # 3. Confidence Intervals for Task Performance
    st.subheader("3. Confidence Intervals for Task Performance")
    
    confidence_data = []
    for task in selected_tasks:
        task_data = filtered_df[filtered_df["task_name"] == task]
        if not task_data.empty:
            for augment in selected_augments:
                augment_data = task_data[task_data["augment"] == augment]
                if not augment_data.empty:
                    values = augment_data["value"].tolist()
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    ci_lower, ci_upper = calculate_confidence_interval(values)
                    
                    confidence_data.append({
                        "task_name": task,
                        "augment": augment,
                        "mean": mean_val,
                        "std": std_val,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "sample_size": len(values)
                    })
    
    if confidence_data:
        conf_df = pd.DataFrame(confidence_data)
        
        # Create confidence interval plot
        fig_ci = px.scatter(
            conf_df,
            x="augment",
            y="mean",
            color="task_name",
            error_y="std",
            title="Task Performance with Confidence Intervals",
            labels={"mean": f"Performance ({collection_name})", "augment": "Augmentation"}
        )
        fig_ci.update_layout(height=500)
        st.plotly_chart(fig_ci, use_container_width=True)
        
        # Display confidence interval table
        st.subheader("Confidence Interval Details")
        ci_display_df = conf_df[["task_name", "augment", "mean", "ci_lower", "ci_upper", "sample_size"]].round(3)
        st.dataframe(ci_display_df)
    
    # 4. Histograms with Error Bars
    st.subheader("4. Performance Distribution with Error Bars")
    
    # Create subplots for each task
    if len(selected_tasks) > 0:
        fig_hist = make_subplots(
            rows=len(selected_tasks), 
            cols=1,
            subplot_titles=selected_tasks,
            vertical_spacing=0.1
        )
        
        traces_added = False
        for i, task in enumerate(selected_tasks, 1):
            task_data = filtered_df[filtered_df["task_name"] == task]
            if not task_data.empty:
                for augment in selected_augments:
                    augment_data = task_data[task_data["augment"] == augment]
                    if not augment_data.empty:
                        values = augment_data["value"].tolist()
                        if values:  # Only add trace if we have values
                            fig_hist.add_trace(
                                go.Histogram(
                                    x=values,
                                    name=f"{augment}",
                                    opacity=0.7,
                                    nbinsx=10
                                ),
                                row=i, col=1
                            )
                            traces_added = True
        
        if traces_added:
            fig_hist.update_layout(
                title="Performance Distribution by Task and Augmentation",
                height=300 * len(selected_tasks),
                showlegend=True
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No data available for histogram visualization.")
    
    # 5. Comprehensive Task Stability Summary
    st.subheader("5. Comprehensive Task Stability Summary")
    
    stability_summary = []
    for task in selected_tasks:
        task_data = filtered_df[filtered_df["task_name"] == task]
        if not task_data.empty:
            # Calculate overall stability metrics
            all_values = task_data["value"].tolist()
            overall_indices = compute_dispersion_indices(all_values)
            overall_mean = np.mean(all_values)
            overall_std = np.std(all_values)
            
            # Calculate stability by augmentation
            augment_stability = {}
            for augment in selected_augments:
                augment_data = task_data[task_data["augment"] == augment]
                if not augment_data.empty:
                    augment_values = augment_data["value"].tolist()
                    augment_indices = compute_dispersion_indices(augment_values)
                    augment_stability[augment] = augment_indices
            
            stability_summary.append({
                "task_name": task,
                "overall_indices": overall_indices,
                "overall_mean": overall_mean,
                "overall_std": overall_std,
                "augment_stability": augment_stability
            })
    
    if stability_summary:
        # Create comprehensive stability summary table
        summary_data = []
        for summary in stability_summary:
            row = {
                "Task": summary["task_name"],
                "Overall CV (%)": summary["overall_indices"]["cv"],
                "Corrected CV (%)": summary["overall_indices"]["cv_corrected"],
                "IQR-CV (%)": summary["overall_indices"]["iqr_cv"],
                "JSD": summary["overall_indices"]["jsd"],
                "Overall Mean": summary["overall_mean"],
                "Overall Std": summary["overall_std"]
            }
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df.round(3))
        
        # Stability ranking by different metrics
        st.subheader("Task Stability Ranking")
        
        metric_ranking_options = ["cv", "cv_corrected", "iqr_cv", "jsd"]
        selected_ranking_metric = st.selectbox(
            "Select metric for ranking:",
            options=metric_ranking_options,
            index=0,
            key=f"ranking_metric_{collection_name}"
        )
        
        metric_display_names = {
            "cv": "CV",
            "cv_corrected": "Corrected CV", 
            "iqr_cv": "IQR-CV",
            "jsd": "JSD"
        }
        
        # Sort by selected metric (lower is better for CV metrics, higher for JSD)
        if selected_ranking_metric == "jsd":
            stability_ranking = sorted(stability_summary, key=lambda x: x["overall_indices"][selected_ranking_metric], reverse=True)
        else:
            stability_ranking = sorted(stability_summary, key=lambda x: x["overall_indices"][selected_ranking_metric])
        
        for i, summary in enumerate(stability_ranking, 1):
            metric_value = summary["overall_indices"][selected_ranking_metric]
            
            if selected_ranking_metric in ["cv", "cv_corrected", "iqr_cv"]:
                stability_level = "Very Stable" if metric_value < 10 else \
                                "Stable" if metric_value < 20 else \
                                "Moderately Stable" if metric_value < 30 else "Unstable"
            else:  # JSD
                stability_level = "Very Stable" if metric_value < 0.1 else \
                                "Stable" if metric_value < 0.2 else \
                                "Moderately Stable" if metric_value < 0.3 else "Unstable"
            
            st.write(f"{i}. **{summary['task_name']}**: {metric_display_names[selected_ranking_metric]} = {metric_value:.1f} ({stability_level})")
        
        # Display interpretation guide
        with st.expander("Interpretation Guide"):
            st.write("**Coefficient of Variation (CV):**")
            st.write("- CV < 10%: Very stable task")
            st.write("- CV 10-20%: Stable task")
            st.write("- CV 20-30%: Moderately stable task")
            st.write("- CV > 30%: Unstable task")
            
            st.write("**Corrected CV:**")
            st.write("- Adjusted for small sample bias using Everitt's correction")
            st.write("- More accurate for small datasets")
            
            st.write("**IQR-CV:**")
            st.write("- Based on interquartile range, robust to outliers")
            st.write("- Good for non-normal distributions")
            
            st.write("**Jensen-Shannon Divergence (JSD):**")
            st.write("- Measures distribution heterogeneity")
            st.write("- Lower values indicate more uniform performance across models")
            st.write("- Higher values indicate more diverse model performance")


def visualize_grouped_metrics(results_data: List[Dict[str, Any]], collection_name: str):
    """Visualization of metrics by groups task_type and dynamic_augments."""
    st.write(f"Starting visualize_grouped_metrics for {collection_name}")
    st.write(f"Number of documents: {len(results_data)}")
    if results_data:
        st.write(f"First document keys: {list(results_data[0].keys())}")
    
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

    # Process dynamic_augments - each record already contains one augmentation
    expanded_rows = []
    for _, row in compare_df.iterrows():
        dynamic_augments = row["dynamic_augments"]
        # Handle both list and single string cases
        if isinstance(dynamic_augments, list):
            if len(dynamic_augments) == 1:
                # Single augmentation in list
                augment = dynamic_augments[0]
            else:
                # Multiple augmentations - this shouldn't happen in current data format
                continue
        else:
            # Single string augmentation
            augment = str(dynamic_augments)
        
        new_row = row.copy()
        new_row["augment"] = augment
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
    
    radar_data = filtered_df[filtered_df["model"] == selected_model_for_radar]
    
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
                    
                    fig_radar = go.Figure()
                    fig_radar.add_trace(
                        go.Scatterpolar(
                            r=task_pivot["value"].tolist(),
                            theta=augment_names,
                            fill="toself",
                            name=f"Task: {task}",
                        )
                    )
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, task_pivot["value"].max() * 1.1],
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
    
    # Initialize database client
    from utils.db_client import MongoDBClient, MongoDBConfig
    db_client = MongoDBClient(MongoDBConfig(database="TrustGen"))
    
    # Switch between metric types
    metric_type = st.radio(
        "Select the type of metric analysis:",
        ["Common metrics", "Group analysis (Model-centric)", "Task-centric analysis (Compare model behaviour)"],
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
    
    elif metric_type == "Group analysis (Model-centric)":
        # Logic for grouped metrics (model-centric view)
        grouped_collections = ["Accuracy_Groups", "TFNR_Groups"]
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
    
    else:  # Task-centric analysis
        # Logic for task-centric analysis
        grouped_collections = ["Accuracy_Groups", "TFNR_Groups"]
        available_collections = []
        
        for coll in grouped_collections:
            try:
                collection = db_client.get_collection(coll)
                if collection.count_documents({}) > 0:
                    available_collections.append(coll)
            except Exception as e:
                st.write(f"Error: {coll}")
                st.error(f"Exception details: {str(e)}")
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")
                continue
        
        if not available_collections:
            st.info("There are no collections available with grouped metrics for task-centric analysis.")
            return
        
        selected_collection = st.selectbox(
            "Select a collection with grouped metrics for task analysis:",
            options=available_collections,
            key="task_centric_collection_selection",
        )
        
        results_collection = db_client.get_collection(selected_collection)
        results_data = list(results_collection.find())
        
        if results_data:
            visualize_task_centric_metrics(results_data, selected_collection)
        else:
            st.info(f"Data in the collection '{selected_collection}' missing.")
