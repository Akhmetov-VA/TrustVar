import logging
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from utils.constants import MODELS
from utils.db_client import MongoDBClient, MongoDBConfig
from utils.sync_task import sync_task_once

# Configuring logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DATABASE client initialization (configuration is taken from environment variables)
config = MongoDBConfig(database="TrustGen")
db_client = MongoDBClient(config)

DEFAULT_REGEX = r"(?:^\W*([01]).*)|(?:.*([01])\W*$)"


def generate_prompt_hint(var_cols: List[str]) -> Tuple[str, str]:
    placeholders = ", ".join("{" + c + "}" for c in var_cols)
    hint = f"You can use any selected columns in curly brackets.: {placeholders}."
    return hint, placeholders


def display_task_summary(df_tasks: pd.DataFrame):
    total_tasks = len(df_tasks)
    unique_datasets = df_tasks["dataset_name"].nunique()
    unique_metrics = df_tasks["metric"].nunique()
    unique_groups = df_tasks["group"].nunique()
    unique_prompts = df_tasks["prompt"].nunique()

    # If the models column contains lists, we extract all models.
    all_models = [
        model
        for sublist in df_tasks["models"]
        if isinstance(sublist, list)
        for model in sublist
    ]
    unique_models = len(set(all_models))

    rta_count = (
        df_tasks["rta_prompt"].notna().sum() if "rta_prompt" in df_tasks.columns else 0
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total tasks", total_tasks)
    col2.metric("Unique datasets", unique_datasets)
    col3.metric("Unique metrics", unique_metrics)

    col4, col5, col6 = st.columns(3)
    col4.metric("Unique groups", unique_groups)
    col5.metric("Unique promts", unique_prompts)
    col6.metric("Unique models", unique_models)

    col7, _ = st.columns([1, 1])
    col7.metric("RTA promptov is used", rta_count)


def filter_tasks_by_group(df_tasks: pd.DataFrame) -> pd.DataFrame:
    if df_tasks.empty:
        return df_tasks
    groups = df_tasks["group"].unique().tolist()
    if len(groups) > 1:
        selected_group = st.selectbox(
            "Select a group to display:", ["All"] + groups, key=str(uuid.uuid4())
        )
        if selected_group != "All":
            df_tasks = df_tasks[df_tasks["group"] == selected_group]
    return df_tasks


def render_update_task():
    with st.expander("Change Task models", expanded=False):
        st.header("Change Task models")
        df_tasks = db_client.get_all_tasks()
        if df_tasks.empty:
            st.write("There are no tasks to update.")
            return

        df_tasks = filter_tasks_by_group(df_tasks)
        if df_tasks.empty:
            st.write("There are no tasks in the selected group to update.")
            return

        # Selecting an update task
        task_names = df_tasks["task_name"].unique().tolist()
        selected_task_name = st.selectbox(
            "Select a task:", task_names, key="update_task_selectbox"
        )
        task_to_update = df_tasks[df_tasks["task_name"] == selected_task_name].iloc[0]

        # Updating the list of models only
        current_models = task_to_update.get("models", [])
        selected_models = st.multiselect(
            "Select the models for the task:", options=MODELS, default=current_models
        )

        if st.button("Update models"):
            update_data = {"models": selected_models}
            task_id = task_to_update["_id"]
            db_client.update_task(task_id, update_data)
            updated_task = db_client.get_task(task_id)
            sync_task_once(db_client.db, updated_task)
            st.success("The list of models has been successfully updated!")


def highlight_status(s: str) -> str:
    if s == "Error":
        return "background-color: red; color: white;"
    elif s == "In process":
        return "background-color: orange; color: white;"
    elif s == "Completed":
        return "background-color: green; color: white;"
    else:
        return ""


def restart_stopped_error_tasks(collection_name: str) -> int:
    count_stopped = db_client.update_tasks_status(collection_name, "stopped", "pending")
    count_error = db_client.update_tasks_status(collection_name, "error", "pending")
    return count_stopped + count_error


def stop_pending_tasks(collection_name: str) -> int:
    return db_client.update_tasks_status(collection_name, "pending", "stopped")


def load_data_for_dashboard(collections: List[str]) -> pd.DataFrame:
    data = []
    exclude = ["delete_me", "test"]
    collections = [c for c in collections if c not in exclude]

    for collection_name in collections:
        total_tasks = db_client.count_total_tasks(collection_name)
        statuses = [
            "pending",
            "completed",
            "stopped",
            "error",
            "extracted",
            "processing",
            "transfered_to_rta",
        ]
        status_counts = {
            status: db_client.count_tasks_by_status(collection_name, status)
            for status in statuses
        }

        error_count = status_counts["stopped"] + status_counts["error"]
        if error_count > 0:
            overall_status = "Error"
        elif status_counts["pending"] > 0 or status_counts["processing"] > 0:
            overall_status = "In process"
        else:
            overall_status = "Completed"
        data_row = {
            "Collection": collection_name,
            "Total tasks": total_tasks,
            "Waiting": status_counts["pending"],
            "Done": status_counts["completed"]
            + status_counts["transfered_to_rta"],
            "Measured": status_counts["extracted"],
            "With errors": error_count,
            "Status": overall_status,
        }
        data.append(data_row)

    return pd.DataFrame(data)


def show_errors(collections: List[str]):
    st.header("Unique error messages")
    with st.expander("Show errors", expanded=False):
        # Dictionary for storing information about collections with errors
        collections_with_errors = {}

        # Collecting information about issues with errors
        for collection_name in collections:
            stopped_tasks = db_client.get_tasks_by_status(collection_name, "stopped")
            error_tasks = db_client.get_tasks_by_status(collection_name, "error")
            failed_tasks = stopped_tasks + error_tasks

            if failed_tasks:
                collections_with_errors[collection_name] = failed_tasks

                # Displaying errors for the collection
                error_messages = [
                    task.get("error", "There is no information about the error")
                    for task in failed_tasks
                ]
                error_counts = Counter(error_messages)
                st.subheader(f"Collection: {collection_name}")

                # Grouping errors by models
                models_errors = {}
                for task in failed_tasks:
                    model = task.get("model", "Unknown model")
                    error = task.get("error", "There is no information about the error")
                    if model not in models_errors:
                        models_errors[model] = Counter()
                    models_errors[model][error] += 1

                # Displaying errors by model
                for model, errors in models_errors.items():
                    st.write(f"**Model:** {model}")
                    for i, (error_message, count) in enumerate(errors.items()):
                        st.write(
                            f"- **Error:** {error_message} | **Quantity:** {count}"
                        )
                        if i > 5:
                            break
                st.write("---")

        if collections_with_errors:
            # Getting a list of collections with errors
            collections_list = list(collections_with_errors.keys())

            # Adding the "All collections" option"
            options = [None, "All collections"] + collections_list

            # Selecting a collection to restart
            selected_collection = st.selectbox(
                "Select a collection to restart tasks.:",
                options,
                index=0,
                key="collection_to_restart",
            )

            total_restarted = 0

            if selected_collection:
                if selected_collection == "All collections":
                    # Restarting tasks in all collections with errors
                    for collection_name in collections_with_errors.keys():
                        modified_count = restart_stopped_error_tasks(collection_name)
                        if modified_count > 0:
                            total_restarted += modified_count
                            st.write(
                                f"The collection '{collection_name}' has restarted {modified_count} issues."
                            )
                else:
                    # Restarting tasks only in the selected collection
                    modified_count = restart_stopped_error_tasks(selected_collection)
                    if modified_count > 0:
                        total_restarted += modified_count
                        st.write(
                            f"The collection '{selected_collection}' has restarted {modified_count} issues."
                        )

                if total_restarted > 0:
                    st.success(f"Total restarted {total_restarted} issues.")
                else:
                    st.info("No tasks found to restart.")
        else:
            st.info("There are no error-prone tasks to restart.")

def render_progressbar():
    st.header("Queue monitoring")
    if st.checkbox("Download Queue Monitoring", value=False, key="load_monitoring"):
        collections_to_process = sorted(
            [col for col in db_client.list_collections() if col.startswith("queue_")]
        )
        df = load_data_for_dashboard(collections_to_process)

        if df.empty:
            st.info("There is no data to display in queues.")
        else:
            df = df.sort_values("Collection").reset_index(drop=True)
            df_style = df.style.applymap(highlight_status, subset=["Status"])
            st.write(df_style)

            pending_queues = df.loc[df["Waiting"] > 0, "Collection"].tolist()
            if pending_queues:
                selected_queue = st.selectbox(
                    "Select a queue to stop in:",
                    pending_queues,
                    index=None,
                    key="fail_pending_selectbox",
                )
                if selected_queue:
                    st.write(f"Start stopping {selected_queue}")
                    count_stopped = stop_pending_tasks(selected_queue)
                    st.success(
                        f"In the collection '{selected_queue}' stopped {count_stopped} tasks."
                    )

            if df["With errors"].sum() > 0:
                show_errors(collections_to_process)
    else:
        st.info("Click the button above to download queue monitoring..")

    st.header("Viewing collection data")
    collections_to_process = sorted(
        [col for col in db_client.list_collections() if col.startswith("queue_")]
    )
    if collections_to_process:
        selected_collection = st.selectbox(
            "Select a collection",
            collections_to_process,
            key="dashboard_select_collection",
        )
        if selected_collection:
            if st.button(
                "Show collection data", key=f"show_data_{selected_collection}"
            ):
                st.session_state[f"data_loaded_{selected_collection}"] = True

            if st.session_state.get(f"data_loaded_{selected_collection}", False):
                collection = db_client.get_collection(selected_collection)
                data = list(collection.find())
                df_collection = pd.DataFrame(data)
                if not df_collection.empty and "_id" in df_collection.columns:
                    df_collection = df_collection.drop(columns=["_id"])

                if not df_collection.empty and "model" in df_collection.columns:
                    models_in_data = df_collection["model"].unique()
                    filter_models = st.multiselect(
                        "Filter by models",
                        options=models_in_data,
                        default=list(models_in_data),
                        key=f"dashboard_filter_models_{selected_collection}",
                    )
                    df_filtered = df_collection[
                        df_collection["model"].isin(filter_models)
                    ]
                else:
                    df_filtered = df_collection

                st.dataframe(df_filtered)
                if not df_filtered.empty:
                    csv = df_filtered.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download the results in CSV format",
                        data=csv,
                        file_name=f"{selected_collection}_results.csv",
                        mime="text/csv",
                        key=f"download_csv_{selected_collection}",
                    )
    else:
        st.info("There are no collections available to view.")


def render_tasks_visualization_tab():
    st.header("Visualization by task")
    df_tasks = db_client.get_all_tasks()
    df_tasks = filter_tasks_by_group(df_tasks)
    if df_tasks.empty:
        st.write("There are no tasks in the database for the selected group or at all.")
    else:
        display_task_summary(df_tasks)
        st.dataframe(
            df_tasks[
                [
                    "task_name",
                    "dataset_name",
                    "group",
                    "metric",
                    "models",
                    "regexp",
                    "prompt",
                    "rta_prompt",
                ]
            ]
        )
        render_update_task()
        render_progressbar()
