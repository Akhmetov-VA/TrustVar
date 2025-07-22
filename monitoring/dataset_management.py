# dataset_management.py
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from monitoring.src import load_file_any_format
from utils.constants import METRICS
from utils.db_client import MongoDBClient, MongoDBConfig

# Initializing the database client
config = MongoDBConfig(database="TrustGen")
db_client = MongoDBClient(config)


def render_dataset_registry_section():
    st.subheader("Content dataset_regestry")
    coll = db_client.get_collection("dataset_regestry")
    docs = list(coll.find({}))
    if docs:
        df = pd.DataFrame(docs)
        if "_id" in df.columns:
            df.drop(columns=["_id"], inplace=True)
        st.dataframe(df)
    else:
        st.write("dataset_regestry empty.")


def render_dataset_upload_section() -> Optional[str]:
    with st.expander("Add a new dataset", expanded=False):
        st.write("You can upload a CSV, Excel, JSON, or Parquet file.")
        uploaded_file = st.file_uploader(
            "Upload a CSV, Excel, JSON, or Parquet file",
            type=["csv", "xlsx", "json", "parquet"],
            key="file_uploader_experiments",
        )
        if uploaded_file is not None:
            dataset_name_input = st.text_input(
                "Enter the name of the new dataset (in Latin):",
                value=uploaded_file.name.split(".")[0],
            )
        if uploaded_file is not None and dataset_name_input:
            df_uploaded = load_file_any_format(uploaded_file)
            if df_uploaded is not None and not df_uploaded.empty:
                st.write("Some lines of the uploaded dataset (random 10 lines):")
                st.dataframe(df_uploaded.sample(min(10, len(df_uploaded))))
                chosen_metric = st.selectbox(
                    "Select a metric for this dataset:",
                    METRICS,
                    key="dataset_upload_selectbox",
                )
                st.write(
                    "Select the columns that will be used as variables for prompt:"
                )
                var_cols = st.multiselect(
                    "Variables for prompt:", list(df_uploaded.columns)
                )

                target_column = None
                include_col = None
                exclude_col = None

                if chosen_metric == "include_exclude":
                    st.write(
                        "For the 'include_exclude' metric, you must specify:\n"
                        "1) The column where the list of rows that should be present in the response is stored.\n"
                        "2) Optionally, a column where a list of rows that should not be present is stored."
                    )
                    potential_cols = [
                        c for c in df_uploaded.columns if c not in var_cols
                    ]
                    include_col = st.selectbox(
                        "A column with rows that should be present(include):",
                        potential_cols,
                        key="dataset_upload_include_selectbox",
                    )
                    exclude_col = st.selectbox(
                        "A column with rows that should not be present (exclude) (optional):",
                        [None] + potential_cols,
                        index=0,
                        key="dataset_upload_exclude_selectbox",
                    )
                else:
                    if chosen_metric != "RtA":
                        potential_targets = [
                            c for c in df_uploaded.columns if c not in var_cols
                        ]
                        if not potential_targets:
                            target_column = st.text_input(
                                "Enter the name of the target column:"
                            )
                        else:
                            target_column = st.selectbox(
                                "Select a column with a target:",
                                potential_targets,
                                key="dataset_upload_target_selectbox",
                            )
                st.subheader("Preview the recording to save:")
                record_preview = {
                    "dataset_name": dataset_name_input,
                    "var_cols": var_cols,
                    "metric": chosen_metric,
                    "target_column": target_column,
                    "include_column": include_col,
                    "exclude_column": exclude_col,
                }
                st.json(record_preview)
                if st.button("Save the dataset to the database"):
                    db_client.insert_dataset_records(dataset_name_input, df_uploaded)
                    db_client.insert_dataset_into_registry(record_preview)
                    st.success(
                        f"Dataset '{dataset_name_input}' uploaded and registered!"
                    )
                    return dataset_name_input
            else:
                st.error("The uploaded file is empty or cannot be read.")
    return None


def render_dataset_management_tab():
    st.header("Managing datasets")
    render_dataset_registry_section()
    render_dataset_upload_section()


def render_dataset_varcols_section(
    dataset_name: str,
) -> Tuple[
    Optional[List[str]], Optional[str], Optional[str], Optional[str], Optional[str]
]:
    registry_info = db_client.get_dataset_registry_info(dataset_name)
    if not registry_info:
        st.write("There are no saved var_cols, metrics, or target for this dataset.")
        return None, None, None, None, None
    else:
        var_cols = registry_info["var_cols"]
        chosen_metric = registry_info.get("metric", METRICS[0])
        target_column = registry_info.get("target_column", None)
        include_column = registry_info.get("include_column", None)
        exclude_column = registry_info.get("exclude_column", None)
        st.write(f"**Variables for prompt (var_cols):** {var_cols}")
        st.write(f"**Metric:** {chosen_metric}")
        st.write(f"**Target column:** {target_column}")
        st.write(f"**The column for include:** {include_column}")
        st.write(f"**The column for exclude:** {exclude_column}")
        return var_cols, chosen_metric, target_column, include_column, exclude_column
