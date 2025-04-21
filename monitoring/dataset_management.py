# dataset_management.py
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from monitoring.src import load_file_any_format
from utils.constants import METRICS
from utils.db_client import MongoDBClient, MongoDBConfig

# Инициализация клиента БД
config = MongoDBConfig(database="TrustGen")
db_client = MongoDBClient(config)


def render_dataset_registry_section():
    st.subheader("Содержимое dataset_regestry")
    coll = db_client.get_collection("dataset_regestry")
    docs = list(coll.find({}))
    if docs:
        df = pd.DataFrame(docs)
        if "_id" in df.columns:
            df.drop(columns=["_id"], inplace=True)
        st.dataframe(df)
    else:
        st.write("dataset_regestry пуст.")


def render_dataset_upload_section() -> Optional[str]:
    with st.expander("Добавить новый датасет", expanded=False):
        st.write("Вы можете загрузить CSV, Excel, JSON или Parquet файл.")
        uploaded_file = st.file_uploader(
            "Загрузите CSV, Excel, JSON или Parquet файл",
            type=["csv", "xlsx", "json", "parquet"],
            key="file_uploader_experiments",
        )
        if uploaded_file is not None:
            dataset_name_input = st.text_input(
                "Введите имя нового датасета (латиницей):",
                value=uploaded_file.name.split(".")[0],
            )
        if uploaded_file is not None and dataset_name_input:
            df_uploaded = load_file_any_format(uploaded_file)
            if df_uploaded is not None and not df_uploaded.empty:
                st.write("Некоторые строки загруженного датасета (случайные 10 строк):")
                st.dataframe(df_uploaded.sample(min(10, len(df_uploaded))))
                chosen_metric = st.selectbox(
                    "Выберите метрику для этого датасета:",
                    METRICS,
                    key="dataset_upload_selectbox",
                )
                st.write(
                    "Выберите колонки, которые будут использоваться как переменные для промпта:"
                )
                var_cols = st.multiselect(
                    "Переменные для промпта:", list(df_uploaded.columns)
                )

                target_column = None
                include_col = None
                exclude_col = None

                if chosen_metric == "include_exclude":
                    st.write(
                        "Для метрики 'include_exclude' необходимо указать:\n"
                        "1) Колонку, где хранится список строк, которые должны присутствовать в ответе.\n"
                        "2) Опционально — колонку, где хранится список строк, которые не должны присутствовать."
                    )
                    potential_cols = [
                        c for c in df_uploaded.columns if c not in var_cols
                    ]
                    include_col = st.selectbox(
                        "Колонка со строками, которые должны присутствовать (include):",
                        potential_cols,
                        key="dataset_upload_include_selectbox",
                    )
                    exclude_col = st.selectbox(
                        "Колонка со строками, которые не должны присутствовать (exclude) (необязательно):",
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
                                "Введите название колонки с таргетом:"
                            )
                        else:
                            target_column = st.selectbox(
                                "Выберите колонку с таргетом:",
                                potential_targets,
                                key="dataset_upload_target_selectbox",
                            )
                st.subheader("Предпросмотр записи для сохранения:")
                record_preview = {
                    "dataset_name": dataset_name_input,
                    "var_cols": var_cols,
                    "metric": chosen_metric,
                    "target_column": target_column,
                    "include_column": include_col,
                    "exclude_column": exclude_col,
                }
                st.json(record_preview)
                if st.button("Сохранить датасет в БД"):
                    db_client.insert_dataset_records(dataset_name_input, df_uploaded)
                    db_client.insert_dataset_into_registry(record_preview)
                    st.success(
                        f"Датасет '{dataset_name_input}' загружен и зарегистрирован!"
                    )
                    return dataset_name_input
            else:
                st.error("Загруженный файл пуст или не может быть прочитан.")
    return None


def render_dataset_management_tab():
    st.header("Управление датасетами")
    render_dataset_registry_section()
    render_dataset_upload_section()


def render_dataset_varcols_section(
    dataset_name: str,
) -> Tuple[
    Optional[List[str]], Optional[str], Optional[str], Optional[str], Optional[str]
]:
    registry_info = db_client.get_dataset_registry_info(dataset_name)
    if not registry_info:
        st.write("Для этого датасета нет сохраненных var_cols, метрики или таргета.")
        return None, None, None, None, None
    else:
        var_cols = registry_info["var_cols"]
        chosen_metric = registry_info.get("metric", METRICS[0])
        target_column = registry_info.get("target_column", None)
        include_column = registry_info.get("include_column", None)
        exclude_column = registry_info.get("exclude_column", None)
        st.write(f"**Переменные для промпта (var_cols):** {var_cols}")
        st.write(f"**Метрика:** {chosen_metric}")
        st.write(f"**Таргет колонка:** {target_column}")
        st.write(f"**Колонка для include:** {include_column}")
        st.write(f"**Колонка для exclude:** {exclude_column}")
        return var_cols, chosen_metric, target_column, include_column, exclude_column
