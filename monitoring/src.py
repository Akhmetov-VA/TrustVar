from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

from bson.json_util import loads
import logging
from io import StringIO
import json

logger = logging.getLogger(__name__)


# -------------------------------------
# Auxiliary functions
# -------------------------------------
def load_file(uploaded_file) -> Optional[pd.DataFrame]:
    """
    A universal function for reading the uploaded file (CSV or Excel).
    Returns Data Frame or None in case of an error.
    """
    if uploaded_file is None:
        return None

    filename = uploaded_file.name.lower()
    if filename.endswith(".xlsx"):
        try:
            df = pd.read_excel(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error when reading an Excel file: {e}")
            return None
    else:
        # We believe that this is a CSV
        # We try utf-8, then latin-1
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
            return df
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(uploaded_file, encoding="latin-1")
                return df
            except Exception as e:
                st.error(f"Couldn't read the CSV file: {e}")
                return None
        except Exception as e:
            st.error(f"Error when reading the CSV file: {e}")
            return None


def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all df values so that they are serializable in MongoDB.
    In particular, convert numpy.ndarray to a list, otherwise it will cause Invalid Document.
    """

    def convert_value(x: Any) -> Any:
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    return df.applymap(convert_value)


def load_file_any_format(uploaded_file) -> Optional[pd.DataFrame]:
    """Upload a file in any format: CSV, XLSX, JSON or Parquet."""
    if uploaded_file is None:
        return None
    try:
        if uploaded_file.name.lower().endswith(".json"):
            string_data = uploaded_file.getvalue().decode("utf-8").strip()

            if not string_data:
                st.error("Файл пустой")
                logger.error("Файл пустой")
                return None

            # Попробуем сначала распарсить как обычный JSON (массив или объект)
            try:
                data = json.loads(string_data)

                # Если это один объект — оборачиваем в список
                if isinstance(data, dict):
                    data = [data]

                # Если это не список — ошибка структуры
                if not isinstance(data, list):
                    st.error("JSON должен быть массивом объектов или одним объектом")
                    logger.error(
                        "JSON должен быть массивом объектов или одним объектом"
                    )
                    return None

                df = pd.DataFrame(data)
                return df

            except json.JSONDecodeError:
                # Не получилось распарсить как обычный JSON — пробуем как JSONL
                pass

            # Пробуем как JSONL: читаем через StringIO + lines=True
            try:
                df = pd.read_json(StringIO(string_data), lines=True)
                return df
            except ValueError as e:
                st.error(f"Невозможно распознать формат JSON: {e}")
                logger.error(f"Невозможно распознать формат JSON: {e}")
                return None

        elif uploaded_file.name.lower().endswith(".xlsx"):
            # Loading Excel
            try:
                df = pd.read_excel(uploaded_file)
                return sanitize_df(df)
            except Exception as e:
                st.error(f"Error when reading a Excel file: {e}")
                return None
        elif uploaded_file.name.lower().endswith(".parquet"):
            # Loading Parquet
            try:
                df = pd.read_parquet(uploaded_file)
                return sanitize_df(df)
            except Exception as e:
                st.error(f"Error when reading a Parquet file: {e}")
                return None
        else:
            # We're trying it as a CSV
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
                return sanitize_df(df)
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(uploaded_file, encoding="latin-1")
                    return sanitize_df(df)
                except Exception as e:
                    st.error(f"Couldn't read the CSV file: {e}")
                    return None
            except Exception as e:
                st.error(f"Error when reading the CSV file: {e}")
                return None
    except Exception as e:
        st.error(f"Failed to upload file: {e}")
        logger.error(f"Failed to upload file: {e}")
        return None
