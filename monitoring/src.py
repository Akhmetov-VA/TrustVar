from typing import Optional

import pandas as pd
import streamlit as st


# -------------------------------------
# Вспомогательные функции
# -------------------------------------
def load_file(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Универсальная функция для чтения загруженного файла (CSV или Excel).
    Возвращает DataFrame или None в случае ошибки.
    """
    if uploaded_file is None:
        return None

    filename = uploaded_file.name.lower()
    if filename.endswith(".xlsx"):
        try:
            df = pd.read_excel(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Ошибка при чтении Excel файла: {e}")
            return None
    else:
        # Считаем, что это CSV
        # Пробуем utf-8, затем latin-1
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
            return df
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(uploaded_file, encoding="latin-1")
                return df
            except Exception as e:
                st.error(f"Не удалось прочесть CSV файл: {e}")
                return None
        except Exception as e:
            st.error(f"Ошибка при чтении CSV файла: {e}")
            return None


def load_file_any_format(uploaded_file) -> Optional[pd.DataFrame]:
    """Загрузка файла в любом формате: CSV, XLSX, JSON или Parquet."""
    if uploaded_file is None:
        return None
    try:
        if uploaded_file.name.lower().endswith(".json"):
            # Загрузка JSON
            try:
                df = pd.read_json(uploaded_file)
                return df
            except ValueError as e:
                st.error(f"Ошибка при чтении JSON файла: {e}")
                return None
        elif uploaded_file.name.lower().endswith(".xlsx"):
            # Загрузка Excel
            try:
                df = pd.read_excel(uploaded_file)
                return df
            except Exception as e:
                st.error(f"Ошибка при чтении Excel файла: {e}")
                return None
        elif uploaded_file.name.lower().endswith(".parquet"):
            # Загрузка Parquet
            try:
                df = pd.read_parquet(uploaded_file)
                return df
            except Exception as e:
                st.error(f"Ошибка при чтении Parquet файла: {e}")
                return None
        else:
            # Пытаемся как CSV
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
                return df
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(uploaded_file, encoding="latin-1")
                    return df
                except Exception as e:
                    st.error(f"Не удалось прочитать CSV файл: {e}")
                    return None
            except Exception as e:
                st.error(f"Ошибка при чтении CSV файла: {e}")
                return None
    except Exception as e:
        st.error(f"Не удалось загрузить файл: {e}")
        return None
