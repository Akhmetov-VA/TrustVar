import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from monitoring.src import load_file
from utils.constants import METRICS, MODELS, RTA_MODEL, STATUSES
from utils.db_client import MongoDBClient, MongoDBConfig

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация клиента БД (один раз для всего приложения)
config = MongoDBConfig(database="TrustGen")
db_client = MongoDBClient(config)

st.set_page_config(page_title="Trust LLM Dashboard", layout="wide")

DEFAULT_REGEX = r"(?:^\W*([01]).*)|(?:.*([01])\W*$)"


def generate_prompt_hint(var_cols: List[str]) -> str:
    """Сгенерировать подсказку для промпта, основанную на var_cols."""
    placeholders = ", ".join("{" + c + "}" for c in var_cols)
    hint = (
        f"Подсказка: Пример промпта: 'изучи текст {placeholders}'\n"
        f"Вы можете использовать любые выбранные колонки в фигурных скобках: {placeholders}."
    )
    return hint


def display_task_summary(df_tasks: pd.DataFrame):
    """Отобразить сводную информацию по задачам."""
    total_tasks = len(df_tasks)
    unique_datasets = df_tasks["dataset_name"].nunique()
    unique_metrics = df_tasks["metric"].nunique()
    unique_groups = df_tasks["group"].nunique()
    unique_prompts = df_tasks["prompt"].nunique()

    all_models = []
    for m in df_tasks["models"]:
        if isinstance(m, list):
            all_models.extend(m)
    unique_models = len(set(all_models))

    rta_count = (
        df_tasks["rta_prompt"].notna().sum() if "rta_prompt" in df_tasks.columns else 0
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Всего задач", total_tasks)
    col2.metric("Уникальных датасетов", unique_datasets)
    col3.metric("Уникальных метрик", unique_metrics)

    col4, col5, col6 = st.columns(3)
    col4.metric("Уникальных групп", unique_groups)
    col5.metric("Уникальных промптов", unique_prompts)
    col6.metric("Уникальных моделей", unique_models)

    col7, _ = st.columns([1, 1])
    col7.metric("Используется RTA промптов", rta_count)


def filter_tasks_by_group(df_tasks: pd.DataFrame) -> pd.DataFrame:
    """Фильтрация задач по группе."""
    if df_tasks.empty:
        return df_tasks
    groups = df_tasks["group"].unique().tolist()
    if len(groups) > 1:
        selected_group = st.selectbox(
            "Выберите группу для отображения:", ["Все"] + groups
        )
        if selected_group != "Все":
            df_tasks = df_tasks[df_tasks["group"] == selected_group]
    return df_tasks


def render_tasks_visualization_tab():
    """Отрисовка вкладки 'Визуализация по задачам'."""
    st.header("Визуализация по задачам")
    df_tasks = db_client.get_all_tasks()

    df_tasks = filter_tasks_by_group(df_tasks)

    if df_tasks.empty:
        st.write("Нет задач в базе для выбранной группы или вообще.")
    else:
        display_task_summary(df_tasks)
        st.dataframe(df_tasks[["task_name", "dataset_name", "group", "metric"]])


def render_dataset_registry_section():
    """Отображает содержимое dataset_regestry (вне экспандера)."""
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


def load_file_any_format(uploaded_file) -> Optional[pd.DataFrame]:
    """Загрузка файла в любом формате: CSV, XLSX или JSON."""
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


def render_dataset_upload_section() -> Optional[str]:
    """Раздел для загрузки нового датасета."""
    with st.expander("Добавить новый датасет", expanded=False):
        st.write("Вы можете загрузить CSV, Excel или JSON файл.")
        uploaded_file = st.file_uploader(
            "Загрузите CSV, Excel или JSON файл",
            type=["csv", "xlsx", "json"],
            key="file_uploader_experiments",
        )
        dataset_name_input = st.text_input("Введите имя нового датасета (латиницей):")

        if uploaded_file is not None and dataset_name_input:
            df_uploaded = load_file_any_format(uploaded_file)
            if df_uploaded is not None and not df_uploaded.empty:
                st.write("Некоторые строки загруженного датасета (случайные 10 строк):")
                st.dataframe(df_uploaded.sample(min(10, len(df_uploaded))))

                # Более лаконичное название для var_cols:
                st.write(
                    "Выберите колонки, которые будут использоваться как переменные для промпта:"
                )
                var_cols = st.multiselect(
                    "Переменные для промпта:", list(df_uploaded.columns)
                )

                chosen_metric = st.selectbox(
                    "Выберите метрику для этого датасета:", METRICS
                )

                target_column = None
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
                            "Выберите колонку с таргетом:", potential_targets
                        )

                if st.button("Сохранить датасет в БД"):
                    db_client.insert_dataset_records(dataset_name_input, df_uploaded)
                    db_client.insert_dataset_into_registry(
                        dataset_name_input, var_cols, chosen_metric, target_column
                    )
                    st.success(
                        f"Датасет '{dataset_name_input}' загружен и зарегистрирован!"
                    )
                    return dataset_name_input
            else:
                st.error("Загруженный файл пуст или не может быть прочитан.")
    return None


def render_dataset_management_tab():
    """Отрисовка вкладки 'Управление датасетами'."""
    st.header("Управление датасетами")
    render_dataset_registry_section()
    render_dataset_upload_section()


def render_dataset_varcols_section(
    dataset_name: str,
) -> Tuple[Optional[List[str]], Optional[str], Optional[str]]:
    """Отображает информацию о var_cols, metric и target для выбранного датасета."""
    registry_info = db_client.get_dataset_registry_info(dataset_name)
    if not registry_info:
        st.write("Для этого датасета нет сохраненных var_cols, метрики или таргета.")
        return None, None, None
    else:
        var_cols = registry_info["var_cols"]
        chosen_metric = registry_info.get("metric", METRICS[0])
        target_column = registry_info.get("target_column", None)
        st.write(f"**Переменные для промпта (var_cols):** {var_cols}")
        st.write(f"**Метрика:** {chosen_metric}")
        st.write(f"**Таргет колонка:** {target_column}")
        return var_cols, chosen_metric, target_column


def show_existing_regexp(metric: str):
    """Показать таблицу с уже существующими регулярками для метрики."""
    coll_name = f"regexp_{metric}"
    if coll_name in db_client.list_collections():
        docs = list(db_client.get_collection(coll_name).find({}))
        if docs:
            df = pd.DataFrame(docs)
            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)
            st.write("Существующие регулярки (name, pattern):")
            st.dataframe(df)
    else:
        st.write("Нет регулярок для данной метрики.")


def render_regexp_section(metric: str) -> Optional[str]:
    """Выбор регулярки."""
    with st.expander("Выбор или создание регулярки для метрики", expanded=False):
        show_existing_regexp(metric)

        use_existing_regexp = st.radio("Регулярка:", ("Существующая", "Своя"))
        selected_regexp = None
        if use_existing_regexp == "Существующая":
            regexps = db_client.get_regexp_docs_for_metric(metric)
            if regexps:
                names = [r["name"] for r in regexps]
                selected_name = st.selectbox("Выберите регулярку по имени:", names)
                for r in regexps:
                    if r["name"] == selected_name:
                        selected_regexp = r["pattern"]
                        break
            else:
                st.write("Нет доступных регулярок для этой метрики.")
        else:
            st.write(f"По умолчанию предлагаем: {DEFAULT_REGEX}")
            custom_regexp = st.text_input(
                "Введите свою регулярку:", value=DEFAULT_REGEX
            )
            if custom_regexp:
                if db_client.validate_regex(custom_regexp):
                    regexp_name = st.text_input("Введите имя для этой регулярки:")
                    if regexp_name and st.button("Добавить регулярку в базу"):
                        db_client.insert_regexp_for_metric(
                            metric, custom_regexp, regexp_name
                        )
                        st.success("Регулярка добавлена!")
                        selected_regexp = custom_regexp
                else:
                    st.error("Неверное регулярное выражение!")
        return selected_regexp


def show_existing_prompts(dataset_name: str):
    """Показать таблицу с уже существующими промптами (name, prompt) для датасета."""
    coll_name = f"prompt_{dataset_name}"
    if coll_name in db_client.list_collections():
        docs = list(db_client.get_prompt_docs_for_dataset(dataset_name))
        if docs:
            df = pd.DataFrame(docs)
            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)
            st.write("Существующие промпты для датасета (name, prompt):")
            st.dataframe(df)
    else:
        st.write("Нет промптов для данного датасета.")


def render_prompt_creation_section(
    dataset_name: str, var_cols: List[str]
) -> Optional[str]:
    """Отображает создание нового промпта."""
    hint = generate_prompt_hint(var_cols)
    st.write(hint)
    prompt_name = st.text_input("Введите имя нового промпта:")
    user_prompt = st.text_area("Введите свой промпт:")
    if user_prompt:
        missing_cols = [c for c in var_cols if f"{{{c}}}" not in user_prompt]
        if missing_cols:
            st.error("Отсутствуют плейсхолдеры: " + ", ".join(missing_cols))
        else:
            if prompt_name and st.button("Добавить промпт в базу"):
                db_client.insert_prompt_for_dataset(
                    dataset_name, user_prompt, prompt_name
                )
                st.success("Промпт добавлен!")
                return user_prompt
    return None


def render_prompt_selection_section(
    dataset_name: str, var_cols: List[str]
) -> Optional[str]:
    """Отображает выбор промпта по имени."""
    with st.expander("Выбор или создание промпта для датасета", expanded=False):
        show_existing_prompts(dataset_name)

        use_existing_prompt = st.radio(
            "Промпт для датасета:", ("Выбрать из базы", "Ввести свой")
        )
        selected_prompt = None
        if use_existing_prompt == "Выбрать из базы":
            prompts = db_client.get_prompt_docs_for_dataset(dataset_name)
            if prompts:
                names = [p["name"] for p in prompts]
                selected_name = st.selectbox("Выберите промпт по имени:", names)
                for p in prompts:
                    if p["name"] == selected_name:
                        selected_prompt = p["prompt"]
                        break
                if selected_prompt:
                    for c in var_cols:
                        if f"{{{c}}}" not in selected_prompt:
                            st.warning(
                                f"В промпте не найден плейсхолдер для колонки {c}"
                            )
            else:
                st.write("Нет доступных промптов. Введите свой.")
        else:
            selected_prompt = render_prompt_creation_section(dataset_name, var_cols)
        return selected_prompt


def show_existing_rta_prompts():
    """Показать таблицу с уже существующими RTA промптами (name, prompt)."""
    coll_name = "prompt_rta"
    if coll_name in db_client.list_collections():
        docs = list(db_client.get_rta_prompt_docs())
        if docs:
            df = pd.DataFrame(docs)
            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)
            st.write("Существующие RTA промпты (name, prompt):")
            st.dataframe(df)
    else:
        st.write("Нет RTA промптов.")


def render_rta_prompt_creation_section() -> Optional[str]:
    """Отображает создание нового RTA промпта."""
    st.write(
        "Для RTA промпта также можно использовать формат с var_cols при необходимости."
    )
    rta_prompt_name = st.text_input("Введите имя нового RTA промпта:")
    rta_user_prompt = st.text_area("Введите RTA промпт:")
    if rta_user_prompt:
        if rta_prompt_name and st.button("Добавить RTA промпт"):
            db_client.insert_rta_prompt(rta_user_prompt, rta_prompt_name)
            st.success("RTA промпт добавлен!")
            return rta_user_prompt
        elif not rta_prompt_name:
            st.error("Пожалуйста, введите имя для RTA промпта.")
    return None


def render_rta_prompt_section() -> Tuple[Optional[str], Optional[str]]:
    """Отображает выбор RTA промпта по имени."""
    with st.expander("Выбор или создание RTA промпта", expanded=False):
        show_existing_rta_prompts()

        st.write("Метрика RtA выбрана. Необходим RTA промпт.")
        use_rta_existing = st.radio("RTA промпт:", ("Выбрать из базы", "Ввести свой"))
        rta_prompt_selected = None
        if use_rta_existing == "Выбрать из базы":
            rta_prompts = db_client.get_rta_prompts()
            if rta_prompts:
                selected_name = st.selectbox(
                    "Выберите RTA промпт по имени:", rta_prompts
                )
                for rp in db_client.get_rta_prompt_docs():
                    if rp["name"] == selected_name:
                        rta_prompt_selected = rp["prompt"]
                        break
            else:
                st.write("Нет доступных RTA промптов.")
        else:
            rta_prompt_selected = render_rta_prompt_creation_section()

        rta_model = st.selectbox(
            "Модель для RTA:",
            MODELS,
            index=MODELS.index(RTA_MODEL) if RTA_MODEL in MODELS else 0,
        )
        return rta_prompt_selected, rta_model


def render_models_section() -> List[str]:
    """Выбор моделей."""
    with st.expander("Выбор моделей для задачи", expanded=False):
        selected_models = st.multiselect("Выберите модели:", MODELS)
        return selected_models


def render_preview_and_save_task(
    dataset_name: str,
    var_cols: List[str],
    selected_prompt: str,
    selected_regexp: str,
    target_column: str,
    selected_models: List[str],
    metric: str,
    rta_prompt_selected: Optional[str],
    rta_model: Optional[str],
):
    """Предпросмотр и сохранение задачи."""
    with st.expander("Предпросмотр и сохранение задачи", expanded=False):
        if (
            selected_prompt
            and selected_regexp
            and selected_models
            and (target_column or metric == "RtA")
        ):
            group_name = st.text_input("Группа задачи (group):", value="default")

            st.subheader("Предпросмотр 5 случайных примеров:")
            df_head = db_client.get_dataset_head(dataset_name, limit=100)
            if not df_head.empty:
                sample_size = min(5, len(df_head))
                preview_data = (
                    df_head[var_cols].sample(n=sample_size).to_dict(orient="records")
                )
                for i, row in enumerate(preview_data):
                    filled_prompt = selected_prompt
                    for k, v in row.items():
                        filled_prompt = filled_prompt.replace(f"{{{k}}}", str(v))
                    st.write(f"**Пример {i+1}:** {filled_prompt}")

            st.write("**Структура записи задачи в БД:**")
            task_data = {
                "task_name": f"task_{dataset_name}_{metric}",
                "dataset_name": dataset_name,
                "prompt": selected_prompt,
                "variables_cols": var_cols,  # те колонки по которым в будущем будет собираться variables внутри очереди
                "models": selected_models,
                "metric": metric,
                "regexp": selected_regexp,
                "group": group_name,
            }
            if metric == "RtA":
                task_data["rta_prompt"] = rta_prompt_selected
                task_data["rta_model"] = rta_model
                task_data["target"] = "RtA"
            else:
                task_data["target"] = target_column

            st.json(task_data, expanded=False)

            if st.button("Загрузить задачу в базу"):
                db_client.insert_task(task_data)
                st.success("Задача успешно добавлена!")


def render_create_task_tab():
    """Отрисовка вкладки 'Создать задачу'."""
    st.header("Создать новую задачу")

    # Выбор датасета в экспандере
    with st.expander("Выбор датасета", expanded=False):
        all_datasets = db_client.get_all_datasets()
        selected_dataset = st.selectbox("Выберите датасет:", all_datasets)
        if selected_dataset:
            # Отфильтровать dataset_regestry
            var_cols, metric, target_column = render_dataset_varcols_section(
                selected_dataset
            )
    if var_cols and metric is not None:
        selected_regexp = render_regexp_section(metric)
        if selected_regexp:
            selected_prompt = render_prompt_selection_section(
                selected_dataset, var_cols
            )
            if selected_prompt:
                rta_prompt_selected = None
                rta_model = None
                if metric == "RtA":
                    rta_prompt_selected, rta_model = render_rta_prompt_section()
                selected_models = render_models_section()

                final_target = "RtA" if metric == "RtA" else target_column

                render_preview_and_save_task(
                    selected_dataset,
                    var_cols,
                    selected_prompt,
                    selected_regexp,
                    final_target,
                    selected_models,
                    metric,
                    rta_prompt_selected,
                    rta_model,
                )


def render_update_task_tab():
    """Отрисовка вкладки 'Обновить задачу'."""
    st.header("Обновить задачу")
    df_tasks = db_client.get_all_tasks()
    if df_tasks.empty:
        st.write("Нет задач для обновления.")
        return

    df_tasks = filter_tasks_by_group(df_tasks)
    if df_tasks.empty:
        st.write("Нет задач в выбранной группе для обновления.")
        return

    task_names = df_tasks["task_name"].unique().tolist()
    selected_task_name = st.selectbox("Выберите задачу:", task_names)
    task_to_update = df_tasks[df_tasks["task_name"] == selected_task_name].iloc[0]

    metrics = METRICS
    current_metric = task_to_update.get("metric", metrics[0])
    if current_metric not in metrics:
        current_metric = metrics[0]

    new_task_name = st.text_input(
        "Название задачи:", value=task_to_update.get("task_name", "")
    )
    new_metric = st.selectbox("Метрика:", metrics, index=metrics.index(current_metric))
    new_group = st.text_input("Группа:", value=task_to_update.get("group", ""))
    current_status = task_to_update.get("status", "pending")
    if current_status not in STATUSES:
        current_status = "pending"
    new_status = st.selectbox("Статус:", STATUSES, index=STATUSES.index(current_status))

    if st.button("Обновить задачу"):
        update_data = {
            "task_name": new_task_name,
            "metric": new_metric,
            "group": new_group,
            "status": new_status,
        }
        task_id = task_to_update["_id"]
        db_client.update_task(task_id, update_data)
        st.success("Задача обновлена!")
        st.experimental_rerun()


# -------------------------------------
# Основной интерфейс
# -------------------------------------
tabs = st.tabs(
    [
        "Визуализация по задачам",
        "Управление датасетами",
        "Создать задачу",
        "Обновить задачу",
    ]
)

with tabs[0]:
    render_tasks_visualization_tab()

with tabs[1]:
    render_dataset_management_tab()

with tabs[2]:
    render_create_task_tab()

with tabs[3]:
    render_update_task_tab()
