import logging
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
import pandas as pd
import streamlit as st

from monitoring.src import load_file_any_format
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


def generate_prompt_hint(var_cols: List[str]) -> Tuple[str, str]:
    """Сгенерировать подсказку для промпта, основанную на var_cols."""
    placeholders = ", ".join("{" + c + "}" for c in var_cols)
    hint = f"Вы можете использовать любые выбранные колонки в фигурных скобках: {placeholders}."
    return hint, placeholders


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
        

        st.dataframe(df_tasks[["task_name", "dataset_name", "group", "metric", "models"]])
        render_update_task()
        
        render_progressbar()


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


def render_dataset_upload_section() -> Optional[str]:
    """Раздел для загрузки нового датасета."""
    with st.expander("Добавить новый датасет", expanded=False):
        st.write("Вы можете загрузить CSV, Excel или JSON файл.")
        uploaded_file = st.file_uploader(
            "Загрузите CSV, Excel, JSON или Parquet файл",
            type=["csv", "xlsx", "json", "parquet"],
            key="file_uploader_experiments",
        )
        dataset_name_input = st.text_input("Введите имя нового датасета (латиницей):")

        if uploaded_file is not None and dataset_name_input:
            df_uploaded = load_file_any_format(uploaded_file)
            if df_uploaded is not None and not df_uploaded.empty:
                st.write("Некоторые строки загруженного датасета (случайные 10 строк):")
                st.dataframe(df_uploaded.sample(min(10, len(df_uploaded))))

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


def get_all_prompts() -> List[Dict[str, Any]]:
    coll = db_client.get_collection("prompt_storage")
    return list(coll.find({}))


def prompt_exists(name: str) -> bool:
    coll = db_client.get_collection("prompt_storage")
    return coll.find_one({"name": name}) is not None


def insert_prompt_global(name: str, prompt: str):
    coll = db_client.get_collection("prompt_storage")
    coll.insert_one({"name": name, "prompt": prompt})


def get_all_regexps_for_metric(metric: str) -> List[Dict[str, Any]]:
    coll = db_client.get_collection("regexp_storage")
    return list(coll.find({"metric": metric}))


def insert_regexp_global(name: str, pattern: str, metric: str):
    coll = db_client.get_collection("regexp_storage")
    coll.insert_one({"name": name, "pattern": pattern, "metric": metric})


def show_existing_regexp(metric: str):
    docs = get_all_regexps_for_metric(metric)
    if docs:
        df = pd.DataFrame(docs)
        if "_id" in df.columns:
            df.drop(columns=["_id"], inplace=True)
        st.write("Существующие регулярки (name, pattern, metric):")
        st.dataframe(df)
    else:
        st.write("Нет регулярок для данной метрики.")


def render_regexp_section(metric: str) -> Optional[str]:
    with st.expander("Выбор или создание регулярки для метрики", expanded=False):
        show_existing_regexp(metric)

        use_existing_regexp = st.radio("Регулярка:", ("Существующая", "Своя"))
        selected_regexp = None
        if use_existing_regexp == "Существующая":
            regexps = get_all_regexps_for_metric(metric)
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
                        insert_regexp_global(regexp_name, custom_regexp, metric)
                        st.success("Регулярка добавлена!")
                        selected_regexp = custom_regexp
                else:
                    st.error("Неверное регулярное выражение!")
        return selected_regexp


def show_all_prompts():
    prompts = get_all_prompts()
    if prompts:
        df = pd.DataFrame(prompts)
        if "_id" in df.columns:
            df.drop(columns=["_id"], inplace=True)
        st.write("Существующие промпты (name, prompt):")
        st.dataframe(df)
    else:
        st.write("Нет промптов в хранилище.")


def render_prompt_creation_section(var_cols: List[str]) -> Optional[str]:
    hint, placeholders = generate_prompt_hint(var_cols)
    st.write(hint)
    prompt_name = st.text_input("Введите имя нового промпта:")
    user_prompt = st.text_area("Введите свой промпт:", value=placeholders)
    if user_prompt and prompt_name:
        missing_cols = [c for c in var_cols if f"{{{c}}}" not in user_prompt]
        if missing_cols:
            st.error("Отсутствуют плейсхолдеры: " + ", ".join(missing_cols))
        else:
            if prompt_exists(prompt_name):
                st.warning(
                    f"Промпт с именем '{prompt_name}' уже существует. Вы можете использовать его."
                )
                if st.button("Использовать существующий промпт"):
                    existing_prompts = get_all_prompts()
                    for p in existing_prompts:
                        if p["name"] == prompt_name:
                            return p["prompt"]
            else:
                if st.button("Добавить промпт в базу"):
                    insert_prompt_global(prompt_name, user_prompt)
                    st.success("Промпт добавлен!")
                    return user_prompt
    return None


def render_prompt_selection_section(var_cols: List[str]) -> Optional[str]:
    with st.expander("Выбор или создание промпта", expanded=False):
        show_all_prompts()

        use_existing_prompt = st.radio("Промпт:", ("Выбрать из базы", "Ввести свой"))
        selected_prompt = None
        all_prompts = get_all_prompts()
        if use_existing_prompt == "Выбрать из базы":
            if all_prompts:
                names = [p["name"] for p in all_prompts]
                selected_name = st.selectbox("Выберите промпт по имени:", names)
                for p in all_prompts:
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
            selected_prompt = render_prompt_creation_section(var_cols)
        return selected_prompt


def show_all_rta_prompts():
    # Для упрощения считаем, что RTA промпты также находятся в prompt_storage
    show_all_prompts()


def render_rta_prompt_section() -> Tuple[Optional[str], Optional[str], Any]:
    with st.expander("Выбор или создание RTA промпта", expanded=False):
        show_all_rta_prompts()

        st.write("Метрика RtA выбрана. Необходим RTA промпт.")
        use_rta_existing = st.radio("RTA промпт:", ("Выбрать из базы", "Ввести свой"))
        rta_prompt_selected = None
        all_prompts = get_all_prompts()
        if use_rta_existing == "Выбрать из базы":
            if all_prompts:
                names = [p["name"] for p in all_prompts]
                selected_name = st.selectbox("Выберите RTA промпт по имени:", names)
                for rp in all_prompts:
                    if rp["name"] == selected_name:
                        rta_prompt_selected = rp["prompt"]
                        break
            else:
                st.write("Нет доступных RTA промптов. Введите свой.")
        else:
            rta_prompt_selected = render_prompt_creation_section(var_cols=[])

        rta_target = st.text_input("Целевое значение для RtA:", value=1)
        rta_model = st.selectbox(
            "Модель для RTA:",
            MODELS,
            index=MODELS.index(RTA_MODEL) if RTA_MODEL in MODELS else 0,
        )
        return rta_prompt_selected, rta_model, rta_target


def render_models_section() -> List[str]:
    with st.expander("Выбор моделей для задачи", expanded=False):
        selected_models = st.multiselect("Выберите модели:", MODELS)
        return selected_models


def render_preview_and_save_task(
    dataset_name: str,
    var_cols: List[str],
    selected_prompt: str,
    selected_regexp: str,
    target_value: Any,
    selected_models: List[str],
    metric: str,
    rta_prompt_selected: Optional[str],
    rta_model: Optional[str],
):
    with st.expander("Предпросмотр и сохранение задачи", expanded=False):
        if (
            selected_prompt
            and selected_regexp
            and selected_models
            and (target_value or metric == "RtA")
        ):
            group_name = st.text_input("Группа задачи (group):", value="default")
            task_name = st.text_input("Имя задачи:", value=f"{dataset_name}_{metric}")

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
                "task_name": task_name,
                "dataset_name": dataset_name,
                "prompt": selected_prompt,
                "variables_cols": var_cols,
                "models": selected_models,
                "metric": metric,
                "regexp": selected_regexp,
                "group": group_name,
            }
            if metric == "RtA":
                task_data["rta_prompt"] = rta_prompt_selected
                task_data["rta_model"] = rta_model
                task_data["target"] = target_value
            else:
                task_data["target"] = target_value

            st.json(task_data, expanded=False)

            if st.button("Загрузить задачу в базу"):
                db_client.insert_task(task_data)
                st.success("Задача успешно добавлена!")


def render_create_task_tab():
    st.header("Создать новую задачу")

    all_datasets = db_client.get_all_datasets()
    selected_dataset = st.selectbox("Выберите датасет:", all_datasets)
    if selected_dataset:
        var_cols, metric, target_value = render_dataset_varcols_section(
            selected_dataset
        )

    if var_cols and metric is not None:
        selected_regexp = render_regexp_section(metric)
        if selected_regexp:
            selected_prompt = render_prompt_selection_section(var_cols)
            if selected_prompt:
                rta_prompt_selected = None
                rta_model = None
                if metric == "RtA":
                    rta_prompt_selected, rta_model, target_value = render_rta_prompt_section()
                selected_models = render_models_section()

                render_preview_and_save_task(
                    selected_dataset,
                    var_cols,
                    selected_prompt,
                    selected_regexp,
                    target_value,
                    selected_models,
                    metric,
                    rta_prompt_selected,
                    rta_model,
                )


def render_update_task():
    with st.expander("Обновить задачу", expanded=False):            
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

        # Получаем текущий список моделей для задачи
        current_models = task_to_update.get("models", [])

        # Предполагается, что есть функция get_models(), которая возвращает список всех доступных моделей
        all_models = MODELS

        # Предлагаем выбрать новые модели для задачи
        selected_models = st.multiselect(
            "Выберите модели для задачи:",
            options=all_models,
            default=current_models
            )

        if st.button("Обновить модели задачи"):
            update_data = {
                "models": selected_models
            }
            task_id = task_to_update["_id"]
            db_client.update_task(task_id, update_data)
            st.success("Модели для задачи обновлены!")
            st.rerun()



# Новый код для вкладки мониторинга очередей
def highlight_status(s):
    if s == "Ошибка":
        return "background-color: red; color: white;"
    elif s == "В процессе":
        return "background-color: orange; color: white;"
    elif s == "Завершено":
        return "background-color: green; color: white;"
    else:
        return ""


def load_data_for_dashboard(db_client, collections):
    data = []
    # Исключаем некоторые коллекции из отображения
    exclude = ["delete_me", "test"]
    collections = [c for c in collections if c not in exclude]

    for collection_name in collections:
        total_tasks = db_client.count_total_tasks(collection_name)
        statuses = [
            "pending",
            "completed",
            "failed",
            "extracted",
            "processing",
            "transfered_to_rta",
        ]
        status_counts = {
            status: db_client.count_tasks_by_status(collection_name, status)
            for status in statuses
        }

        # Определение статуса коллекции
        if status_counts["failed"] > 0:
            status = "Ошибка"
        elif status_counts["pending"] > 0 or status_counts["processing"] > 0:
            status = "В процессе"
        else:
            status = "Завершено"

        data_row = {
            "Коллекция": collection_name,
            "Всего задач": total_tasks,
            "В ожидании": status_counts["pending"],
            "Выполнено": status_counts["completed"] + status_counts["transfered_to_rta"],
            "Измерено": status_counts["extracted"],
            "С ошибками": status_counts["failed"],
            "Статус": status,
        }
        data.append(data_row)
    return pd.DataFrame(data)


def show_errors(db_client, collections):
    st.header("Уникальные сообщения об ошибках")
    with st.expander("Показать ошибки", expanded=False):
        for collection_name in collections:
            failed_tasks = db_client.get_tasks_by_status(collection_name, "failed")
            if len(failed_tasks) > 0:
                error_messages = [
                    task.get("error", "Нет информации об ошибке")
                    for task in failed_tasks
                ]
                error_counts = Counter(error_messages)
                st.subheader(f"Коллекция: {collection_name}")
                for error_message, count in error_counts.items():
                    st.write(
                        f"**Ошибка:** {error_message} | **Количество:** {count}"
                    )
                st.write("---")

        if st.button("Перезапустить задачи с ошибками", key="restart_failed_tasks"):
            for collection_name in collections:
                modified_count = restart_failed_tasks(db_client, collection_name)
                if modified_count > 0:
                    st.write(
                        f"В коллекции '{collection_name}' перезапущено {modified_count} задач."
                    )
        else:
            st.write(
                "Нажмите кнопку выше, чтобы перезапустить все задачи с ошибками."
            )

def restart_failed_tasks(db_client, collection_name):
    count1 = db_client.update_tasks_status(
        collection_name, "failed", "pending"
    )
    return count1


def render_progressbar():
    st.header("Мониторинг очередей")
    collections_to_process = sorted(
        [
            col
            for col in db_client.list_collections()
            if col.startswith("queue_")
        ]
    )

    df = load_data_for_dashboard(db_client, collections_to_process)

    if st.button("Обновить таблицу", key="refresh_dashboard"):
        df = load_data_for_dashboard(db_client, collections_to_process)

    df = df.sort_values("Коллекция").reset_index(drop=True)
    df_style = df.style.applymap(highlight_status, subset=["Статус"])

    st.write(df_style)

    if df["С ошибками"].sum() > 0:
        show_errors(db_client, collections_to_process)

    st.header("Просмотр данных коллекции")
    if collections_to_process:
        selected_collection = st.selectbox(
            "Выберите коллекцию",
            collections_to_process,
            key="dashboard_select_collection",
        )

        if selected_collection:
            if st.button("Показать данные коллекции", key=f"show_data_{selected_collection}"):
                st.session_state[f"data_loaded_{selected_collection}"] = True

            if st.session_state.get(f"data_loaded_{selected_collection}", False):
                collection = db_client.get_collection(selected_collection)
                data = list(collection.find())
                df_collection = pd.DataFrame(data)

                if "_id" in df_collection.columns:
                    df_collection = df_collection.drop(columns=["_id"])

                if "model" in df_collection.columns:
                    models_in_data = df_collection["model"].unique()
                    filter_models = st.multiselect(
                        "Фильтровать по моделям",
                        options=models_in_data,
                        default=models_in_data,
                        key=f"dashboard_filter_models_{selected_collection}",
                    )
                    df_filtered = df_collection[df_collection["model"].isin(filter_models)]
                else:
                    df_filtered = df_collection

                st.dataframe(df_filtered)

                csv = df_filtered.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Скачать результаты в CSV",
                    data=csv,
                    file_name=f"{selected_collection}_results.csv",
                    mime="text/csv",
                    key=f"download_csv_{selected_collection}",
                )
    else:
        st.info("Нет доступных коллекций для просмотра.")


def render_metrics_tab():
    st.header("Метрики моделей")

    # Получаем все коллекции, начинающиеся с 'results'
    results_collections = ['RtAR', 'TFNR', 'Accuracy', 'Correlation']
    if results_collections:
        # Позволяем пользователю выбрать коллекцию
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

def visualize_metrics(results_data, collection_name):
    # Преобразуем данные в DataFrame
    results_df = pd.DataFrame(results_data)

    # Удаляем служебное поле '_id' если оно есть
    if "_id" in results_df.columns:
        results_df = results_df.drop(columns=["_id"])

    # Предполагается, что данные метрик содержат поля 'dataset_name', 'model', 'value'
    # и 'task_name' (если требуется). Для визуализации используем 'dataset_name' и 'model'.
    if "dataset_name" not in results_df.columns or "model" not in results_df.columns or "value" not in results_df.columns:
        st.error("В данных отсутствуют необходимые поля (dataset_name, model, value).")
        return

    datasets = results_df["dataset_name"].unique()
    models = results_df["model"].unique()

    # Добавляем виджеты для фильтрации
    selected_datasets = st.multiselect(
        "Выберите датасеты",
        options=datasets,
        default=list(datasets),
        key=f"metrics_datasets_{collection_name}",
    )
    selected_models = st.multiselect(
        "Выберите модели",
        options=models,
        default=list(models),
        key=f"metrics_models_{collection_name}",
    )

    # Применяем фильтры к данным
    filtered_df = results_df[
        (results_df["dataset_name"].isin(selected_datasets))
        & (results_df["model"].isin(selected_models))
    ]

    if filtered_df.empty:
        st.info("Нет данных для отображения с выбранными фильтрами.")
        return

    # Группируем данные и вычисляем среднее значение 'value'
    # для каждой пары 'dataset_name' - 'model'
    pivot_table = filtered_df.pivot_table(
        index="model", columns="dataset_name", values="value", aggfunc="mean"
    )

    st.subheader("Таблица метрик по датасетам и моделям")
    st.dataframe(pivot_table)

    # Визуализация данных
    st.subheader("Визуализация метрик")
    st.bar_chart(pivot_table)


# -------------------------------------
# Основной интерфейс
# -------------------------------------

tabs = st.tabs(
    [
        "Визуализация по задачам",
        "Управление датасетами",
        "Создать задачу",
        "Метрики моделей",
    ]
)

with tabs[0]:
    render_tasks_visualization_tab()

with tabs[1]:
    render_dataset_management_tab()

with tabs[2]:
    render_create_task_tab()

with tabs[3]:
    render_metrics_tab()
