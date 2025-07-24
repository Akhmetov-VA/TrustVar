# prompts_jobs.py
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from dataset_management import render_dataset_varcols_section
from utils.constants import MODELS, RTA_MODEL, AUGMENTATIONS, TASKS, AUGMENT_PROMPT, CURRENT_AUGMENT_PROMPT
from utils.db_client import MongoDBClient, MongoDBConfig

# Initializing the DB client
config = MongoDBConfig(database="TrustGen")
db_client = MongoDBClient(config)

DEFAULT_REGEX = r"(?:^\W*([01]).*)|(?:.*([01])\W*$)"

def show_all_prompts() -> None:
    """Show all prompt from the database."""
    coll_name = "prompt_storage"
    if coll_name not in db_client.list_collections():
        st.write("There are no prompts in the storage.")
        return
    coll = db_client.get_collection(coll_name)
    prompts = list(coll.find({}))
    if prompts:
        df = pd.DataFrame(prompts)
        if "_id" in df.columns:
            df = df.drop(columns=["_id"])
        st.write("Existing prompt (name, prompt):")
        st.dataframe(df)
    else:
        st.write("There are no prompts in the storage.")

def get_all_prompts() -> List[Dict[str, Any]]:
    coll_name = "prompt_storage"
    if coll_name not in db_client.list_collections():
        return []
    coll = db_client.get_collection(coll_name)
    return list(coll.find({}))

def prompt_exists(name: str) -> bool:
    coll_name = "prompt_storage"
    if coll_name not in db_client.list_collections():
        return False
    coll = db_client.get_collection(coll_name)
    return coll.find_one({"name": name}) is not None

def insert_prompt_global(name: str, prompt: str) -> None:
    coll_name = "prompt_storage"
    coll = db_client.get_collection(coll_name)
    coll.insert_one({"name": name, "prompt": prompt})

def show_all_rta_prompts() -> None:
    show_all_prompts()

def render_prompt_creation_section(var_cols: List[str]) -> Optional[str]:
    """UI to create a new fragrance."""
    hint = f"You can use any selected speakers: {', '.join('{' + c + '}' for c in var_cols)}."
    st.write(hint)
    prompt_name = st.text_input("Enter the name of the new prompt:")
    user_prompt = st.text_area("Enter your prompt:", value=hint)
    if user_prompt and prompt_name:
        missing_cols = [c for c in var_cols if f"{{{c}}}" not in user_prompt]
        if missing_cols:
            st.error("Placeholders are missing: " + ", ".join(missing_cols))
        else:
            if prompt_exists(prompt_name):
                st.warning(f"A prompt named '{prompt_name}' already exists. You can use it.")
                if st.button("Use an existing prompt"):
                    for p in get_all_prompts():
                        if p["name"] == prompt_name:
                            return p["prompt"]
            else:
                if st.button("Add prompt to the database"):
                    insert_prompt_global(prompt_name, user_prompt)
                    st.success("Prompt added!")
                    return user_prompt
    return None

def render_prompt_selection_section(var_cols: List[str]) -> Optional[str]:
    """UI for selecting or creating a product."""
    selected_prompt = None
    with st.expander("Selecting or creating a product", expanded=False):
        show_all_prompts()
        use_existing_prompt = st.radio("Promt:", ("Select from the database", "Enter your own"))
        all_prompt_docs = get_all_prompts()
        if use_existing_prompt == "Select from the database":
            if all_prompt_docs:
                names = [p["name"] for p in all_prompt_docs]
                selected_name = st.selectbox("Select the prompt by name:", names, key="prompt_selectbox")
                for p in all_prompt_docs:
                    if p["name"] == selected_name:
                        selected_prompt = p["prompt"]
                        break
                if selected_prompt:
                    for c in var_cols:
                        if f"{{{c}}}" not in selected_prompt:
                            st.warning(f"The placeholder for the column was not found in the product. {c}")
            else:
                st.write("There are no prompt available. Enter your own.")
        else:
            selected_prompt = render_prompt_creation_section(var_cols)
    return selected_prompt

def show_existing_regexp(metric: str) -> None:
    coll_name = f"regexp_{metric}"
    if coll_name not in db_client.list_collections():
        st.write("There are no adjustments for this metric.")
        return
    coll = db_client.get_collection(coll_name)
    docs = list(coll.find({}))
    if docs:
        df = pd.DataFrame(docs)
        if "_id" in df.columns:
            df = df.drop(columns=["_id"])
        st.write("Existing regular schedules (name, pattern, metric):")
        st.dataframe(df)
    else:
        st.write("There are no adjustments for this metric.")

def get_all_regexps_for_metric(metric: str) -> List[Dict[str, Any]]:
    coll_name = f"regexp_{metric}"
    if coll_name not in db_client.list_collections():
        return []
    coll = db_client.get_collection(coll_name)
    return list(coll.find({}))

def insert_regexp_global(name: str, pattern: str, metric: str) -> None:
    coll_name = f"regexp_{metric}"
    coll = db_client.get_collection(coll_name)
    coll.insert_one({"name": name, "pattern": pattern, "metric": metric})

def render_regexp_section(metric: str) -> Optional[str]:
    selected_regexp = None
    with st.expander("Selecting or creating a regular schedule for metrics", expanded=False):
        show_existing_regexp(metric)
        use_existing_regexp = st.radio("Regular schedule:", ("Existing", "Own"))
        if use_existing_regexp == "Existing":
            regexps = get_all_regexps_for_metric(metric)
            if regexps:
                names = [r["name"] for r in regexps]
                selected_name = st.selectbox("Select a regular by name:", names, key="regexp_selectbox")
                for r in regexps:
                    if r["name"] == selected_name:
                        selected_regexp = r["pattern"]
                        break
            else:
                st.write("There are no adjustments available for this metric..")
        else:
            st.write(f"By default, we offer: {DEFAULT_REGEX}")
            custom_regexp = st.text_input("Enter your regular schedule:", value=DEFAULT_REGEX)
            if custom_regexp:
                if db_client.validate_regex(custom_regexp):
                    regexp_name = st.text_input("Enter a name for this regular:")
                    if regexp_name and st.button("Add regular season tickets to the database"):
                        insert_regexp_global(regexp_name, custom_regexp, metric)
                        st.success("Regular season added!")
                        selected_regexp = custom_regexp
                else:
                    st.error("Invalid regular expression!")
    return selected_regexp

def render_rta_prompt_section() -> Tuple[Optional[str], Optional[str], Any]:
    rta_prompt_selected = None
    rta_model = None
    rta_target = None
    with st.expander("Selecting or creating an RTA prompt", expanded=False):
        show_all_rta_prompts()
        st.write("The RtA metric is selected. An RTA prompt is required.")
        use_rta_existing = st.radio("RTA promt:", ("Select from the database", "Enter your"))
        all_prompt_docs = get_all_prompts()
        if use_rta_existing == "Select from the database":
            if all_prompt_docs:
                names = [p["name"] for p in all_prompt_docs]
                selected_name = st.selectbox("Select RTA prompt by name:", names, key="rta_prompt_selectbox")
                for rp in all_prompt_docs:
                    if rp["name"] == selected_name:
                        rta_prompt_selected = rp["prompt"]
                        break
            else:
                st.write("There are no RTA prompt available. Enter your.")
        else:
            rta_prompt_selected = render_prompt_creation_section(var_cols=[])
        rta_target = st.text_input("The target value for RtA:", value="1")
        rta_model = st.selectbox(
            "The model for RTA:",
            MODELS,
            index=MODELS.index(RTA_MODEL) if RTA_MODEL in MODELS else 0,
            key="rta_model_selectbox",
        )
    return rta_prompt_selected, rta_model, rta_target

def render_models_section() -> List[str]:
    with st.expander("Selecting models for the task", expanded=False):
        return st.multiselect("Select models:", MODELS)

# def render_dynamic_variations() -> List[str]:
#     with st.expander("Dynamic dataset augmentation [AUG]", expanded=False):
#         return st.multiselect("Choose the augmentation method:", AUGMENTATIONS)

# 🔑 session_state keys
PROMPT_KEY = "user_prompt_temp"
MODE_KEY = "augmentation_mode"

def render_dynamic_variations() -> Optional[List[str]]:
    global CURRENT_AUGMENT_PROMPT
    # Инициализируем состояние
    if PROMPT_KEY not in st.session_state:
        st.session_state[PROMPT_KEY] = AUGMENT_PROMPT

    if MODE_KEY not in st.session_state:
        st.session_state[MODE_KEY] = "Use predefined augmentations"

    with st.expander("Dynamic dataset augmentation", expanded=False):
        # Выбор режима: внутренний или пользовательский
        st.session_state[MODE_KEY] = st.radio(
            "Select input method:",
            ("Use predefined augmentations", "Write your own augmentation prompt"),
            index=0 if st.session_state[MODE_KEY] == "Use predefined augmentations" else 1,
            key="radio_mode"
        )

        selected = None
        if st.session_state[MODE_KEY] == "Use predefined augmentations":
            selected = st.multiselect("Choose the augmentation method:", AUGMENTATIONS)

        elif st.session_state[MODE_KEY] == "Write your own augmentation prompt":
            st.session_state[PROMPT_KEY] = st.text_area(
                "Write your augmentation prompt here:",
                value=st.session_state[PROMPT_KEY],
                key="prompt_area"
            )

        st.markdown("---")
        st.subheader("📝 Current prompt in use:")
        current_prompt = (
            st.session_state[PROMPT_KEY]
            if st.session_state[MODE_KEY] == "Write your own augmentation prompt"
            else AUGMENT_PROMPT
        )
        st.code(current_prompt, language="markdown")

        if st.session_state[MODE_KEY] == "Write your own augmentation prompt":
            CURRENT_AUGMENT_PROMPT = st.session_state[PROMPT_KEY]
        else:
            CURRENT_AUGMENT_PROMPT = AUGMENT_PROMPT

        return selected if st.session_state[MODE_KEY] == "Use predefined augmentations" else None
            
def build_task_data(
    task_type: str,
    task_name: str,
    dataset_name: str,
    var_cols: List[str],
    prompt: str,
    regexp: str,
    target_value: Any,
    models: List[str],
    metric: str,
    variations: Optional[List[str]],
    rta_prompt: Optional[str],
    rta_model: Optional[str],
    include_column: Optional[str],
    exclude_column: Optional[str],
    group_name: str = "default",
) -> Dict[str, Any]:
    data = {
        "task_type": task_type,
        "task_name": task_name,
        "dataset_name": dataset_name,
        "prompt": prompt,
        "variables_cols": var_cols,
        "models": models,
        "metric": metric,
        "regexp": regexp,
        "group": group_name,
    }
    if metric == "RtA":
        data["rta_prompt"] = rta_prompt
        data["rta_model"] = rta_model
        data["target"] = target_value
    elif metric == "include_exclude":
        data["include_column"] = include_column
        data["exclude_column"] = exclude_column
    else:
        data["target"] = target_value
    data["dynamic_augments"] = variations if variations else []
    return data

def render_preview_and_save_task(
    task_type: str,
    dataset_name: str,
    var_cols: List[str],
    selected_prompt: str,
    selected_regexp: str,
    target_value: Any,
    selected_models: List[str],
    metric: str,
    selected_variations: Optional[List[str]],
    rta_prompt_selected: Optional[str],
    rta_model: Optional[str],
    include_column: Optional[str],
    exclude_column: Optional[str],
):
    with st.expander("Previewing and saving an issue", expanded=False):
        if (
            selected_prompt
            and selected_regexp
            and selected_models
            and (target_value or metric in ["RtA", "include_exclude"])
        ):
            task_name = st.text_input("Task Name:", value=f"{dataset_name}")
            group_name = st.text_input("Task Group:", value="default")
            st.subheader("5 Random Sample Preview:")
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
                    st.write(f"**Example {i + 1}:** {filled_prompt}")
            if selected_variations:
                st.write(f"**Dynamic augmentation methods:** {' | '.join(selected_variations)}")
            st.write("**DB Record Structure:**")
            task_data = build_task_data(
                task_type=task_type,
                task_name=task_name,
                dataset_name=dataset_name,
                var_cols=var_cols,
                prompt=selected_prompt,
                regexp=selected_regexp,
                target_value=target_value,
                models=selected_models,
                metric=metric,
                variations=selected_variations,
                rta_prompt=rta_prompt_selected,
                rta_model=rta_model,
                include_column=include_column,
                exclude_column=exclude_column,
                group_name=group_name,
            )
            st.json(task_data, expanded=False)
            if st.button("Upload task"):
                db_client.insert_task(task_data)
                st.success("Task was uploaded successfully!")

def render_create_task_tab():
    st.header("Create new task")
    all_datasets = db_client.get_all_datasets()
    if "regestry" in all_datasets:
        all_datasets.remove("regestry")
    task_type = st.selectbox("Task type:", TASKS, key="select_task_type_selectbox")
    selected_dataset = st.selectbox("Select dataset:", sorted(all_datasets), key="select_ds_selectbox")
    if task_type and selected_dataset:
        var_cols, metric, target_column, include_column, exclude_column = render_dataset_varcols_section(selected_dataset)
        if var_cols and metric is not None:
            selected_prompt = render_prompt_selection_section(var_cols)
            if selected_prompt:
                if metric != "include_exclude":
                    selected_regexp = render_regexp_section(metric)
                else:
                    selected_regexp = "The include_exclude metric does not use regexp."
                if selected_regexp:
                    rta_prompt_selected = None
                    rta_model = None
                    rta_target_value = None
                    if metric == "RtA":
                        rta_prompt_selected, rta_model, rta_target_value = render_rta_prompt_section()
                    selected_models = render_models_section()
                    final_target = rta_target_value if metric == "RtA" else target_column
                    # The augmentation expander is only for Compare model behaviour
                    if task_type == "Compare model behaviour":
                        selected_variations = render_dynamic_variations()
                    else:
                        selected_variations = None
                    render_preview_and_save_task(
                        task_type=task_type,
                        dataset_name=selected_dataset,
                        var_cols=var_cols,
                        selected_prompt=selected_prompt,
                        selected_regexp=selected_regexp,
                        target_value=final_target,
                        selected_models=selected_models,
                        metric=metric,
                        selected_variations=selected_variations,
                        rta_prompt_selected=rta_prompt_selected,
                        rta_model=rta_model,
                        include_column=include_column,
                        exclude_column=exclude_column,
                    )
