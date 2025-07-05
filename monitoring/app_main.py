import streamlit as st
import streamlit_authenticator as stauth
import yaml
from dataset_management import render_dataset_management_tab
from metrics import render_metrics_tab
from prompts_tasks import render_create_task_tab
from tasks import render_tasks_visualization_tab
from yaml.loader import SafeLoader

st.set_page_config(page_title="TrustGen Dashboard", layout="wide")


@st.cache_data
def load_config():
    with open("monitoring/config.yaml") as file:
        return yaml.load(file, Loader=SafeLoader)


config = load_config()

# Инициализация аутентификатора
authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

try:
    authenticator.login()
except Exception as e:
    st.error(e)

if st.session_state["authentication_status"]:
    st.write(f"Добро пожаловать, {st.session_state['name']}!")

    # Если пользователь авторизовался, показываем вкладки приложения
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

elif st.session_state["authentication_status"] is False:
    st.error("Неверный логин/пароль")
elif st.session_state["authentication_status"] is None:
    st.warning("Введите ваш логин и пароль")
