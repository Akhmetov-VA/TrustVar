import streamlit as st
import streamlit_authenticator as stauth
import yaml
from dataset_management import render_dataset_management_tab
from metrics import render_metrics_tab
from model_management import render_model_management_tab
from prompts_tasks import render_create_task_tab
from tasks import render_tasks_visualization_tab
from yaml.loader import SafeLoader

st.set_page_config(
    page_title="TrustVar Dashboard", layout="wide", initial_sidebar_state="collapsed"
)

# Пользовательские стили
st.markdown(
    """
<style>
    /* Основные стили */
    .main {
        padding-top: 2rem;
    }
    
    /* Стиль для заголовка */
    .main-header {
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .main-title {
        font-size: 4.5rem;
        font-weight: 800;
        color: #8b5cf6;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        text-shadow: 0 0 30px rgba(139, 92, 246, 0.5);
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        color: #a78bfa;
        font-weight: 400;
        opacity: 0.9;
    }
    
    /* Стили для кнопок-карточек */
    .stButton > button {
        background: linear-gradient(145deg, #1e1e1e, #2a2a2a);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 16px;
        padding: 2rem 1rem;
        height: 200px;
        width: 100%;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        color: white;
        font-weight: normal;
        text-align: center;
        white-space: pre-wrap;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(145deg, rgba(139, 92, 246, 0.1), rgba(167, 139, 250, 0.1));
        opacity: 0;
        transition: opacity 0.3s ease;
        border-radius: 16px;
    }
    
    .stButton > button:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 40px rgba(139, 92, 246, 0.3);
        border-color: #8b5cf6;
        background: linear-gradient(145deg, #2a2a2a, #353535);
    }
    
    .stButton > button:hover::before {
        opacity: 1;
    }
    
    .stButton > button:active {
        transform: translateY(-4px) scale(1.01);
        transition: all 0.1s ease;
    }
    
    /* Анимация для иконок в карточках */
    .stButton > button:hover {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% {
            box-shadow: 0 12px 40px rgba(139, 92, 246, 0.3);
        }
        50% {
            box-shadow: 0 12px 50px rgba(139, 92, 246, 0.4);
        }
        100% {
            box-shadow: 0 12px 40px rgba(139, 92, 246, 0.3);
        }
    }
    
    /* Стили для форм входа */
    .stTextInput > div > div > input {
        background-color: #1e1e1e;
        border: 2px solid rgba(139, 92, 246, 0.2);
        border-radius: 10px;
        color: white;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #8b5cf6;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.1);
        background-color: #2a2a2a;
    }
    
    /* Кнопка входа в систему */
    div[data-testid="stForm"] .stButton > button {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
        height: auto;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    div[data-testid="stForm"] .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.4);
        background: linear-gradient(135deg, #9333ea 0%, #8b5cf6 100%);
    }
    
    /* Кнопка закрытия */
     /* button[key="close_button"] {
        background: rgba(239, 68, 68, 0.9) !important;
        padding: 0.5rem 1.5rem !important;
        height: auto !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }
    
    button[key="close_button"]:hover {
        background: #ef4444 !important;
        transform: scale(1.05) !important;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4) !important;
    } */
    
    /* Разделитель */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, rgba(139, 92, 246, 0.3), transparent);
        margin: 2.5rem 0;
    }
    
    /* Дополнительные эффекты */
    .stButton > button {
        font-size: 0.95rem;
        line-height: 1.4;
    }
    
    /* Градиентный фон для всего приложения */
    .stApp {
        background: radial-gradient(circle at top right, rgba(139, 92, 246, 0.05), transparent),
                    radial-gradient(circle at bottom left, rgba(167, 139, 250, 0.05), transparent);
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_config():
    with open("monitoring/config.yaml") as file:
        return yaml.load(file, Loader=SafeLoader)


config = load_config()

# Initializing the authenticator
authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

# Заголовок приложения
st.markdown(
    """
<div class="main-header">
    <h1 class="main-title">TrustVar</h1>
    <p class="main-subtitle">A Dynamic Framework for Trustworthiness Evaluation and Task Variation Analysis in LLMs</p>
</div>
""",
    unsafe_allow_html=True,
)

try:
    authenticator.login()
except Exception as e:
    st.error(e)

if st.session_state["authentication_status"]:
    # Создаем состояние для выбранной вкладки
    if "selected_tab" not in st.session_state:
        st.session_state.selected_tab = None

    # Данные для карточек
    features = [
        {
            "icon": "📊",
            "title": "TASK VISUALIZATION",
            "description": "Monitor and analyze task performance",
        },
        {
            "icon": "🗄️",
            "title": "DATASET MANAGEMENT",
            "description": "Upload and manage your datasets",
        },
        {
            "icon": "🤖",
            "title": "MODEL MANAGEMENT",
            "description": "Deploy and LLM models",
        },
        {
            "icon": "✨",
            "title": "TASK CREATION",
            "description": "Configure new LLM tasks",
        },
        {
            "icon": "📈",
            "title": "MODEL METRICS",
            "description": "Track performance and evaluations",
        },
    ]

    # Создаем 5 колонок для карточек
    cols = st.columns(5)
    for idx, (col, feature) in enumerate(zip(cols, features)):
        with col:
            # Форматируем текст для кнопки
            button_text = (
                f"{feature['icon']}\n\n{feature['title']}\n\n{feature['description']}"
            )

            if st.button(
                button_text, key=f"card_{idx}", use_container_width=True, help=None
            ):
                st.session_state.selected_tab = idx

    # Отображаем выбранную вкладку под карточками
    if st.session_state.selected_tab is not None:
        st.markdown("<hr>", unsafe_allow_html=True)

        # Кнопка для скрытия контента
        col1, col2, col3 = st.columns([1, 10, 1])
        # with col3:
        #     if st.button("✕ Close", key="close_button"):
        #         st.session_state.selected_tab = None
        #         st.rerun()

        # Отображаем соответствующий контент
        if st.session_state.selected_tab == 0:
            render_tasks_visualization_tab()
        elif st.session_state.selected_tab == 1:
            render_dataset_management_tab()
        elif st.session_state.selected_tab == 2:
            render_model_management_tab()
        elif st.session_state.selected_tab == 3:
            render_create_task_tab()
        elif st.session_state.selected_tab == 4:
            render_metrics_tab()

elif st.session_state["authentication_status"] is False:
    # Стилизованное сообщение об ошибке
    st.markdown(
        """
    <div style='background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.2)); 
                border: 1px solid #ef4444; 
                color: #fca5a5; 
                padding: 1rem; 
                border-radius: 10px; 
                text-align: center;
                backdrop-filter: blur(10px);'>
        ❌ Invalid username or password. Please try again.
    </div>
    """,
        unsafe_allow_html=True,
    )

elif st.session_state["authentication_status"] is None:
    # Стилизованная форма входа
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h3 style='color: #e9d5ff; font-weight: 600;'>Sign in to your account</h3>
            <p style='color: #a78bfa; opacity: 0.8;'>Enter your credentials to access the dashboard</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
