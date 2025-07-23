# TrustVar

## Project Description

**TrustVar** is a system for evaluating and monitoring machine learning models in the field of natural language processing. The project includes tools for downloading and processing data, launching inference models, collecting metrics, and visualizing the state of experiments via a web interface.

## Content

- [Project Architecture](#project-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Setup](#setup)
- [Contribution to the project](#contribution-to-the-project)

## Project Architecture

The inference architecture includes the following components:

1. **MongoDB** is a database for storing tasks and results.
2. **LangchainBackend** is a server part for processing requests and interacting with models. (can go to Ollama/Yandex/OpenAI).
3. **StreamlitMonitoring** is a graphical interface for displaying metrics, the current calculation status using LLM, and launching model calculations from the graphical interface.
4. **RunnersGroup** is a group of programs that interacts with MongoDB to update tasks, redirect them, and calculate metrics.
5. **OllamaServer** is a local `Ollama` server for local model inference.

## Project Structure

```plaintext
.
├── benchmark
│   └── runners
│       ├── add_collumn_collection.py
│       ├── add_dataset_field.py
│       ├── drop_collection_pattern.py
│       ├── drop_not_in_models.py
│       ├── drop_pending.py
│       ├── fix_rta_tasks.py
│       ├── revert_old.py
│       ├── revert_rta.py
│       ├── run.py
│       ├── run_metrics.py
│       ├── run_regexp.py
│       ├── run_rta_queuer.py
│       ├── run_sync.py
│       ├── task_processor.py
│       └── update_target.py
├── data_tools
│   ├── download_from_mongo.py
│   ├── download_from_remote.py
│   ├── init_db.py
│   └── upload.py
├── langchain_back
│   └── main.py
├── monitoring
│   ├── .DS_Store
│   ├── app_main.py
│   ├── dataset_management.py
│   ├── metrics.py
│   ├── prompts_tasks.py
│   ├── src.py
│   └── tasks.py
├── utils
│    ├── constants.py
│    ├── db_client.py
│    ├── src.py
│    └── sync_task.py
├── Dockerfile 
├── docker-compose-gpu.yml
├── docker-compose.yml
├── .env
├── pyproject.toml
└── README.md
```

## Installation

### Requirements

- **Python 3.11** or higher
- **Docker Compose**

### Installation Steps

1. **Clone the repository:**

    ```bash
    git clone https://83.143.66.64:27367/lia_icii/trust_llm_ru.git
    cd TrustVar
    ```

2. **Create a `.env` file in the root of the project and add the necessary environment variables:**

    ```dotenv
    API_URL=http://langchain_backend:45321/generate
    FRONTEND_PORT=<>
    MONGO_HOST=mongodb
    MONGO_INITDB_ROOT_USERNAME=<>
    MONGO_INITDB_ROOT_PASSWORD=<>
    MONGO_INITDB_ROOT_PORT=<>
    OLLAMA_BASE_URL=http://host.docker.internal:<>
    OPENAI_BASE_URL=<>
    OPENAI_KEY=<>
    YANDEX_API_KEY=<>
    YANDEX_BASE_URL=<>
    YANDEX_MODEL_URI=<>
    ```

    Replace the values with your own.

## Setup

```bash
docker-compose up -d
```

## Contribution to the project

We welcome the contribution to the development of the project! If you want to make changes or improvements:

1. **Fork the repository**.
2. **Create a new branch** for your changes:

    ```bash
    git checkout -b feature/your-feature
    ```

3. **Make changes** and **commit them**:

    ```bash
    git commit -m "Added a new feature"
    ```

4. **Push the branch** into your fork:

    ```bash
    git push origin feature/your-feature
    ```

5. **Create a Pull Request** to the original repository.

