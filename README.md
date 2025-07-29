# TrustVar: A Dynamic Framework for Trustworthiness Evaluation and Task Variation Analysis in Large Language Models

> 📖 **Start with [Documentation Overview](docs/overview.md)** for quick understanding of structure and navigation.

## Project Description

**TrustVar** is a framework built on our previous LLM trustworthiness testing system. While we previously focused on how LLMs handle tasks, we now rethink the evaluation procedure itself. TrustVar shifts the focus: we investigate the quality of tasks themselves, not just model behavior.

### Key Innovation

Unlike traditional frameworks that test models through tasks, TrustVar tests tasks through models. We analyze tasks as research objects, measuring their ambiguity, sensitivity, and structure, then examine how these parameters influence model behavior.

### Core Features

- **Task Variation Generation**: Automatically creates families of task reformulations
- **Model Robustness Testing**: Evaluates model stability under formulation changes
- **Task Sensitivity Index (TSI)**: Measures how strongly formulations affect model success
- **Multi-language Support**: English and Russian tasks with extensible architecture
- **Interactive Pipeline**: Unified system for data loading, task generation, variation, model evaluation, and visual analysis

## Table of Contents

- [Project Architecture](#project-architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [System Components](#system-components)
- [API](#api)
- [Metrics](#metrics)
- [Deployment](#deployment)
- [Development](#development)
- [Support](#support)

## Project Architecture

![TrustVar Architecture](docs/Screenshot%202025-07-24%20at%2013.35.16.png)

### Core Components

1. **MongoDB** — Primary database for storing tasks, results, and metrics
2. **Langchain Backend** — Server-side for request processing and interaction with language models
3. **Streamlit Frontend** — Modern web interface for monitoring and management
4. **Task Runners** — Set of specialized task processors
5. **Ollama** — Service for local language model execution

### Data Flow

1. **Task Creation** → MongoDB (collection `tasks`)
2. **Task Processing** → Task Processor → MongoDB (collections `queue_*`)
3. **Inference Execution** → Runner → Langchain Backend → LLM
4. **Metrics Collection** → Metrics Runner → MongoDB
5. **Visualization** → Streamlit Frontend → MongoDB

## Project Structure

```
TrustVar/
├── docs/                          # Project documentation
│   ├── README.md                  # Main documentation
│   ├── components.md              # Component documentation
│   ├── deployment.md              # Deployment guide
│   ├── api.md                     # API documentation
│   ├── metrics.md                 # Metrics documentation
│   └── Screenshot 2025-07-24 at 13.35.16.png  # Architecture diagram
├── utils/                         # Utilities and common components
│   ├── constants.py               # Constants and settings
│   ├── db_client.py               # MongoDB client
│   ├── src.py                     # Helper functions
│   ├── sync_task.py               # Task synchronization
│   └── __init__.py
├── runners/                       # Task processors
│   ├── run.py                     # Main task processor
│   ├── run_metrics.py             # Metrics processor
│   ├── run_regexp.py              # Data extraction from responses
│   ├── run_rta_queuer.py          # RtA task processor
│   ├── task_processor.py          # Task processor
│   └── README.md                  # Processors documentation
├── monitoring/                    # Web monitoring interface
│   ├── app_main.py                # Main Streamlit application
│   ├── config.yaml                # Authentication configuration
│   ├── dataset_management.py      # Dataset management
│   ├── metrics.py                 # Metrics display
│   ├── prompts_tasks.py           # Task creation
│   ├── tasks.py                   # Task visualization
│   └── src.py                     # Helper functions
├── langchain_back/                # Backend service
├── pyproject.toml                 # Poetry configuration
├── docker-compose.yml             # Docker Compose configuration
├── Dockerfile                     # Docker image
└── README.md                      # Main README
```

## Quick Start

### Requirements

- **Docker** and **Docker Compose**
- **Python 3.11+** (for local development)
- **Poetry** (for dependency management)

### Launch with Docker

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd TrustVar
   ```

2. **Create `.env` file with environment variables:**
   ```env
   MONGO_INITDB_ROOT_USERNAME=admin
   MONGO_INITDB_ROOT_PASSWORD=password
   MONGO_INITDB_ROOT_PORT=27017
   YANDEX_API_KEY=your_yandex_key
   OPENAI_KEY=your_openai_key
   API_URL=http://localhost:45321/generate
   OLLAMA_BASE_URL=http://localhost:12345
   CURRENT_UID=1000
   CURRENT_GID=1000
   ```

3. **Launch all services:**
   ```bash
   docker-compose up -d
   ```

4. **Download datasets and auxiliary information:**
   
   After running `docker-compose up`, you need to download the datasets and auxiliary information from our Google Drive and upload them to MongoDB:
   
   **[📥 Download Datasets](https://drive.google.com/drive/folders/1jvBWvAc9JcjLYQ8T09xoDKUkwjCf7tiI?usp=sharing)**
   
   The drive contains:
   - **Accuracy_Groups.json** - Accuracy metrics grouped by categories
   - **Accuracy.json** - Main accuracy dataset
   - **Correlation.json** - Correlation metrics
   - **IncludeExclude.json** - Include/Exclude analysis data
   - **RtAR.json** - Refuse to Answer metrics
   - **TFNR.json** - True False Negative Rate metrics
   - **jailbreak.json** - Jailbreak detection tasks
   - **ood_detection.json** - Out-of-distribution detection
   - **privacy_assessment.json** - Privacy assessment tasks
   - **stereotypes_detection_3.json** - Stereotype detection
   - **tasks.json** - Task definitions
   - And many more specialized datasets...

   **Instructions:**
   1. Download all JSON files from the Google Drive
   2. Place them in the `data/datasets/` directory of your TrustVar installation
   3. Run the upload script to populate MongoDB:
      ```bash
      cd data
      python upload.py
      ```
   4. Restart the services if necessary: `docker-compose restart`

5. **Open the web interface:**
   - Monitoring: http://localhost:27366 (or http://83.143.66.61:27366 for remote access)
   - MongoDB Express: http://localhost:8081

   **Authentication credentials:**
   - Username: `user`
   - Password: `resu123`

### Local Development

1. **Install dependencies:**
   ```bash
   poetry install
   ```

2. **Activate virtual environment:**
   ```bash
   poetry shell
   ```

3. **Launch individual components:**
   ```bash
   # Backend
   python langchain_back/main.py
   
   # Frontend
   streamlit run monitoring/app_main.py --server.port 27366
   
   # Runners
   python -m runners.run
   python -m runners.run_metrics
   python -m runners.task_processor
   ```

## Documentation

Detailed documentation is available in the `docs/` folder:

- **[Main Documentation](docs/README.md)** - System overview and architecture
- **[System Components](docs/components.md)** - Detailed description of all components
- **[Setup](docs/setup.md)** - Setup and installation guide
- **[Deployment](docs/deployment.md)** - Deployment guide
- **[API](docs/api.md)** - API documentation
- **[Metrics](docs/metrics.md)** - Description of supported metrics

## System Components

### 1. MongoDB - Database
Primary database for storing tasks, results, and metrics.

### 2. Langchain Backend
Server-side for processing requests to language models.

### 3. Streamlit Frontend
Modern web interface for monitoring and management.

### 4. Task Runners
Set of specialized task processors:
- `run.py` - Main task processor
- `run_metrics.py` - Metrics processor
- `run_regexp.py` - Data extraction from responses
- `run_rta_queuer.py` - RtA task processor
- `task_processor.py` - Task processor

### 5. Ollama
Service for local language model execution.

## API

The system provides REST API for interaction with language models:

- **POST /generate** - Response generation
- **GET /health** - Health check
- **GET /models** - Model list

Detailed API documentation: [docs/api.md](docs/api.md)

## Metrics

Supported metric types:

- **Accuracy** - Response accuracy
- **RtA (Refuse to Answer)** - Analysis of answer refusals
- **Correlation** - Correlation with reference answers
- **Include/Exclude** - Analysis of element inclusion/exclusion

Detailed metrics description: [docs/metrics.md](docs/metrics.md)

## Deployment

### Docker Compose (recommended)
```bash
docker-compose up -d
```

### Kubernetes
Detailed Kubernetes deployment guide: [docs/deployment.md](docs/deployment.md)

## Development

### Code Structure
- **Modular architecture** with clear separation of responsibilities
- **Docker containerization** for simplified deployment
- **Poetry** for dependency management
- **Streamlit** for modern web interface

### Adding New Metrics
1. Create a new module in `runners/`
2. Add configuration in `utils/constants.py`
3. Update web interface in `monitoring/`

### Adding New Models
1. Add model to `MODELS` list in `utils/constants.py`
2. Configure corresponding provider in Langchain Backend
3. Update documentation

## Support

For support:
1. Check documentation in the `docs/` folder
2. Study component logs
3. Create an issue in the project repository

## License

This project is licensed under the MIT License.
