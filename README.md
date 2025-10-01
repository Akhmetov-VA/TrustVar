# TrustVar: A Dynamic Framework for Trustworthiness Evaluation and Task Variation Analysis in Large Language Models

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Akhmetov-VA/TrustVar)
 
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
- [Metrics](#metrics)

## Project Architecture

![TrustVar Architecture](docs/TrustVar_Pipeline.jpg)

### Core Components

- **Data Ingestion** - accepts preformatted datasets in CSV, JSON, Excel, and Parquet formats, supporting both user uploads and built-in collections like SLAVA, RuBia, etc;
- **Task Generator** - applies five controlled transformations: lexico-syntactic paraphrasing, length variation, stylistic shifts, synonym substitution, and word reordering to create semantically equivalent variants;
- **Perturbation Settings** - sets up each transformation with user-configurable parameters (10 by default);
- **Task Pool** - erves as a persistent repository organizing tasks by six trustworthiness dimensions (truthfulness, safety, fairness, robustness, privacy, ethics) and maintaining evaluation queues;
- **LLM Tester** - executes inference on both local models via Ollama and remote APIs, recording outputs with complete metadata for reproducibility;
- **Analyzer** - measures response stability using coefficient of variation, feeding instability flags back for task refinement;
- **Task Meta-Evaluator** - computes the Task Sensitivity Index (TSI) across all model-task pairs, flagging high-TSI items for revision;
- **Evaluator & Visualizer** - computes RtAR, TFNR, Accuracy, and Pearson correlation metrics;
- **Dashboard and Leaderboard** - combine Metrics with Analyser data and display the results for user convenience


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
   BACKEND_HOST=0.0.0.0
   BACKEND_PORT=45321
   FRONTEND_PORT=27366
   API_URL=http://langchain_backend:${BACKEND_PORT}/generate
   MONGO_HOST=mongodb
   MONGO_INITDB_ROOT_USERNAME=username
   MONGO_INITDB_ROOT_PASSWORD=password
   MONGO_INITDB_ROOT_PORT=27017
   OLLAMA_PORT=12345
   OLLAMA_BASE_URL=http://host.docker.internal:${OLLAMA_PORT}
   OPENAI_BASE_URL=base_url_for_providers
   OPENAI_KEY=openai_key
   YANDEX_API_KEY=yandex_key
   YANDEX_BASE_URL=https://llm.api.cloud.yandex.net/v1
   YANDEX_MODEL_URI=model_uri
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
   - Monitoring: http://localhost:27366
   - MongoDB Express: http://localhost:8081

   **Authentication credentials:**
   - Username: `user`
   - Password: `resu123`


## Documentation

Detailed documentation is available in the `docs/` folder:

- **[Main Documentation](docs/README.md)** - System overview and architecture
- **[System Components](docs/components.md)** - Detailed description of all components
- **[Setup](docs/setup.md)** - Setup and installation guide
- **[Deployment](docs/deployment.md)** - Deployment guide
- **[API](docs/api.md)** - API documentation
- **[Metrics](docs/metrics.md)** - Description of supported metrics


## Metrics

Supported metric types:

- **Accuracy** - Response accuracy
- **RtA (Refuse to Answer)** - Analysis of answer refusals
- **Correlation** - Correlation with reference answers
- **Include/Exclude** - Analysis of element inclusion/exclusion

Detailed metrics description: [docs/metrics.md](docs/metrics.md)

## License

This project is licensed under the MIT License.