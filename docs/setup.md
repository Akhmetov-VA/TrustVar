# TrustVar Setup Guide

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+), macOS, Windows (WSL2)
- **RAM**: Minimum 8GB (recommended 16GB+)
- **Disk Space**: 50GB+ free space
- **GPU**: NVIDIA GPU with CUDA support (optional, for local models)

### Software Requirements

- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Python**: 3.11+ (for local development)
- **Poetry**: 1.4+ (for dependency management)

## Dependency Installation

### 1. Docker and Docker Compose

**Ubuntu/Debian:**
```bash
# Update packages
sudo apt-get update

# Install dependencies
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release

# Add Docker GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Add Docker repository
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add user to docker group
sudo usermod -aG docker $USER

# Reboot to apply changes
sudo reboot
```

**macOS:**
```bash
# Install via Homebrew
brew install --cask docker

# Or download Docker Desktop from official website
# https://www.docker.com/products/docker-desktop
```

**Windows:**
1. Download Docker Desktop from [official website](https://www.docker.com/products/docker-desktop)
2. Install and start Docker Desktop
3. Enable WSL2 for better performance

### 2. NVIDIA Container Toolkit (optional)

If you have an NVIDIA GPU and want to use local models:

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Verify installation
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### 3. Python and Poetry

**Python 3.11+:**
```bash
# Ubuntu/Debian
sudo apt-get install python3.11 python3.11-venv python3-pip

# macOS
brew install python@3.11

# Windows
# Download from https://www.python.org/downloads/
```

**Poetry:**
```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Add to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.local/bin:$PATH"

# Verify installation
poetry --version
```

## Project Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd TrustVar
```

### 2. Create Configuration File

Create `.env` file in project root:

```bash
cp docs/env.example .env
```

Edit the `.env` file:

```env
# MongoDB Configuration
MONGO_INITDB_ROOT_USERNAME=admin
MONGO_INITDB_ROOT_PASSWORD=your_secure_password
MONGO_INITDB_ROOT_PORT=27017

# API Keys
YANDEX_API_KEY=your_yandex_api_key_here
YANDEX_MODEL_URI=your_yandex_model_uri_here
YANDEX_BASE_URL=https://llm.api.cloud.yandex.net

OPENAI_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1

# Service URLs
API_URL=http://localhost:45321/generate
OLLAMA_BASE_URL=http://localhost:12345

# Docker User Permissions
CURRENT_UID=1000
CURRENT_GID=1000
```

### 3. Download Datasets and Auxiliary Information

**Important**: After setting up the environment, you need to download the datasets and auxiliary information from our Google Drive:

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

### 4. Upload Datasets to MongoDB

After downloading the datasets, you need to upload them to MongoDB using the provided script:

```bash
# Navigate to the data directory
cd data

# Run the upload script
python upload.py
```

The `upload.py` script will:
- Connect to the local MongoDB instance
- Read all JSON files from the `data/datasets/` directory
- Upload them to appropriate MongoDB collections:
  - `tasks` → `tasks` collection
  - Metrics files (Accuracy, Correlation, etc.) → direct collection names
  - Other datasets → `dataset_*` prefixed collections
- Provide logging output for each uploaded file

**Expected output:**
```
INFO:root:Attempting to connect to local MongoDB...
INFO:root:Connecting to: mongodb://admin:password@localhost:27364/
INFO:root:Connected to local MongoDB successfully.
INFO:root:Uploaded 1500 documents to collection 'Accuracy' from Accuracy.json
INFO:root:Uploaded 500 documents to collection 'tasks' from tasks.json
INFO:root:Datasets upload completed!
INFO:root:All datasets successfully uploaded to local MongoDB!
```

### 5. Get API Keys

#### Yandex Cloud API

1. Register at [Yandex Cloud](https://cloud.yandex.ru/)
2. Create a billing account
3. Create a service account
4. Assign role `ai.languageModels.user`
5. Create API key
6. Get Model URI for the required model

```bash
# Example of getting Model URI
yc ai language-models list
```

#### OpenAI API

1. Register at [OpenAI](https://platform.openai.com/)
2. Go to API Keys section
3. Create new API key
4. Copy key to `.env` file

### 6. Configure Local Models (optional)

If you want to use local models through Ollama:

1. **Download models:**
```bash
# Create models folder
mkdir -p /path/to/models

# Download required models in GGUF format
# Examples:
# - Qwen 2.5 7B: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF
# - Llama 3.1 8B: https://huggingface.co/TheBloke/Llama-3.1-8B-Instruct-GGUF
```

2. **Configure path in docker-compose.yml:**
```yaml
ollama:
  volumes:
    - /path/to/your/models:/models:ro
```

## Installation Verification

### 1. Docker Check

```bash
# Check Docker
docker --version
docker-compose --version

# Check NVIDIA Container Toolkit (if installed)
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### 2. Python and Poetry Check

```bash
# Check Python
python3 --version

# Check Poetry
poetry --version
```

### 3. Environment Variables Check

```bash
# Check .env file loading
source .env && echo "MongoDB Username: $MONGO_INITDB_ROOT_USERNAME"
```

## First Launch

### 1. Launch with Docker Compose

```bash
# Build and start all services
docker-compose up -d --build

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### 2. Check Service Availability

```bash
# Check MongoDB
curl -f http://localhost:27017/ || echo "MongoDB unavailable"

# Check Backend
curl -f http://localhost:45321/health || echo "Backend unavailable"

# Check Frontend
curl -f http://localhost:27366/ || echo "Frontend unavailable"

# Check Ollama (if used)
curl -f http://localhost:12345/api/tags || echo "Ollama unavailable"
```

### 3. Access Web Interfaces

- **Monitoring**: http://localhost:27366 (or http://83.143.66.61:27366 for remote access)
- **MongoDB Express**: http://localhost:8081

**Authentication credentials for monitoring interface:**
- Username: `user`
- Password: `resu123`

## Authentication

### Default Credentials

The TrustVar monitoring interface is protected by authentication. Use the following credentials to access the system:

- **Username**: `user`
- **Password**: `resu123`

### Access URLs

- **Local access**: http://localhost:27366
- **Remote access**: http://83.143.66.61:27366

### Security Note

For production deployments, it's recommended to:
1. Change the default credentials
2. Use HTTPS for secure communication
3. Implement proper user management
4. Configure session timeouts

## Local Development

### 1. Install Dependencies

```bash
# Install dependencies via Poetry
poetry install

# Activate virtual environment
poetry shell
```

### 2. Configure MongoDB

**Option A: MongoDB in Docker**
```bash
docker run -d \
  --name mongodb \
  -p 27017:27017 \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=password \
  mongo:latest
```

**Option B: Local MongoDB Installation**
```bash
# Ubuntu/Debian
sudo apt-get install mongodb

# macOS
brew install mongodb-community

# Start
sudo systemctl start mongod  # Linux
brew services start mongodb-community  # macOS
```

### 3. Launch Components

```bash
# Backend
python langchain_back/main.py

# Frontend (in separate terminal)
streamlit run monitoring/app_main.py --server.port 27366

# Runners (in separate terminals)
python -m runners.run
python -m runners.run_metrics
python -m runners.task_processor
```

## Troubleshooting

### Common Issues

#### 1. Docker Won't Start

```bash
# Check Docker status
sudo systemctl status docker

# Start Docker
sudo systemctl start docker

# Check user permissions
groups $USER
```

#### 2. GPU Issues

```bash
# Check NVIDIA drivers
nvidia-smi

# Check CUDA
nvcc --version

# Check NVIDIA Container Toolkit
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

#### 3. Port Issues

```bash
# Check occupied ports
netstat -tlnp | grep -E ':(27017|45321|27366|12345)'

# Stop services on occupied ports
sudo lsof -ti:27017 | xargs kill -9
```

#### 4. Permission Issues

```bash
# Change folder permissions
sudo chown -R $USER:$USER .

# Change Docker volumes permissions
sudo chown -R $USER:$USER /var/lib/docker/volumes/
```

### Logs and Diagnostics

```bash
# All services logs
docker-compose logs -f

# Specific service logs
docker-compose logs -f langchain_backend

# Check resource usage
docker stats

# Check network connections
docker network ls
docker network inspect trustvar_default
```

## Security

### Security Recommendations

1. **Change default passwords**
2. **Use HTTPS in production**
3. **Configure firewall**
4. **Regularly update dependencies**
5. **Use secrets management**
6. **Configure security monitoring**

### Firewall Configuration

```bash
# Ubuntu/Debian
sudo ufw allow 27017/tcp  # MongoDB
sudo ufw allow 45321/tcp  # Backend
sudo ufw allow 27366/tcp  # Frontend
sudo ufw allow 12345/tcp  # Ollama
sudo ufw enable
```

### SSL/TLS Configuration

For production, SSL/TLS configuration is recommended:

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:27366;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Next Steps

After successful setup:

1. **Study documentation**: [docs/README.md](docs/README.md)
2. **Create first task** through web interface
3. **Configure monitoring** and alerts
4. **Optimize performance** for your needs
5. **Configure data backup** 