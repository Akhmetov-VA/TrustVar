# TrustVar Documentation Index

Complete index of all TrustVar documentation organized by category and topic.

## Quick Start

- **[Main README](../README.md)** - Project overview and quick start
- **[Documentation Overview](overview.md)** - Navigation guide and learning paths
- **[Setup Guide](setup.md)** - Installation and configuration
- **[Environment Template](env.example)** - Configuration file template

## Main Documentation

### Core Documentation
- **[README](README.md)** - Comprehensive system overview and architecture
- **[Components](components.md)** - Detailed description of all system components
- **[API Reference](api.md)** - Complete API documentation with examples
- **[Metrics Guide](metrics.md)** - Metrics calculation and analysis

### Deployment & Operations
- **[Deployment Guide](deployment.md)** - Docker, Kubernetes, and production setup
- **[Setup Instructions](setup.md)** - Step-by-step installation guide

## Configuration Files

### Environment & Settings
- **[env.example](env.example)** - Environment variables template
- **[pyproject.toml](../pyproject.toml)** - Poetry configuration
- **[docker-compose.yml](../docker-compose.yml)** - Docker services configuration
- **[Dockerfile](../Dockerfile)** - Docker image definition

## User Role Navigation

### 🎯 For Researchers
- **[Metrics Guide](metrics.md)** - Understanding and analyzing metrics
- **[API Reference](api.md)** - Programmatic access to system
- **[Components](components.md)** - Understanding system architecture

### 🔧 For Developers
- **[API Reference](api.md)** - Integration and development
- **[Components](components.md)** - System architecture and customization
- **[Setup Guide](setup.md)** - Development environment setup

### 🚀 For System Administrators
- **[Deployment Guide](deployment.md)** - Production deployment
- **[Setup Guide](setup.md)** - System configuration
- **[Components](components.md)** - Monitoring and maintenance

### 📊 For Data Scientists
- **[Metrics Guide](metrics.md)** - Metrics analysis and interpretation
- **[API Reference](api.md)** - Data extraction and analysis
- **[Components](components.md)** - Understanding data flow

## Topic-Based Search

### Architecture & Design
- **[Main README](README.md)** - System overview and architecture
- **[Components](components.md)** - Detailed component descriptions
- **[Project Structure](../README.md#project-structure)** - File organization

### Setup & Installation
- **[Setup Guide](setup.md)** - Complete installation instructions
- **[Environment Template](env.example)** - Configuration template
- **[Quick Start](../README.md#quick-start)** - Minimal setup guide

### API & Integration
- **[API Reference](api.md)** - Complete API documentation
- **[Examples](api.md#examples)** - Code examples in multiple languages
- **[Error Handling](api.md#error-handling)** - API error codes and responses

### Deployment & Operations
- **[Deployment Guide](deployment.md)** - Production deployment options
- **[Docker Setup](deployment.md#docker-compose)** - Containerized deployment
- **[Kubernetes](deployment.md#kubernetes)** - Scalable deployments
- **[Monitoring](deployment.md#monitoring-and-maintenance)** - System monitoring

### Metrics & Analysis
- **[Metrics Guide](metrics.md)** - Metrics types and calculations
- **[Task Sensitivity Index](metrics.md#task-sensitivity-index-tsi)** - TSI metric details
- **[Visualization](metrics.md#visualization)** - Charts and graphs
- **[Export](metrics.md#export)** - Data export options

### Security & Configuration
- **[Security](deployment.md#security)** - Security recommendations
- **[Authentication](components.md#streamlit-frontend)** - User authentication
- **[Environment Variables](setup.md#environment-variables)** - Configuration options

## Component-Specific Documentation

### Database
- **[MongoDB](components.md#mongodb)** - Database structure and collections
- **[Data Models](components.md#data-structures)** - Document schemas

### Backend Services
- **[Langchain Backend](components.md#langchain-backend)** - API server
- **[Task Runners](components.md#task-runners)** - Background processors
- **[Utils](components.md#utils)** - Common utilities

### Frontend
- **[Streamlit Frontend](components.md#streamlit-frontend)** - Web interface
- **[Monitoring](components.md#monitoring)** - Real-time monitoring

### External Services
- **[Ollama](components.md#ollama)** - Local LLM service
- **[Model Providers](components.md#supported-models)** - API integrations

## Troubleshooting & Support

### Common Issues
- **[Setup Troubleshooting](setup.md#troubleshooting)** - Installation problems
- **[API Issues](api.md#error-handling)** - API error resolution
- **[Deployment Issues](deployment.md#troubleshooting)** - Deployment problems

### Getting Help
- **[Support](../README.md#support)** - How to get help
- **[Logs](deployment.md#monitoring-and-maintenance)** - System logging
- **[Issue Reporting](../README.md#support)** - Bug reports and feature requests

## Development Resources

### Code Structure
- **[Project Structure](../README.md#project-structure)** - File organization
- **[Adding Metrics](components.md#adding-new-metrics)** - Custom metrics
- **[Adding Models](components.md#adding-new-models)** - New model support

### External Resources
- **[Docker Documentation](https://docs.docker.com/)** - Container technology
- **[MongoDB Documentation](https://docs.mongodb.com/)** - Database
- **[Streamlit Documentation](https://docs.streamlit.io/)** - Web interface
- **[FastAPI Documentation](https://fastapi.tiangolo.com/)** - API framework
- **[Poetry Documentation](https://python-poetry.org/docs/)** - Dependency management

## Dataset Resources

### Download Datasets
- **[Google Drive Datasets](https://drive.google.com/drive/folders/1jvBWvAc9JcjLYQ8T09xoDKUkwjCf7tiI?usp=sharing)** - All project datasets
- **[Dataset Instructions](../README.md#download-datasets-and-auxiliary-information)** - Setup instructions

### Dataset Types
- **Accuracy Metrics**: Accuracy_Groups.json, Accuracy.json
- **Correlation Analysis**: Correlation.json
- **RtA Analysis**: RtAR.json, TFNR.json
- **Safety Testing**: jailbreak.json, stereotypes_detection_3.json
- **Privacy Assessment**: privacy_assessment.json
- **OOD Detection**: ood_detection.json

---

**Need help finding something?** Use the search function in your browser or check the [Documentation Overview](overview.md) for guided navigation. 