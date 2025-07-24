# TrustVar Documentation Overview

Welcome to the TrustVar documentation! This guide will help you navigate through the comprehensive documentation and understand the framework's capabilities.

## What is TrustVar?

**TrustVar** is a framework built on our previous LLM trustworthiness testing system. While we previously focused on how LLMs handle tasks, we now rethink the evaluation procedure itself. TrustVar shifts the focus: we investigate the quality of tasks themselves, not just model behavior.

### Key Innovation

Unlike traditional frameworks that test models through tasks, TrustVar tests tasks through models. We analyze tasks as research objects, measuring their ambiguity, sensitivity, and structure, then examine how these parameters influence model behavior.

## Documentation Structure

### 📚 Main Documentation
- **[README.md](README.md)** - Comprehensive system overview and architecture
- **[components.md](components.md)** - Detailed description of all system components
- **[setup.md](setup.md)** - Setup and installation guide

### 🚀 Deployment & Operations
- **[deployment.md](deployment.md)** - Deployment guides (Docker, Kubernetes)
- **[api.md](api.md)** - Complete API documentation
- **[metrics.md](metrics.md)** - Metrics calculation and analysis

### 📋 Configuration & Reference
- **[env.example](env.example)** - Environment variables template
- **[index.md](index.md)** - Complete documentation index

## Learning Paths

### 🎯 For New Users
1. Start with **[README.md](README.md)** for system overview
2. Follow **[setup.md](setup.md)** for installation
3. Explore **[components.md](components.md)** to understand architecture

### 🔧 For Developers
1. Review **[api.md](api.md)** for integration
2. Study **[metrics.md](metrics.md)** for custom metrics
3. Check **[deployment.md](deployment.md)** for production setup

### 🚀 For DevOps
1. Focus on **[deployment.md](deployment.md)**
2. Review **[setup.md](setup.md)** for configuration
3. Understand **[components.md](components.md)** for monitoring

## Core Concepts

### Task Variation Analysis
- **Task Sensitivity Index (TSI)**: Measures how strongly formulations affect model success
- **Coefficient of Variation (CV)**: Quantifies model behavior changes between task versions
- **Radar Diagrams**: Visual representation of task stability across models

### Multi-language Support
- English and Russian tasks with extensible architecture
- Language-independent analysis pipeline
- Custom dataset support

### Interactive Pipeline
- Unified system for data loading, task generation, variation, model evaluation, and visual analysis
- Real-time monitoring and visualization
- Automated metrics calculation

## Quick Navigation

### 🔍 Search by Topic
- **Architecture**: [README.md](README.md), [components.md](components.md)
- **Setup**: [setup.md](setup.md), [env.example](env.example)
- **API**: [api.md](api.md)
- **Deployment**: [deployment.md](deployment.md)
- **Metrics**: [metrics.md](metrics.md)

### 📖 Search by User Role
- **Researchers**: [metrics.md](metrics.md), [api.md](api.md)
- **Developers**: [components.md](components.md), [api.md](api.md)
- **System Administrators**: [deployment.md](deployment.md), [setup.md](setup.md)
- **End Users**: [README.md](README.md), [setup.md](setup.md)

## Getting Help

If you need assistance:
1. Check the relevant documentation section
2. Review the [index.md](index.md) for specific topics
3. Create an issue in the project repository

## Contributing

We welcome contributions to improve the documentation:
1. Fork the repository
2. Make your changes
3. Submit a pull request

---

**Ready to get started?** Begin with the [Main Documentation](README.md) or jump directly to [Setup Guide](setup.md). 