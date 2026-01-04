# Stock Analysis Multi-Agent System

A production-ready, LLM-powered multi-agent system for intelligent stock analysis and trading recommendations.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🌟 Overview

This system implements a sophisticated multi-agent architecture for stock market analysis, combining:

- **3 Specialized Junior Agents:** News Sentiment, Technical Analysis, Fundamental Analysis
- **1 Senior Strategist Agent:** Synthesizes junior agent outputs into actionable trading decisions
- **1 Supervisor Agent:** Intelligent routing for optimal agent selection (optional)
- **LLM Judge System:** Automated evaluation and continuous improvement
- **Complete Training Pipeline:** SFT, GRPO/PPO, and online learning

### Key Features

✅ **Multi-Agent Architecture** - Specialized agents for different analysis types  
✅ **LangGraph Workflow** - State-based orchestration with memory  
✅ **LLM Judge System** - Automated quality evaluation  
✅ **Complete Training Pipeline** - SFT → RL → Online Learning  
✅ **REST API** - Production-ready FastAPI server  
✅ **Comprehensive Testing** - 53 unit + integration tests  
✅ **Docker Support** - Containerized deployment  
✅ **Extensive Documentation** - Architecture, training, API, deployment guides

## 📋 Table of Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Deployment](#deployment)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    System Coordinator                        │
│                   (LangGraph Workflow)                       │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
         ┌──────────▼──────────┐  ┌────▼─────────┐
         │  Supervisor Agent   │  │  Data Layer  │
         │  (Neural-UCB)       │  │              │
         └──────────┬──────────┘  └──────────────┘
                    │
         ┌──────────┴──────────────────────┐
         │                                  │
┌────────▼────────┐  ┌──────────────────┐  │
│  Junior Agents  │  │ Senior Strategist│  │
│  - News         │  │  Agent           │  │
│  - Technical    │  │                  │  │
│  - Fundamental  │  │                  │  │
└─────────────────┘  └──────────────────┘  │
                                            │
                    ┌───────────────────────▼──┐
                    │    LLM Judge System      │
                    │  (Evaluation & Feedback) │
                    └──────────────────────────┘
```

### Agent Hierarchy

1. **Junior Agents** (Specialized Analysis)
   - **News Sentiment Agent:** Analyzes news articles, social media, earnings calls
   - **Technical Analysis Agent:** Chart patterns, indicators, price action
   - **Fundamental Analysis Agent:** Financial statements, valuation metrics

2. **Senior Strategist Agent** (Decision Making)
   - Synthesizes junior agent outputs
   - Makes final trading decisions
   - Manages risk and position sizing

3. **Supervisor Agent** (Intelligent Routing)
   - Contextual bandit approach
   - Selects optimal agent combinations
   - Reduces computational cost

4. **LLM Judge System** (Quality Assurance)
   - Evaluates agent outputs
   - Provides feedback for training
   - Enables continuous improvement

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- 16GB+ RAM (32GB recommended)
- GPU with 8GB+ VRAM (optional but recommended)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/stock-agent-system-final.git
cd stock-agent-system-final

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export ANTHROPIC_API_KEY=your_key_here
```

### Run API Server

```bash
# Start server
python -m uvicorn api.server:app --reload --host 0.0.0.0 --port 8000

# Test endpoint
curl http://localhost:8000/health
```

### Analyze a Stock

```python
import requests

response = requests.post(
    'http://localhost:8000/analyze',
    json={
        'symbol': 'AAPL',
        'use_supervisor': False,
        'lookback_days': 7
    }
)

result = response.json()
print(f"Recommendation: {result['recommendation']}")
print(f"Confidence: {result['confidence']}")
print(f"Position Size: {result['position_size']}")
```

## 📦 Installation

### From Source

```bash
# Clone repository
git clone https://github.com/yourusername/stock-agent-system-final.git
cd stock-agent-system-final

# Install in development mode
pip install -e .
```

### Using Docker

```bash
# Build image
docker build -t stock-agent-system:latest .

# Run container
docker run -d \
  --name stock-agent-api \
  -p 8000:8000 \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  stock-agent-system:latest
```

### Using Docker Compose

```bash
docker-compose up -d
```

## 💻 Usage

### Command Line Interface

```bash
# Analyze single symbol
python -m orchestration.coordinator analyze AAPL

# Batch analysis
python -m orchestration.coordinator batch AAPL MSFT GOOGL

# With supervisor
python -m orchestration.coordinator analyze AAPL --use-supervisor
```

### Python API

```python
from orchestration.coordinator import SystemCoordinator

# Initialize coordinator
coordinator = SystemCoordinator(config_path='config/system.yaml')

# Analyze symbol
result = coordinator.analyze_symbol('AAPL', use_supervisor=False)

print(f"Recommendation: {result['recommendation']}")
print(f"Confidence: {result['confidence']}")
print(f"Reasoning: {result['reasoning']}")
```

### REST API

See [API Documentation](docs/API_DOCUMENTATION.md) for complete API reference.

**Endpoints:**

- `GET /health` - Health check
- `GET /models` - Model information
- `POST /analyze` - Single symbol analysis
- `POST /batch` - Batch analysis
- `POST /backtest` - Historical backtesting

## 🎓 Training

### Phase 1: Supervised Fine-Tuning (SFT)

```bash
# Train News Agent
python training/sft/train_news_agent.py \
  --config config/sft/news_agent.yaml

# Train Technical Agent
python training/sft/train_technical_agent.py \
  --config config/sft/technical_agent.yaml

# Train Fundamental Agent
python training/sft/train_fundamental_agent.py \
  --config config/sft/fundamental_agent.yaml
```

### Phase 2: Reinforcement Learning

#### Option A: GRPO (Memory Efficient)

```bash
python training/rl/train_strategist_grpo.py \
  --config config/rl/grpo_config.yaml
```

#### Option B: PPO (Better Performance)

```bash
python training/rl/train_strategist_ppo.py \
  --config config/rl/ppo_config.yaml
```

### Phase 3: Supervisor Training

```bash
python training/supervisor/train_supervisor.py \
  --config config/supervisor/neural_ucb.yaml \
  --episodes 1000
```

### Phase 4: Online Learning

```bash
# Generate synthetic data
python scripts/generate_synthetic_data.py \
  --num-examples 1000 \
  --output data/synthetic_trajectories.jsonl

# Re-train with experience library
python training/data_synthesis/synthesize_trajectories.py \
  --db data/experience_library.db \
  --output data/refined_trajectories.jsonl
```

See [Training Guide](docs/TRAINING.md) for detailed instructions.

## 📚 API Reference

### Single Analysis

```python
POST /analyze
{
  "symbol": "AAPL",
  "use_supervisor": false,
  "lookback_days": 7
}
```

**Response:**

```json
{
  "symbol": "AAPL",
  "recommendation": "buy",
  "confidence": 0.85,
  "position_size": 0.08,
  "entry_target": 185.50,
  "stop_loss": 178.00,
  "take_profit": 195.00,
  "reasoning": "Strong bullish signals...",
  "risk_assessment": "Moderate risk...",
  "agent_outputs": {...},
  "timestamp": "2024-01-04T12:00:00"
}
```

See [API Documentation](docs/API_DOCUMENTATION.md) for complete reference.

## 🧪 Testing

### Run All Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test suite
pytest tests/unit/
pytest tests/integration/
```

### Test Coverage

- **Unit Tests:** 39 test cases
- **Integration Tests:** 14 test cases
- **Total Coverage:** 53 test cases

See [Testing Guide](docs/TESTING.md) for detailed information.

## 🚢 Deployment

### Docker Deployment

```bash
# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f
```

### Cloud Deployment

#### AWS ECS

```bash
# Push to ECR
docker tag stock-agent-system:latest your-ecr-repo/stock-agent-system:latest
docker push your-ecr-repo/stock-agent-system:latest

# Deploy to ECS
aws ecs update-service \
  --cluster stock-agent-cluster \
  --service stock-agent-service \
  --force-new-deployment
```

#### GCP Cloud Run

```bash
gcloud run deploy stock-agent-api \
  --image gcr.io/your-project/stock-agent-system \
  --platform managed \
  --region us-central1
```

See [Deployment Guide](docs/DEPLOYMENT.md) for complete instructions.

## 📖 Documentation

- [Architecture Overview](docs/ARCHITECTURE.md)
- [Training Guide](docs/TRAINING.md)
- [API Documentation](docs/API_DOCUMENTATION.md)
- [Testing Guide](docs/TESTING.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Quick Start](QUICKSTART.md)
- [Project Summary](PROJECT_SUMMARY.md)

## 🗂️ Project Structure

```
stock-agent-system-final/
├── agents/                  # Agent implementations
│   ├── junior/             # News, Technical, Fundamental agents
│   ├── senior/             # Strategist agent
│   └── supervisor/         # Supervisor agent
├── api/                    # FastAPI REST API
│   ├── server.py
│   └── schemas.py
├── config/                 # Configuration files
│   ├── sft/               # SFT training configs
│   ├── rl/                # RL training configs
│   └── supervisor/        # Supervisor configs
├── data/                   # Data storage
├── docs/                   # Documentation
├── judge/                  # LLM Judge system
├── orchestration/          # Workflow orchestration
│   ├── coordinator.py
│   └── langgraph_workflow.py
├── scripts/                # Utility scripts
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── training/               # Training pipelines
│   ├── sft/               # Supervised fine-tuning
│   ├── rl/                # Reinforcement learning
│   ├── supervisor/        # Supervisor training
│   └── data_synthesis/    # Data generation
└── utils/                  # Utility functions
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests before committing
pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Anthropic** - Claude API for LLM Judge
- **Meta** - Llama models for agent implementation
- **LangChain/LangGraph** - Workflow orchestration
- **Unsloth** - Efficient fine-tuning
- **FastAPI** - REST API framework

## 📧 Contact

- **GitHub Issues:** [repository]/issues
- **Email:** support@example.com
- **Documentation:** [repository]/docs

## 🔄 Changelog

### Version 1.0.0 (2024-01-04)

- ✅ Complete multi-agent architecture
- ✅ LangGraph workflow integration
- ✅ Full training pipeline (SFT + RL)
- ✅ REST API with FastAPI
- ✅ Comprehensive test suite
- ✅ Docker deployment support
- ✅ Complete documentation

---

**Built with ❤️ for intelligent stock analysis**
