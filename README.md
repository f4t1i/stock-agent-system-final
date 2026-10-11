# Stock Agent System

**Multi-Agent Stock Analysis Research Prototype with Advanced ML**

An experimental AI system combining multi-agent intelligence, deep learning, reinforcement learning, real-time analytics, explainability, and risk management for stock market analysis.

> [!WARNING]
> **Research prototype — not production-ready, not for real-money trading.**
> The full pipeline does not run out of the box (fine-tuned agent models are not included),
> the default backtester has look-ahead bias, and there is no evidence yet that the system beats simple baselines.
> Read **[Project Status & Honest Assessment](docs/PROJECT_STATUS.md)** and the
> **[How to Use This Repository](#how-to-use-this-repository)** section before you start.

[![Version](https://img.shields.io/badge/version-2.1.0-green.svg)](https://github.com/f4t1i/stock-agent-system-final/releases)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![React](https://img.shields.io/badge/react-19-blue.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/typescript-5.0+-blue.svg)](https://www.typescriptlang.org/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-research%20prototype-orange.svg)](docs/PROJECT_STATUS.md)

---

## System Overview

```
~100,000 Lines of Code (≈78k Python, ≈23k TypeScript)
Test suite partially passing (see Project Status)
4 LLM Agents + Supervisor + 9 ML Systems
Full-Stack Real-Time Dashboard
Advanced Visualizations (8 Chart Types)
Explainability & Confidence Calibration  <- Phase A3
Risk Management & Trading Guardrails     <- Phase A3
Alerts & Watchlist Monitoring            <- Phase A3
~150 Git Commits | v2.1.0
```

### What This System Does

**Stock Agent System** is a research platform for multi-agent trading analysis that combines:

- **Multi-Agent AI**: 4 specialized agents (News/Sentiment, Technical, Fundamental, Strategist) plus a routing Supervisor
- **Advanced ML**: Ensemble models, Deep Learning (LSTM/Transformer), RL (DQN), AutoML
- **Real-Time System**: WebSocket streaming, live charts, sub-second updates
- **Explainability**: AI decision transparency with reasoning visualization and confidence gauges
- **Risk Management**: Multi-layer safety gates, trading policies, and position validation
- **Alerts & Watchlists**: Real-time price monitoring with multi-channel notifications
- **Confidence Calibration**: Isotonic regression-based calibration with reliability diagrams
- **Advanced Visualizations**: 8 interactive chart types with technical indicators
- **Enterprise Database**: PostgreSQL with 8 ORM models for full persistence
- **Market Intelligence**: OpenBB integration, anomaly detection, feature engineering

---

## Project Status

| Area | Rating | Summary |
|------|--------|---------|
| Concept / architecture | 8/10 | Strong design: junior agents → strategist, bandit supervisor, LLM judge for SFT/RL |
| Scope | 9/10 | Training, risk, calibration, explainability, alerts, dashboard |
| End-to-end runnability | 2/10 | Agents load fine-tuned models (`models/*_v1`) that are **not in the repo** → `main.py` and the API server fail on startup |
| Backtest validity | 2/10 | `training/rl/backtester.py` analyses each historical day with **today's** data (look-ahead bias) |
| Production readiness | 1–2/10 | Hard-coded regime/volatility, no broker execution, no CI |

**Known gaps** (details and file references in [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md)):

- Market regime (`'bull'`) and volatility (`20.0`) are hard-coded in `orchestration/coordinator.py`; trade execution is a TODO.
- `agents/supervisor_v2.py` uses a random reward in its training loop.
- DeepMaster / Epistemic Shield / Memory (`deepmaster_*.py`, `epistemic_shield.py`, `memory_knowledge/`) are **not wired into** the main system.
- The `web-dashboard/` is a separate Node/tRPC app and is not connected to the Python API.
- Offline test run with lightweight dependencies: **118 passed, 10 failed, 56 collection errors** (missing `torch`/`pandas_ta`, no network, no `pytest-asyncio`).

---

## How to Use This Repository

Pick the path that matches your goal.

### A) Explore the architecture (no GPU, no API keys)

1. Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), then `orchestration/coordinator.py` (how agents are wired).
2. Read one agent, e.g. `agents/junior/news_agent.py` (prompt → JSON → parsed result).
3. Run the tests that work offline:
   ```bash
   python -m venv venv && source venv/bin/activate
   pip install pytest loguru python-dotenv pyyaml scipy scikit-learn \
               yfinance anthropic fastapi httpx pydantic sqlalchemy
   pytest -q tests/unit/test_error_taxonomy.py \
             tests/acceptance/test_phase_a3_complete.py \
             tests/acceptance/test_corporate_actions.py \
             tests/acceptance/test_survivorship_bias.py \
             test_deepmaster_orchestration.py
   ```

### B) Reuse individual components (works today)

These modules can be imported without the LLM agents:

| Module | Use |
|--------|-----|
| `risk_management/` | Risk gates, trading policies, position validation |
| `calibration/confidence_calibrator.py` | Isotonic-regression confidence calibration |
| `training/rl/backtester_v2.py` | Backtester with commission, slippage, corporate actions, survivorship-bias handling |
| `utils/market_regime_detector.py` | Bull/bear/sideways regime detection |
| `ml/` | Feature engineering, anomaly detection, ensembles, LSTM/Transformer |

### C) Make the full agent system run (contributors)

Recommended order:

1. **Runnable path first** – replace local `from_pretrained` loading in the agents with an API-based LLM backend (Claude/OpenAI) behind a common interface, so no fine-tuned checkpoints are needed.
2. **Fix the backtest** – add an `as_of` date to `analyze_symbol`, feed agents point-in-time prices and date-filtered news, and use `backtester_v2`.
3. **Compare against baselines** – buy & hold and SMA crossover on an out-of-sample period.
4. **Remove placeholders** – real regime/volatility, real reward signal for `supervisor_v2`.
5. **Decide on DeepMaster** – integrate into the coordinator or move to `experimental/`.
6. **Housekeeping** – CI with offline tests, split requirements (core / training / dashboard), consolidate root status files.

Only after step 3 does it make sense to fine-tune agents (`training/`) and to measure whether they add value.

### Do not

- Trade real money based on this system's output.
- Trust numbers from `python main.py --mode backtest` (look-ahead bias).
- Expect `docker-compose up` or the Quick Start below to give a working analysis – both hit the missing-model problem until step C.1 is done.

---

## Core Capabilities

### 1. Multi-Agent Intelligence (~8,200 Lines)

**5 Specialized AI Agents**:
- **News Agent** — Sentiment analysis from news & social media
- **Technical Agent** — Chart patterns, indicators (RSI, MACD, Bollinger)
- **Fundamental Agent** — Financial statement analysis, valuation
- **Sentiment Agent** — Market sentiment & social trends
- **Strategist Agent** — Meta-analysis & final decision synthesis

**Features**:
- Hierarchical decision-making with LangGraph workflow
- Agent-to-Agent communication and consensus
- Supervisor v2 with Contextual Bandit routing (Neural-UCB)
- Regime-specific models (Bull/Bear/Sideways market adaptation)
- Confidence scoring & calibration

### 2. Advanced Training Infrastructure (~12,500 Lines)

**4 Training Pipelines**:
- **Supervised Learning**: LoRA/QLoRA fine-tuning with judge feedback
- **Reinforcement Learning**: GRPO/PPO with experience replay
- **Self-Training**: Automatic label generation
- **Judge System**: Quality control & automated evaluation

**Features**:
- Experience store (10,000+ records)
- Model versioning & registry
- A/B testing framework
- Multi-iteration training with convergence tracking

### 3. Explainability System (~1,200 Lines) — Phase A3

**Backend Modules**:
- `agents/reasoning_extractor.py` — Extract and parse reasoning from agent outputs
- `agents/decision_logger.py` — Audit trail for all AI decisions
- `api/explainability.py` — REST API with 3 explainability endpoints

**Frontend Components**:
- `ExplainabilityCard.tsx` — Decision explanation with confidence meter
- `ReasoningVisualization.tsx` — Bar charts, confidence timeline, decision tree
- `ConfidenceGauge.tsx` — Visual confidence indicator

**Features**:
- Decision reasoning extraction with factor weighting
- Confidence score visualization
- Historical decision audit log
- Agent contribution breakdown
- REST API: `GET /explain/{decision_id}`, `POST /explain/analyze`

### 4. Alerts & Watchlists (~1,800 Lines) — Phase A3

**Backend Modules**:
- `monitoring/alert_evaluator.py` — Condition matching (price, confidence, RSI, MACD)
- `monitoring/notification_dispatcher.py` — Email, push, webhook delivery
- `monitoring/watchlist_monitor.py` — Background scheduler (every 5 min)
- `api/alerts.py` — CRUD API for alerts management
- `api/watchlist.py` — Watchlist management endpoints

**Frontend Components**:
- `AlertsPanel.tsx` — Alert list with status indicators
- `AlertForm.tsx` — Create/edit alert dialog
- `WatchlistManager.tsx` — Symbol tracking and management
- `NotificationCenter.tsx` — Notification history and preferences

**Alert Types**:
- Price threshold (above/below target)
- Confidence change (increase/decrease)
- Recommendation change (Buy/Sell/Hold transition)
- Technical signal (RSI overbought/oversold, MACD crossover)

### 5. Risk Management (~2,100 Lines) — Phase A3

**Backend Modules**:
- `risk_management/risk_gates.py` — 5 core risk checks (430 lines)
- `risk_management/risk_engine.py` — Central risk evaluation engine (370 lines)
- `risk_management/trading_policies.py` — Policy definition and enforcement
- `risk_management/position_validator.py` — Pre-trade position validation
- `risk_management/policy_evaluator.py` — Custom rule evaluation

**Frontend Components**:
- `RiskPanel.tsx` — Risk dashboard with visual indicators
- `PolicyEditor.tsx` — YAML-based policy configuration

**Risk Gates**:

| Gate | Description | Default Limit |
|------|-------------|---------------|
| Position Size | Max % per symbol | 10% |
| Daily Loss | Max daily drawdown | 2% |
| Concentration | Max sector exposure | 30% |
| Leverage | Max portfolio leverage | 2x |
| Volatility | Block high-volatility trades | 3x avg |

**Trading Policy Templates**: Conservative · Moderate · Aggressive

### 6. Confidence Calibration (~800 Lines) — Phase A3

**Backend Modules**:
- `calibration/confidence_calibrator.py` — Isotonic regression calibration (231 lines)

**Frontend Components**:
- `CalibrationDashboard.tsx` — Calibration metrics and reliability diagram

**Metrics**: ECE · MCE · Brier Score · Reliability Diagram · Auto-recalibration

### 7. Advanced ML Systems (~6,500 Lines)

| Module | Lines | Description |
|--------|-------|-------------|
| Ensemble Models | 750 | Stacking, voting, bagging, adaptive |
| Deep Learning | 670 | LSTM, Transformer, TCN |
| Reinforcement Learning | 610 | DQN trading agent |
| AutoML | 370 | Optuna hyperparameter optimization |
| Model Explainability | 480 | SHAP & LIME |
| Feature Engineering | 520 | 60+ automated features |
| Anomaly Detection | 400 | 4 detection methods |

### 8. Real-Time System (~1,285 Lines)

**WebSocket Channels**: prices · alerts · signals · portfolio · notifications

**Features**: Auto-reconnect · Ping/pong keep-alive · Sub-second latency · 100+ concurrent clients

### 9. Advanced Visualizations (~6,193 Lines)

**8 Chart Components**: CandlestickChart · LineChart · AreaChart · Heatmap · RealTimeChart · TechnicalChart · PortfolioAnalytics · DashboardGrid

### 10. Enterprise Database (~1,170 Lines)

**8 ORM Models**: Analysis · TrainingRun · ModelVersion · ExperienceRecord · Alert · Watchlist · Decision · RiskViolation

**Dual Support**: SQLite (dev) + PostgreSQL (prod) with connection pooling

### 11. Market Data Integration (~630 Lines)

**OpenBB Platform**: 7 REST endpoints · Natural language queries · WebSocket streaming

---

## Architecture

```
+----------------------------------------------------------------------+
|                    Web Dashboard (React 19 + TypeScript)              |
|  Explainability | Alerts | Risk Management | Charts | Portfolio       |
+----------------------------------------------------------------------+
                                     | REST / WebSocket
+----------------------------------------------------------------------+
|                        FastAPI Backend (Python)                       |
|  /analyze | /explain | /alerts | /risk | /watchlist | /ws            |
+----------------------------------------------------------------------+
          |                          |                       |
+---------+----------+  +-----------+----------+  +--------+---------+
|  Supervisor Agent  |  |  Risk Management     |  |  Monitoring      |
|  (Neural-UCB)      |  |  - Risk Gates (5)    |  |  - Alert Eval    |
|  Regime Routing    |  |  - Trading Policies  |  |  - Watchlists    |
+--------------------+  +----------------------+  +------------------+
          |
+---------+-----------------------------------------------------------+
|                          Agent Layer                                 |
|   News Agent | Technical Agent | Fundamental Agent | Strategist     |
+---------------------------------------------------------------------+
          |
+---------+-----------------------------------------------------------+
|                   Explainability & Calibration                       |
|   Reasoning Extractor | Decision Logger | Confidence Calibrator     |
+---------------------------------------------------------------------+
          |
+---------+-----------------------------------------------------------+
|                    ML & Training Infrastructure                      |
|   GRPO Trainer | Ensemble | Deep Learning | RL (DQN) | AutoML       |
+---------------------------------------------------------------------+
          |
+---------+-----------------------------------------------------------+
|                       LLM Judge System                               |
|   Quality Evaluation | Reward Signals | Continuous Improvement      |
+---------------------------------------------------------------------+
```

---

## Quick Start

> [!NOTE]
> The commands below describe the **intended** setup. Until trained agent models exist in `models/`
> (or an API-based LLM backend is added, see [How to Use This Repository](#how-to-use-this-repository)),
> the API server fails during startup when `SystemCoordinator` tries to load them.

### Prerequisites

```bash
Python 3.11+  |  Node.js 18+  |  PostgreSQL 15+ (optional)
8GB RAM  |  20GB Disk  |  NVIDIA GPU (optional, for ML training)
```

### Installation

```bash
# Clone repository
git clone https://github.com/f4t1i/stock-agent-system-final.git
cd stock-agent-system-final

# Backend setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend setup (web dashboard)
cd web-dashboard
pnpm install

# Database setup (optional - uses SQLite by default)
docker-compose -f docker-compose.postgres.yml up -d
```

### Environment Variables

```bash
export ANTHROPIC_API_KEY=your_anthropic_key
export OPENBB_API_KEY=your_openbb_key
export DATABASE_URL=postgresql://stock_user:stock_password@localhost:5432/stock_agent
export VITE_WS_URL=ws://localhost:8000/ws
```

### Start the System

```bash
# Terminal 1: Start API Server
python -m uvicorn api.server:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start Web Dashboard
cd web-dashboard && pnpm dev

# Open http://localhost:5173
```

### Analyze a Stock

```python
import requests

# Basic analysis
response = requests.post('http://localhost:8000/analyze', json={
    'symbol': 'AAPL',
    'use_supervisor': False,
    'lookback_days': 7
})
result = response.json()
print(f"Recommendation: {result['recommendation']}")
print(f"Confidence:     {result['confidence']:.1%}")
print(f"Position Size:  {result['position_size']:.1%}")

# Get explainability for a decision
explain = requests.get(f"http://localhost:8000/explain/{result['decision_id']}")
print(f"Reasoning: {explain.json()['reasoning']}")

# Create a price alert
requests.post('http://localhost:8000/alerts', json={
    'symbol': 'AAPL',
    'alert_type': 'price_above',
    'threshold': 200.0,
    'notification_channels': ['email', 'push']
})

# Validate a trade against risk gates
risk = requests.post('http://localhost:8000/risk/validate', json={
    'symbol': 'AAPL', 'action': 'buy', 'quantity': 100, 'price': 195.0
})
print(f"Risk check: {risk.json()['approved']}")
```

---

## Project Structure

```
stock-agent-system-final/
├── agents/                     # AI Agent implementations
│   ├── junior/                 # News, Technical, Fundamental agents
│   ├── senior/                 # Strategist agent
│   ├── supervisor/             # Supervisor v1 & v2 (Neural-UCB)
│   ├── reasoning_extractor.py  # Decision reasoning extraction
│   └── decision_logger.py      # Audit trail for decisions
├── api/                        # FastAPI REST endpoints
│   ├── server.py               # Main API server
│   ├── explainability.py       # Explainability endpoints
│   ├── alerts.py               # Alert management
│   ├── risk.py                 # Risk assessment
│   ├── watchlist.py            # Watchlist management
│   └── websocket.py            # WebSocket handler
├── calibration/                # Confidence calibration
│   └── confidence_calibrator.py
├── config/                     # Configuration files (YAML)
├── data_pipeline/              # Data ingestion & processing
├── docs/                       # Documentation
├── evaluation/                 # Model evaluation & reporting
├── judge/                      # LLM Judge system
├── ml/                         # Advanced ML modules
│   ├── ensemble/               # Stacking, voting, bagging
│   ├── deep_learning/          # LSTM, Transformer, TCN
│   ├── reinforcement/          # DQN trading agent
│   ├── automl/                 # Optuna optimization
│   ├── explainability/         # SHAP & LIME
│   ├── feature_engineering/    # 60+ automated features
│   └── anomaly_detection/      # 4 detection methods
├── monitoring/                 # Alerts & watchlist monitoring
│   ├── alert_evaluator.py
│   ├── notification_dispatcher.py
│   └── watchlist_monitor.py
├── orchestration/              # LangGraph workflow coordinator
├── risk_management/            # Risk gates & trading policies
│   ├── risk_gates.py           # 5 core risk checks
│   ├── risk_engine.py          # Central risk evaluation
│   ├── trading_policies.py     # Policy definition
│   └── position_validator.py   # Pre-trade validation
├── scripts/                    # Training & utility scripts
├── tests/                      # Test suites
│   ├── acceptance/             # E2E acceptance tests (Phase A3)
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   ├── stress/                 # Load & stress tests
│   └── adversarial/            # Adversarial robustness tests
├── training/                   # Training infrastructure
│   ├── sft/                    # Supervised fine-tuning
│   ├── grpo/                   # GRPO reinforcement learning
│   └── supervisor/             # Supervisor training
├── web-dashboard/              # Full-stack React dashboard
│   ├── client/src/
│   │   ├── components/
│   │   │   ├── explainability/ # ExplainabilityCard, ConfidenceGauge
│   │   │   ├── alerts/         # AlertsPanel, WatchlistManager
│   │   │   ├── risk/           # RiskPanel, PolicyEditor
│   │   │   ├── calibration/    # CalibrationDashboard
│   │   │   ├── charts/         # 8 chart components
│   │   │   └── analytics/      # PortfolioAnalytics
│   │   └── pages/              # Dashboard pages
│   └── server/                 # tRPC backend
├── CHANGELOG.md                # Full version history
├── VERSION                     # Current version
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Testing

```bash
# Run all tests
make test

# Phase A3 acceptance tests
pytest tests/acceptance/test_phase_a3_complete.py -v
```

Many suites need the full ML stack (`torch`, `transformers`, `pandas_ta`) and internet access (`yfinance`).
Last offline run with lightweight dependencies (2026-10-11): **118 passed, 10 failed, 56 collection errors**.
There is no CI yet. See [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md#36-test-suite).

---

## Docker Deployment

```bash
# Full stack with PostgreSQL
docker-compose up -d

# API server only
docker build -t stock-agent-system:latest .
docker run -d -p 8000:8000 -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY stock-agent-system:latest
```

---

## Performance Benchmarks

> Target / design values — not yet backed by reproducible benchmark runs.

| Feature | Latency | Throughput |
|---------|---------|------------|
| Stock Analysis (5 agents) | ~2-5s | 12 req/min |
| Risk Gate Validation | <1ms | 10,000 req/s |
| Alert Evaluation | ~5ms | 1,000 alerts/s |
| WebSocket Updates | <50ms | 100 clients |
| Confidence Calibration | ~10ms | 500 req/s |
| Ensemble Prediction | ~30ms | 200 req/s |
| Deep Learning (LSTM) | ~50ms | 100 req/s |

---

## Roadmap

### v1.0.0 - Foundation (Completed 2026-01-05)
- Multi-agent system (5 agents), training infrastructure (4 pipelines)
- Phase A3: Explainability, Alerts, Risk Management, Confidence Calibration
- Full-stack web dashboard (React 19 + FastAPI)

### v1.1.0 - Real-Time & Persistence (Completed)
- WebSocket streaming, OpenBB integration, PostgreSQL, Docker

### v1.2.0 - Advanced Visualizations (Completed)
- 8 interactive chart types, portfolio analytics, technical analysis

### v2.0.0 - Advanced ML (Completed)
- Ensemble, Deep Learning (LSTM/Transformer), RL (DQN), AutoML, SHAP/LIME

### v2.1.0 - Distributed Training (Completed)
- Distributed multi-GPU training, GANs, Meta-learning (MAML)

### v2.2.0 - Enterprise Features (Q2 2026)
- Multi-user auth, RBAC, mobile app, email/SMS notifications

### v3.0.0 - Scale & Cloud (Q3 2026)
- Kubernetes, AWS/GCP/Azure, options & futures support

---

## Code Statistics

| Category | Lines | Modules |
|----------|-------|---------|
| Multi-Agent System | ~8,200 | 15 |
| Training Infrastructure | ~12,500 | 20 |
| Advanced ML | ~6,500 | 7 |
| Phase A3 (Explainability, Alerts, Risk, Calibration) | ~5,900 | 12 |
| Web Dashboard | ~25,000 | 40+ |
| Real-Time System | ~1,285 | 5 |
| Database Layer | ~1,600 | 8 |
| Tests | ~8,900 | 30+ |
| Documentation | ~5,000 | 20+ |
| **Total** | **~100,000** | **70+** |

By Language: Python ~74,000 lines · TypeScript/React ~24,000 lines · Config ~3,100 lines · Docs ~5,000 lines

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | System design overview |
| [API Reference](docs/API_DOCUMENTATION.md) | Complete REST API docs |
| [WebSocket Guide](docs/WEBSOCKET.md) | Real-time streaming |
| [Training Guide](TRAINING_GUIDE.md) | Model training pipelines |
| [Advanced ML](docs/ADVANCED_ML.md) | ML modules documentation |
| [Advanced Visualizations](docs/ADVANCED_VISUALIZATIONS.md) | Chart components |
| [Deployment](docs/DEPLOYMENT.md) | Production deployment |
| [Changelog](CHANGELOG.md) | Full version history |

---

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

```bash
pip install -r requirements.txt
cd web-dashboard && pnpm install
make test && make format
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Anthropic · Meta AI · PyTorch · FastAPI · React · Recharts · Optuna · SHAP · OpenBB

---

## Support

- Issues: [GitHub Issues](https://github.com/f4t1i/stock-agent-system-final/issues)
- Documentation: [docs/](docs/)
- Discussions: [GitHub Discussions](https://github.com/f4t1i/stock-agent-system-final/discussions)

---

## Version History

### v2.1.0 (2026-01-05) - Distributed Training & Advanced ML
- Distributed multi-GPU training infrastructure
- GANs for synthetic financial data generation
- Meta-learning (MAML) for few-shot adaptation

### v2.0.0 (2026-01-05) - Advanced ML Release
- Ensemble models (stacking, voting, bagging, adaptive)
- Deep learning (LSTM, Transformer, TCN)
- Reinforcement learning (DQN trading agent)
- AutoML with hyperparameter optimization
- Model explainability (SHAP, LIME)
- Feature engineering (60+ automated features)
- Anomaly detection (4 methods)

### v1.2.0 (2026-01-05) - Advanced Visualizations
- 8 interactive chart types (Candlestick, Line, Area, Heatmap)
- Real-time chart updates via WebSocket
- Portfolio analytics dashboard
- Technical analysis with 10+ indicators

### v1.1.0 (2026-01-05) - Real-Time & Persistence
- WebSocket real-time streaming (5 channels)
- OpenBB market data integration (7 endpoints)
- PostgreSQL database (8 ORM models)
- Docker Compose deployment

### v1.0.0 (2026-01-05) - Foundation Release (Phase A3 Complete)
- Multi-agent architecture (5 agents)
- Advanced training (4 pipelines: SFT, GRPO, RL, Judge)
- Explainability System - Decision reasoning & confidence visualization
- Alerts & Watchlists - Real-time monitoring with multi-channel notifications
- Risk Management - 5 risk gates, trading policies, position validation
- Confidence Calibration - Isotonic regression with reliability diagrams
- Full-stack web dashboard (React 19 + FastAPI)
- Complete testing suite (originally reported as passing; see Project Status)

---

Built with care for intelligent, explainable, and safe stock trading

[Star on GitHub](https://github.com/f4t1i/stock-agent-system-final) | [Read the Docs](docs/) | [Get Started](#quick-start)

Research Prototype | Open Source | Contributions Welcome
