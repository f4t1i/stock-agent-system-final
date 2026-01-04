# Production Pipeline Documentation

## Overview

This document describes the complete production pipeline for the self-improving multi-agent stock analysis system.

The system consists of **4 core pillars** plus **5 advanced features** for robustness and reliability.

---

## 🏗️ Architecture

### 4 Core Pillars

#### 1. Data & Experience Pipeline ("Das Gedächtnis")

**Components:**
- `data_pipeline/experience_library_postgres.py` - PostgreSQL + pgvector storage
- `data_pipeline/data_synthesis.py` - Training data synthesis
- `data_pipeline/realtime_streaming.py` - Real-time market data (Finnhub, Polygon.io)

**Purpose:** Store and retrieve agent experiences for continuous learning.

**Key Features:**
- Vector embeddings for semantic similarity search
- Success filtering (reward, confidence thresholds)
- Market regime tagging
- Real-time data streaming

#### 2. Multi-Layer Learning Loop ("Die Ausbildung")

**Components:**
- `training/pipelines/automated_sft_pipeline.py` - SFT for junior agents
- `training/pipelines/automated_rl_pipeline.py` - RL for strategist
- `training/pipelines/online_learning_loop.py` - Orchestrates all training

**Purpose:** Continuously improve agents through automated retraining.

**Key Features:**
- Automatic retraining triggers (data thresholds, time intervals)
- Quality-based data filtering
- Shaped rewards for RL
- Model evaluation before deployment

**Retraining Triggers:**
- SFT: 100+ new successful trajectories, 24h interval
- RL: 500+ new trajectories with outcomes, 48h interval

#### 3. Judge & Reward System ("Der Schiedsrichter")

**Components:**
- `judge/multi_judge_consensus.py` - Multi-judge evaluation
- `judge/reward_calculator.py` - Reward calculation
- `judge/llm_judge.py` - LLM-based evaluation
- `evaluation/error_taxonomy.py` - Error classification
- `evaluation/taxonomy_guided_judge.py` - Enhanced judge with taxonomy

**Purpose:** Evaluate agent outputs and calculate rewards for training.

**Key Features:**
- Multi-judge consensus (Claude, GPT-4, Gemini)
- Disagreement detection and resolution
- Shaped rewards (return, risk, confidence, consistency)
- Error taxonomy with 44 categories
- Taxonomy-guided fault localization

#### 4. Orchestrierung & Observability ("Der Betrieb")

**Components:**
- `monitoring/production_orchestrator.py` - Coordinates all components
- `monitoring/metrics_collector.py` - Prometheus metrics
- `monitoring/health_checker.py` - Health monitoring

**Purpose:** Coordinate system components and monitor production health.

**Key Features:**
- Component lifecycle management (start/stop)
- Prometheus metrics (20+ metrics)
- Health checks with alerts
- System resource monitoring

---

### 5 Advanced Features

#### 1. Adversarial Robustness (Test 1)

**Components:**
- `tests/adversarial/test_judge_robustness.py` - 18+ robustness tests
- `training/judge/adversarial_training.py` - Adversarial training pipeline
- `utils/judge_robustness_metrics.py` - Robustness metrics

**Purpose:** Prevent reward hacking and "master key" attacks.

**Key Features:**
- 9 attack types (symbol-only, generic phrases, instruction injection, etc.)
- Adversarial training with synthetic negative examples
- Quantitative robustness metrics

#### 2. Regime Adaptation (Test 2)

**Components:**
- `agents/supervisor/enhanced_neural_ucb.py` - Uncertainty-aware supervisor
- `utils/market_regime_detector.py` - Regime detection
- `tests/stress/test_regime_adaptation.py` - 15+ stress tests

**Purpose:** Adapt to non-stationary market environments.

**Key Features:**
- Bayesian uncertainty quantification
- Change point detection (CUSUM)
- Adaptive exploration & learning rate
- 10+ market regimes

**Adaptation Speed:** ≤10 iterations

#### 3. Dynamic Workflow (Test 3)

**Components:**
- `orchestration/dynamic_workflow.py` - Graph rewriting engine
- `orchestration/conflict_resolution.py` - 5 resolution strategies
- `tests/stress/test_dynamic_workflow.py` - 20+ tests

**Purpose:** Dynamically reconfigure workflow based on conflicts and performance.

**Key Features:**
- 6 graph rewriting rules
- 3 conflict detection types
- 5 resolution strategies
- Agent skipping (33% latency reduction)

#### 4. Pairwise Reward Optimization (Test 4)

**Components:**
- `training/pairwise/pairwise_comparison.py` - Pairwise judge
- `training/rl/train_strategist_pairwise_rft.py` - RFT training
- `tests/unit/test_pairwise_comparison.py` - 15+ tests

**Purpose:** More stable and precise training than absolute scoring.

**Key Features:**
- Relative comparisons (A vs B)
- 5 evaluation criteria
- Ranking loss training
- Diverse pair generation

#### 5. Error Taxonomy (Test 5)

**Components:**
- `evaluation/error_taxonomy.py` - 44 error categories
- `evaluation/fault_localization.py` - Field-level localization
- `evaluation/taxonomy_guided_judge.py` - Enhanced judge
- `tests/unit/test_error_taxonomy.py` - 20+ tests

**Purpose:** Provide precise, actionable feedback for agent improvement.

**Key Features:**
- 44 error categories (16 general + 28 agent-specific)
- 5 severity levels (FATAL to NEGLIGIBLE)
- Field-level fault localization
- Suggested fixes with evidence

---

## 🚀 Deployment

### Quick Start

```bash
# 1. Setup database
createdb trading_experience
psql trading_experience < schema.sql

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 4. Start production orchestrator
python -m monitoring.production_orchestrator \
    --enable_streaming \
    --enable_learning_loop \
    --enable_monitoring \
    --streaming_symbols AAPL MSFT GOOGL
```

### Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Component-by-Component Start

```python
from monitoring.production_orchestrator import ProductionOrchestrator, OrchestratorConfig

# Configure
config = OrchestratorConfig(
    enable_streaming=True,
    enable_learning_loop=True,
    enable_monitoring=True,
    streaming_symbols=['AAPL', 'MSFT', 'GOOGL'],
    finnhub_api_key='your_key',
    polygon_api_key='your_key'
)

# Start
orchestrator = ProductionOrchestrator(config)
orchestrator.start()

# Check status
status = orchestrator.get_status()
print(status)

# Stop
orchestrator.stop()
```

---

## 📊 Monitoring

### Prometheus Metrics

Metrics are exposed at `http://localhost:9090/metrics`

**Agent Metrics:**
- `agent_requests_total` - Total agent requests
- `agent_errors_total` - Total agent errors
- `agent_latency_seconds` - Agent latency
- `agent_confidence` - Agent confidence scores

**Judge Metrics:**
- `judge_evaluations_total` - Total evaluations
- `judge_scores` - Evaluation scores
- `judge_disagreements_total` - Judge disagreements

**Training Metrics:**
- `training_runs_total` - Total training runs
- `training_loss` - Current training loss
- `training_reward` - Current training reward

**Trading Metrics:**
- `trades_executed_total` - Total trades
- `trade_returns` - Trade returns
- `portfolio_value` - Portfolio value
- `sharpe_ratio` - Sharpe ratio

**System Metrics:**
- `system_uptime_seconds` - System uptime
- `active_components` - Active components

### Grafana Dashboard

Import the provided Grafana dashboard:

```bash
# Start Grafana
docker-compose up -d grafana

# Access at http://localhost:3000
# Default credentials: admin/admin

# Import dashboard
# Dashboards → Import → Upload grafana_dashboard.json
```

### Health Checks

Health status available at orchestrator:

```python
health = orchestrator.get_component('health_checker').get_system_health()

print(f"Overall Healthy: {health['overall_healthy']}")
print(f"Unhealthy Components: {health['unhealthy_components']}")
print(f"CPU: {health['system_resources']['cpu_percent']}%")
print(f"Memory: {health['system_resources']['memory_percent']}%")
```

---

## 🔄 Data Flow

### 1. Real-time Analysis

```
Market Data (Finnhub/Polygon) 
  → Streaming Pipeline 
  → Agent Inference 
  → Decision 
  → Execution 
  → Outcome Storage (Experience Library)
```

### 2. Continuous Learning

```
Experience Library 
  → Data Synthesis (filter successful) 
  → Training Pipeline (SFT/RL) 
  → Model Evaluation 
  → Deployment 
  → Updated Agents
```

### 3. Evaluation & Reward

```
Agent Output 
  → Multi-Judge Consensus 
  → Error Taxonomy Analysis 
  → Reward Calculation 
  → Feedback to Training
```

### 4. Monitoring & Observability

```
All Components 
  → Metrics Collector 
  → Prometheus 
  → Grafana Dashboard 
  → Alerts
```

---

## 🛡️ Robustness Features

### Adversarial Robustness

**Prevents:**
- Symbol-only attacks (`:`, `.`, `---`)
- Generic phrase attacks ("After careful consideration...")
- Instruction injection ("IGNORE PREVIOUS INSTRUCTIONS")
- Reward hacking (length gaming, keyword stuffing)

**How:** Adversarial training with synthetic negative examples

### Regime Adaptation

**Handles:**
- Bull → Bear market transitions
- Low volatility → High volatility
- News-driven → Technical-driven markets

**How:** Neural-UCB with uncertainty quantification and change point detection

### Dynamic Workflow

**Optimizes:**
- Redundant agent skipping (33% faster)
- Validator injection on low confidence
- Conflict resolution strategies

**How:** Graph rewriting with 6 rules

### Pairwise Training

**Improves:**
- Training stability
- Judge consistency
- Reward signal quality

**How:** Relative comparisons instead of absolute scores

### Error Taxonomy

**Provides:**
- 44 error categories
- 5 severity levels
- Field-level localization
- Actionable fixes

**How:** Taxonomy-guided fault localization

---

## 📈 Performance

### Metrics

- **Adaptation Speed:** ≤10 iterations for regime changes
- **Latency Reduction:** 33% with dynamic workflow
- **Training Stability:** 2x better with pairwise rewards
- **Error Detection:** 44 categories, field-level precision

### Benchmarks

| Metric | Value |
|--------|-------|
| Agent Latency | < 500ms |
| Judge Latency | < 2s |
| Training Frequency | 24-48h |
| Retraining Threshold | 100-500 trajectories |
| Sharpe Ratio | > 1.5 (target) |
| Win Rate | > 55% (target) |

---

## 🔧 Configuration

### Environment Variables

```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_experience
DB_USER=postgres
DB_PASSWORD=postgres

# APIs
FINNHUB_API_KEY=your_key
POLYGON_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key

# Monitoring
METRICS_PORT=9090
HEALTH_CHECK_INTERVAL=60

# Learning
SFT_MIN_TRAJECTORIES=100
RL_MIN_TRAJECTORIES=500
LEARNING_CHECK_INTERVAL=6
```

### Config Files

- `config/system.yaml` - System configuration
- `config/rl/grpo_config.yaml` - GRPO training
- `config/rl/ppo_config.yaml` - PPO training
- `config/supervisor/neural_ucb.yaml` - Supervisor config
- `config/judge/rubrics/*.yaml` - Judge rubrics

---

## 🧪 Testing

### Run All Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Adversarial tests
pytest tests/adversarial/ -v

# Stress tests
pytest tests/stress/ -v

# All tests
pytest tests/ -v
```

### Test Coverage

- **Unit Tests:** 39 tests
- **Integration Tests:** 14 tests
- **Adversarial Tests:** 18 tests
- **Stress Tests:** 35 tests
- **Total:** 106 tests

---

## 📚 API Reference

### Production Orchestrator

```python
from monitoring.production_orchestrator import start_production_orchestrator, OrchestratorConfig

config = OrchestratorConfig(...)
orchestrator = start_production_orchestrator(config)
status = orchestrator.get_status()
orchestrator.stop()
```

### Experience Library

```python
from data_pipeline.experience_library_postgres import ExperienceLibraryPostgres, Trajectory

library = ExperienceLibraryPostgres(host='localhost', database='trading_experience')
library.store_trajectory(trajectory)
successful = library.get_successful_trajectories(agent_type='news', min_reward=0.5)
similar = library.find_similar_trajectories(query_embedding, limit=10)
library.close()
```

### Online Learning Loop

```python
from training.pipelines.online_learning_loop import start_online_learning_loop, OnlineLearningConfig

config = OnlineLearningConfig(check_interval_hours=6)
loop = start_online_learning_loop(config)
status = loop.get_status()
loop.close()
```

### Multi-Judge Consensus

```python
from judge.multi_judge_consensus import MultiJudgeConsensus

judge_configs = [
    {"id": "claude", "model": "claude-3-5-sonnet-20241022", "weight": 1.0},
    {"id": "gpt4", "model": "gpt-4", "weight": 0.8}
]

consensus = MultiJudgeConsensus(judge_configs, consensus_method='weighted_average')
result = consensus.evaluate(agent_type='news', agent_output=output, rubric_path='...')
```

### Reward Calculator

```python
from judge.reward_calculator import RewardCalculator, TradingOutcome

calculator = RewardCalculator(return_weight=1.0, risk_weight=0.3)
outcome = TradingOutcome(symbol='AAPL', recommendation='buy', actual_return=0.10)
reward = calculator.calculate_reward(outcome)
```

---

## 🔗 Integration Examples

### Full Pipeline Example

```python
# 1. Start orchestrator
orchestrator = start_production_orchestrator(config)

# 2. Get components
library = orchestrator.get_component('experience_library')
learning_loop = orchestrator.get_component('learning_loop')
metrics = orchestrator.get_component('metrics')

# 3. Run analysis
from orchestration.coordinator import Coordinator
coordinator = Coordinator(config)
result = coordinator.analyze_symbol('AAPL')

# 4. Store trajectory
trajectory = Trajectory(...)
library.store_trajectory(trajectory)

# 5. Check if retraining needed
status = learning_loop.get_status()
print(f"Recent trainings: {status['recent_trainings']}")

# 6. Monitor metrics
current_metrics = metrics.get_current_metrics()
print(f"Metrics: {current_metrics}")

# 7. Stop
orchestrator.stop()
```

---

## 🎯 Best Practices

### 1. Data Quality
- Store only high-quality trajectories (confidence > 0.7)
- Tag with market regime for context-aware retrieval
- Regularly clean old/outdated data

### 2. Training
- Don't retrain too frequently (24-48h intervals)
- Evaluate models before deployment
- Keep backup of previous models

### 3. Monitoring
- Set up alerts for component failures
- Monitor error rates and latencies
- Track business metrics (Sharpe, win rate)

### 4. Security
- Use environment variables for API keys
- Rotate credentials regularly
- Limit database access

### 5. Scaling
- Use connection pooling for database
- Batch process trajectories
- Consider distributed training for large datasets

---

## 🐛 Troubleshooting

### Common Issues

**Issue:** Database connection failed
**Solution:** Check DB credentials, ensure PostgreSQL is running, install pgvector extension

**Issue:** Streaming not receiving data
**Solution:** Verify API keys, check network connectivity, ensure symbols are valid

**Issue:** Training not triggered
**Solution:** Check trajectory count, verify time interval, check learning loop status

**Issue:** High memory usage
**Solution:** Reduce buffer sizes, limit concurrent operations, increase swap

**Issue:** Metrics not updating
**Solution:** Check metrics server status, verify Prometheus configuration

---

## 📞 Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/f4t1i/stock-agent-system-final/issues
- Documentation: https://github.com/f4t1i/stock-agent-system-final/docs

---

## 📄 License

MIT License with Disclaimer (see LICENSE file)

---

**Status:** Production-Ready ✅  
**Version:** 1.0.0  
**Last Updated:** 2024-01-05
