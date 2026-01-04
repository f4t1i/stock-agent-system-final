# Claude Code Integration Guide

Diese Anleitung ermöglicht es Claude Code, das Self-Improving Stock Analysis Multi-Agent System direkt aus diesem Projektbaum zu implementieren.

## Projekt-Übersicht

Dieses Projekt implementiert ein hierarchisches Multi-Agenten-System für quantitative Aktienanalyse mit:

- **Junior-Agenten**: News, Technical, Fundamental (SFT-trainiert)
- **Supervisor**: Intelligentes Routing mit NeuralUCB
- **Senior Strategist**: RL-optimierter Entscheidungsagent (PPO/GRPO)
- **LLM-Judge**: Automatisierte Qualitätsbewertung
- **Data Synthesis**: Selbst-verbessernde Pipeline

## Setup-Schritte für Claude Code

### 1. Umgebung einrichten

```bash
# Conda-Umgebung erstellen
conda create -n stock_agent python=3.10 -y
conda activate stock_agent

# PyTorch mit CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Unsloth installieren
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Alle Dependencies
pip install -r requirements.txt
```

### 2. Konfiguration

```bash
# .env erstellen
cp .env.example .env

# API-Schlüssel eintragen
nano .env
```

Erforderliche Keys:
- `ANTHROPIC_API_KEY`: Für Claude Judge
- `FINNHUB_API_KEY`: Marktdaten
- `SERPER_API_KEY`: News-Suche
- `WANDB_API_KEY`: Experiment-Tracking

### 3. Daten sammeln

```bash
# Historische Marktdaten
python scripts/collect_data.py --symbols AAPL,MSFT,GOOGL --days 365

# News-Daten
python scripts/collect_news.py --symbols AAPL,MSFT,GOOGL --days 30

# Synthetic Training Data generieren
python scripts/generate_synthetic_data.py --output data/processed/news_training.jsonl
```

### 4. Junior-Agenten trainieren

```bash
# News Agent
python training/sft/train_news_agent.py --config config/sft/news_agent.yaml

# Technical Agent
python training/sft/train_technical_agent.py --config config/sft/technical_agent.yaml

# Fundamental Agent
python training/sft/train_fundamental_agent.py --config config/sft/fundamental_agent.yaml
```

### 5. Supervisor trainieren

```bash
python training/supervisor/train_supervisor.py --config config/supervisor/neural_ucb.yaml
```

### 6. Senior Strategist mit RL

```bash
# Option 1: GRPO (empfohlen für <24GB VRAM)
python training/rl/train_strategist_grpo.py --config config/rl/grpo_config.yaml

# Option 2: PPO (für >24GB VRAM)
python training/rl/train_strategist_ppo.py --config config/rl/ppo_config.yaml
```

### 7. System testen

```bash
# Interactive Mode
python main.py --mode interactive --symbol AAPL

# Batch Mode
python main.py --mode batch --symbols-file watchlist.txt --output results.json

# Backtesting
python main.py --mode backtest --start-date 2023-01-01 --end-date 2024-01-01
```

## Projektstruktur-Details

```
stock-agent-system/
├── agents/
│   ├── base_agent.py          # Base-Klasse für alle Agenten
│   ├── junior/
│   │   ├── news_agent.py      # News Sentiment Analyst
│   │   ├── technical_agent.py # Technical Indicator Expert
│   │   └── fundamental_agent.py # Financial Statement Analyzer
│   ├── supervisor/
│   │   └── supervisor_agent.py # NeuralUCB Router
│   └── senior/
│       └── strategist_agent.py # Final Decision Maker
│
├── training/
│   ├── sft/                   # Supervised Fine-Tuning
│   │   ├── train_news_agent.py
│   │   ├── train_technical_agent.py
│   │   └── train_fundamental_agent.py
│   ├── rl/                    # Reinforcement Learning
│   │   ├── train_strategist_ppo.py
│   │   ├── train_strategist_grpo.py
│   │   └── backtester.py
│   ├── supervisor/            # Bandit Training
│   │   └── train_supervisor.py
│   └── data_synthesis/        # Automatic Data Generation
│       ├── synthesize_trajectories.py
│       └── experience_library.py
│
├── judge/
│   ├── llm_judge.py          # LLM-as-a-Judge
│   ├── multi_judge.py        # Multi-Judge Consensus
│   └── rubrics/              # Evaluation Criteria
│       ├── news_rubric.yaml
│       ├── technical_rubric.yaml
│       └── strategist_rubric.yaml
│
├── orchestration/
│   ├── coordinator.py        # System Coordinator
│   ├── langgraph_workflow.py # LangGraph Integration
│   └── ensemble.py           # Multi-Model Ensemble
│
├── utils/
│   ├── config_loader.py      # YAML Config Loader
│   ├── logging_setup.py      # Logging Configuration
│   ├── news_fetcher.py       # News API Client
│   ├── market_data.py        # Market Data Utils
│   └── metrics.py            # Performance Metrics
│
├── config/
│   ├── system.yaml           # System-wide Config
│   ├── agents/               # Agent Configs
│   ├── sft/                  # SFT Configs
│   ├── rl/                   # RL Configs
│   ├── judge/                # Judge Configs
│   └── data/                 # Data Source Configs
│
├── scripts/
│   ├── collect_data.py       # Data Collection
│   ├── generate_synthetic_data.py
│   ├── recalibrate_models.py
│   └── view_metrics.py
│
├── docs/
│   ├── ARCHITECTURE.md       # System Architecture
│   ├── AGENTS.md             # Agent Implementation
│   ├── TRAINING.md           # Training Process
│   ├── JUDGE.md              # Judge System
│   └── DEPLOYMENT.md         # Deployment Guide
│
└── tests/
    ├── unit/                 # Unit Tests
    ├── integration/          # Integration Tests
    └── backtest/             # Backtesting Suite
```

## Implementierungs-Reihenfolge

Wenn Claude Code das System von Grund auf implementiert:

### Phase 1: Foundation (Tag 1-2)
1. Base-Klassen und Utils implementieren
2. Config-System aufbauen
3. Logging und Monitoring
4. Data Collection Scripts

### Phase 2: Junior Agents (Tag 3-5)
1. News Agent implementieren und trainieren
2. Technical Agent implementieren und trainieren
3. Fundamental Agent implementieren und trainieren
4. Integration-Tests

### Phase 3: Orchestration (Tag 6-7)
1. Supervisor implementieren
2. LangGraph Workflow
3. System Coordinator
4. End-to-end Tests

### Phase 4: Senior Strategist (Tag 8-10)
1. LLM Judge implementieren
2. Reward Function
3. GRPO Training Pipeline
4. Backtesting Framework

### Phase 5: Self-Improvement (Tag 11-12)
1. Experience Library
2. Data Synthesis Pipeline
3. Online Learning Loop
4. Monitoring Dashboard

### Phase 6: Production (Tag 13-14)
1. API Server (FastAPI)
2. Docker Deployment
3. Performance Optimization
4. Documentation

## Wichtige Implementierungs-Hinweise

### Memory-Optimierung

**Für Training auf GPUs <24GB**:
- Nutze `load_in_4bit=True`
- Gradient Checkpointing aktivieren
- GRPO statt PPO verwenden
- Batch Size reduzieren

```python
# In config/rl/grpo_config.yaml
model:
  load_in_4bit: true

training:
  batch_size: 2
  gradient_accumulation_steps: 8
  num_generations: 4  # Nicht höher für <16GB VRAM
```

### Reward Hacking vermeiden

**KL-Divergenz Penalty**:
```yaml
training:
  kl_penalty: 0.1  # Verhindert zu starke Abweichung vom SFT-Modell
```

**Brevity Penalty**:
```python
def compute_reward(prediction, outcome, context):
    # ... andere Komponenten
    
    # Strafe für unnötig lange Outputs
    token_count = len(tokenizer.encode(prediction))
    if token_count > 500:
        brevity_penalty = -0.01 * (token_count - 500)
    else:
        brevity_penalty = 0
    
    return base_reward + brevity_penalty
```

### Daten-Drift Management

**Kontinuierliche Re-Kalibrierung**:
```bash
# Cron-Job für wöchentliches Update
0 2 * * 0 cd /path/to/project && python scripts/recalibrate_models.py --window 30days
```

**Regime-Shift Detection**:
```python
# In data_synthesis/experience_library.py
def detect_regime_shift(recent_trajectories):
    # Vergleiche Performance-Metrics
    recent_sharpe = calculate_sharpe(recent_trajectories[-100:])
    historical_sharpe = calculate_sharpe(recent_trajectories[-1000:-100])
    
    if abs(recent_sharpe - historical_sharpe) > threshold:
        return True
    return False
```

## Testing-Strategie

### Unit Tests
```bash
pytest tests/unit/ -v
```

### Integration Tests
```bash
pytest tests/integration/ -v
```

### Backtesting
```bash
python tests/backtest/run_backtest.py \
    --config tests/backtest/config.yaml \
    --start-date 2022-01-01 \
    --end-date 2023-12-31
```

## Performance Benchmarks

Erwartete Metriken nach Training:

**Junior Agents**:
- News Sentiment Accuracy: >75%
- Technical Signal Accuracy: >70%
- Fundamental Analysis F1: >0.80

**Supervisor**:
- Routing Accuracy: >85%
- Latency: <100ms

**Senior Strategist**:
- Backtested Sharpe Ratio: >1.5
- Max Drawdown: <15%
- Win Rate: >55%

## Deployment-Checklist

- [ ] Alle Tests bestanden
- [ ] Backtesting-Metriken akzeptabel
- [ ] API-Server läuft stabil
- [ ] Monitoring aufgesetzt (Wandb/Prometheus)
- [ ] Disaster Recovery getestet
- [ ] Dokumentation vollständig
- [ ] A/B Testing konfiguriert

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduziere Batch-Size
export BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=16

# Oder nutze CPU-Offloading
export USE_CPU_OFFLOAD=true
```

### Slow Training
```bash
# Enable Flash Attention 2
pip install flash-attn --no-build-isolation

# In model config
model:
  use_flash_attention_2: true
```

### Poor Convergence
```bash
# Erhöhe Warmup
training:
  warmup_steps: 500  # Statt 100

# Reduziere Learning Rate
training:
  learning_rate: 5e-6  # Statt 1e-5
```

## Weiterführende Ressourcen

- [Unsloth Docs](https://github.com/unslothai/unsloth)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [LangGraph Guide](https://langchain-ai.github.io/langgraph/)
- [Verdict Framework](https://github.com/haizelabs/verdict)

## Support

Bei Problemen:
1. Check Logs: `tail -f logs/system.log`
2. Wandb Dashboard prüfen
3. GitHub Issues erstellen
4. Community Slack beitreten
