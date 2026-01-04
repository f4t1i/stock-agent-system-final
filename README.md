# Self-Improving Stock Analysis Multi-Agent System

Ein produktionsreifes, selbst-verbesserndes Multi-Agenten-System für quantitative Aktienanalyse mit Reinforcement Learning, LLM-Judge Evaluation und automatisierter Datensynthese.

## 🎯 Systemübersicht

Dieses System implementiert eine hierarchische Multi-Agenten-Architektur für die Aktienanalyse:

- **Junior-Agenten**: Spezialisierte Agenten für News-Sentiment, technische Analyse, Fundamentaldaten
- **Supervisor**: Intelligenter Router mit Contextual Bandits (NeuralUCB)
- **Senior Strategist**: RL-optimierter Entscheidungsagent (PPO/GRPO)
- **LLM-Judge**: Automatisiertes Reward-System für kontinuierliches Lernen
- **Data Synthesis Pipeline**: Automatische Generierung von Trainingsdaten aus erfolgreichen Trajektorien

## 🏗️ Architektur

```
┌─────────────────────────────────────────────────────────────┐
│                        Supervisor Agent                      │
│              (Contextual Bandit - NeuralUCB)                 │
└───────────────┬─────────────────────────────────────────────┘
                │
    ┌───────────┴───────────┐
    │                       │
┌───▼─────┐  ┌──────▼──────┐  ┌──────────▼─────────┐
│ News    │  │ Technical   │  │ Fundamental        │
│ Agent   │  │ Agent       │  │ Agent              │
│ (SFT)   │  │ (SFT)       │  │ (SFT)              │
└───┬─────┘  └──────┬──────┘  └──────────┬─────────┘
    │               │                     │
    └───────────────┴─────────────────────┘
                    │
            ┌───────▼──────────┐
            │ Senior Strategist │
            │   (PPO/GRPO)      │
            └───────┬───────────┘
                    │
            ┌───────▼──────────┐
            │   LLM Judge      │
            │ (Verdict-based)  │
            └───────┬───────────┘
                    │
            ┌───────▼──────────┐
            │ Data Synthesis   │
            │    Pipeline      │
            └──────────────────┘
```

## 📋 Voraussetzungen

### Hardware
- **GPU**: NVIDIA mit mindestens 16GB VRAM (empfohlen: RTX 4090, A100)
- **RAM**: 32GB+ System-RAM
- **Storage**: 100GB+ für Modelle und Trainingsdaten

### Software
- Python 3.10+
- CUDA 12.1+
- Docker (optional, für containerisierte Deployments)

## 🚀 Installation

### 1. Umgebung erstellen

```bash
# Conda-Umgebung
conda create -n stock_agent python=3.10
conda activate stock_agent

# Oder venv
python3.10 -m venv venv
source venv/bin/activate
```

### 2. Abhängigkeiten installieren

```bash
# PyTorch mit CUDA-Support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Unsloth für effizientes Training
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Alle weiteren Abhängigkeiten
pip install -r requirements.txt
```

### 3. Konfiguration

```bash
# .env-Datei erstellen
cp .env.example .env

# API-Schlüssel konfigurieren
nano .env
```

Erforderliche API-Schlüssel:
- `ANTHROPIC_API_KEY`: Für Claude als LLM-Judge
- `FINNHUB_API_KEY`: Für Marktdaten
- `SERPER_API_KEY`: Für News-Recherche
- `WANDB_API_KEY`: Für Experiment-Tracking

## 📖 Schnellstart

### Schritt 1: Daten sammeln

```bash
# Historische Daten für Training herunterladen
python scripts/collect_data.py --symbols AAPL,MSFT,GOOGL --days 365
```

### Schritt 2: Junior-Agenten trainieren (SFT)

```bash
# News-Agent trainieren
python training/sft/train_news_agent.py --config config/sft/news_agent.yaml

# Technical-Agent trainieren
python training/sft/train_technical_agent.py --config config/sft/technical_agent.yaml

# Fundamental-Agent trainieren
python training/sft/train_fundamental_agent.py --config config/sft/fundamental_agent.yaml
```

### Schritt 3: Supervisor trainieren

```bash
# Supervisor mit Contextual Bandits
python training/supervisor/train_supervisor.py --config config/supervisor/neural_ucb.yaml
```

### Schritt 4: Senior Strategist mit RL optimieren

```bash
# PPO Training
python training/rl/train_strategist_ppo.py --config config/rl/ppo_config.yaml

# Oder GRPO (speichereffizienter)
python training/rl/train_strategist_grpo.py --config config/rl/grpo_config.yaml
```

### Schritt 5: System ausführen

```bash
# Interaktive Analyse
python main.py --mode interactive --symbol AAPL

# Batch-Analyse
python main.py --mode batch --symbols-file watchlist.txt

# Backtesting
python main.py --mode backtest --start-date 2023-01-01 --end-date 2024-01-01
```

## 🔧 Konfiguration

Alle Konfigurationsdateien befinden sich in `config/`:

- `config/agents/`: Agent-spezifische Konfigurationen
- `config/sft/`: Supervised Fine-Tuning Parameter
- `config/rl/`: Reinforcement Learning Hyperparameter
- `config/judge/`: LLM-Judge Bewertungskriterien
- `config/data/`: Datenquellen und Preprocessing

### Beispiel: Junior-Agent Konfiguration

```yaml
# config/agents/news_agent.yaml
model:
  base_model: "unsloth/Meta-Llama-3.1-8B-Instruct"
  lora_rank: 16
  lora_alpha: 32
  
training:
  learning_rate: 2e-4
  batch_size: 4
  gradient_accumulation_steps: 4
  max_steps: 1000
  
data:
  max_length: 2048
  dataset_path: "data/processed/news_training.jsonl"
```

## 📊 Monitoring & Evaluation

Das System nutzt Weights & Biases für Experiment-Tracking:

```bash
# W&B Dashboard öffnen
wandb login
python scripts/view_metrics.py
```

Wichtige Metriken:
- **Junior-Agenten**: Accuracy, F1-Score, Sentiment-Korrelation
- **Supervisor**: Routing-Accuracy, Bandit-Regret
- **Senior Strategist**: Sharpe Ratio, Max Drawdown, Win Rate
- **Judge**: Inter-Judge Agreement, Reward Distribution

## 🧪 Testing

```bash
# Unit-Tests
pytest tests/unit/

# Integration-Tests
pytest tests/integration/

# Backtesting
python tests/backtest/run_backtest.py --config tests/backtest/config.yaml
```

## 📁 Projektstruktur

```
stock-agent-system/
├── agents/                    # Agenten-Implementierungen
│   ├── junior/               # News, Technical, Fundamental
│   ├── supervisor/           # Routing-Logik
│   └── senior/               # Senior Strategist
├── training/                 # Training-Pipelines
│   ├── sft/                  # Supervised Fine-Tuning
│   ├── rl/                   # Reinforcement Learning
│   └── data_synthesis/       # Automatische Datengenerierung
├── judge/                    # LLM-Judge System
├── orchestration/            # LangGraph Workflows
├── utils/                    # Hilfsfunktionen
├── config/                   # Konfigurationsdateien
├── docs/                     # Ausführliche Dokumentation
├── tests/                    # Test-Suite
├── examples/                 # Beispiel-Notebooks
├── data/                     # Daten-Verzeichnisse
├── models/                   # Trainierte Modelle
└── scripts/                  # Utility-Skripte
```

## 🔬 Erweiterte Features

### 1. Data Synthesis Pipeline

Automatische Generierung von Trainingsdaten aus erfolgreichen Trajektorien:

```bash
python training/data_synthesis/synthesize_trajectories.py \
    --experience-library data/trajectories/experience.db \
    --min-reward 0.7 \
    --output data/processed/synthetic_sft.jsonl
```

### 2. Online Learning

Kontinuierliche Verbesserung während des Betriebs:

```bash
python scripts/online_learning.py \
    --mode production \
    --update-frequency daily \
    --min-samples 100
```

### 3. Multi-Model Ensemble

Kombination mehrerer Modellvarianten:

```bash
python orchestration/ensemble.py \
    --models models/strategist_v1,models/strategist_v2 \
    --weights 0.6,0.4
```

## 🐛 Troubleshooting

### VRAM Out of Memory

```bash
# Reduziere Batch-Size
export BATCH_SIZE=2

# Aktiviere Gradient Checkpointing
export USE_GRADIENT_CHECKPOINTING=true

# Nutze GRPO statt PPO (kein Value-Model nötig)
python training/rl/train_strategist_grpo.py
```

### Reward Hacking

Wenn Agenten die Reward-Funktion ausnutzen:

1. KL-Divergenz-Strafe erhöhen (`kl_penalty: 0.1` → `0.2`)
2. Brevity Penalty aktivieren
3. Judge-Kriterien verschärfen (siehe `config/judge/rubrics.yaml`)

### Daten-Drift

Bei sinkender Performance über Zeit:

```bash
# Regelmäßige Re-Kalibrierung
python scripts/recalibrate_models.py --window 30days

# Neue Marktregimes in Experience Library
python training/data_synthesis/update_experience.py --detect-regime-shift
```

## 📚 Dokumentation

Ausführliche Dokumentation in `docs/`:

- [Architektur-Guide](docs/ARCHITECTURE.md)
- [Agent-Implementierung](docs/AGENTS.md)
- [Training-Prozess](docs/TRAINING.md)
- [LLM-Judge System](docs/JUDGE.md)
- [Deployment-Guide](docs/DEPLOYMENT.md)
- [API-Referenz](docs/API.md)

## 🤝 Contributing

Contributions sind willkommen! Bitte beachte:

1. Fork das Repository
2. Erstelle einen Feature-Branch (`git checkout -b feature/AmazingFeature`)
3. Commit deine Änderungen (`git commit -m 'Add AmazingFeature'`)
4. Push zum Branch (`git push origin feature/AmazingFeature`)
5. Öffne einen Pull Request

## 📄 Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert - siehe [LICENSE](LICENSE) für Details.

## 🙏 Acknowledgments

- **PrimoAgent**: Referenzarchitektur für LangGraph-basierte Systeme
- **Unsloth**: Effizientes Training von LLMs
- **TRL**: Reinforcement Learning für Language Models
- **Verdict**: LLM-as-a-Judge Framework
- **TradingGroup Paper**: Theoretisches Fundament

## 📧 Kontakt

Für Fragen und Support:
- GitHub Issues: [Create Issue](https://github.com/yourusername/stock-agent-system/issues)
- Email: support@stock-agent-system.com

---

**Hinweis**: Dieses System dient ausschließlich zu Forschungs- und Bildungszwecken. Keine Anlageberatung. Eigenverantwortliche Nutzung.
