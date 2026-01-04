# Quickstart Guide

Schnellstart in 15 Minuten - von der Installation bis zur ersten Aktienanalyse.

## 🚀 Schnellinstallation

### Voraussetzungen

- Python 3.10+
- NVIDIA GPU mit CUDA 12.1+ (empfohlen: 16GB+ VRAM)
- 32GB RAM

### 1. Repository klonen & Setup

```bash
# Umgebung erstellen
conda create -n stock_agent python=3.10 -y
conda activate stock_agent

# In Projektverzeichnis wechseln
cd stock-agent-system

# PyTorch mit CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Unsloth (kann 2-3 Minuten dauern)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Restliche Dependencies
pip install -r requirements.txt
```

### 2. Konfiguration

```bash
# .env erstellen
cp .env.example .env

# Editieren
nano .env
```

**Minimale erforderliche Keys:**
```bash
ANTHROPIC_API_KEY=sk-ant-xxxxx  # Für LLM Judge
FINNHUB_API_KEY=xxxxx           # Für Marktdaten
```

Optional aber empfohlen:
```bash
WANDB_API_KEY=xxxxx  # Für Experiment-Tracking
```

### 3. Vortrainierte Modelle (Optional)

Starte mit vortrainierten Basismodellen:

```bash
# News Agent (Llama 3.1 8B SFT)
python scripts/download_model.py --model news_agent_v1

# Technical Agent
python scripts/download_model.py --model technical_agent_v1

# Oder nutze Hugging Face Basis-Modelle direkt
```

## 📊 Erste Analyse in 5 Minuten

### Option A: Mit vortrainierten Basis-Modellen

```bash
# Interaktive Analyse
python main.py --mode interactive --symbol AAPL

# Ausgabe:
# ================================================================================
# Analysis for AAPL
# ================================================================================
# 
# Recommendation: Buy
# Confidence: 87.5%
# 
# Reasoning:
# Strong bullish momentum with RSI at 62 indicating room for growth.
# Recent earnings beat expectations by 8%, driving positive sentiment.
# Technical indicators show uptrend continuation pattern.
# 
# ...
```

### Option B: Training von Grund auf

Falls du deine eigenen Modelle trainieren möchtest:

#### Schritt 1: Trainingsdaten sammeln

```bash
# Historische Daten (dauert ~5 Minuten)
python scripts/collect_data.py --symbols AAPL,MSFT,GOOGL --days 365

# Synthetische Trainingsdaten generieren
python scripts/generate_synthetic_data.py \
    --agent-type news \
    --num-examples 1000 \
    --output data/processed/news_training.jsonl
```

#### Schritt 2: News Agent trainieren

```bash
# Training (dauert ~30 Minuten auf RTX 4090)
python training/sft/train_news_agent.py --config config/sft/news_agent.yaml
```

#### Schritt 3: System mit eigenem Modell testen

```bash
python main.py --mode interactive --symbol AAPL
```

## 🎯 Schnell-Demo: Batch-Analyse

Analysiere mehrere Aktien gleichzeitig:

```bash
# Erstelle Watchlist
cat > watchlist.txt << EOF
AAPL
MSFT
GOOGL
AMZN
TSLA
EOF

# Batch-Analyse
python main.py \
    --mode batch \
    --symbols-file watchlist.txt \
    --output results.json

# Ergebnisse anzeigen
cat results.json | python -m json.tool
```

## 📈 Backtesting

Teste die Strategie auf historischen Daten:

```bash
python main.py \
    --mode backtest \
    --start-date 2023-01-01 \
    --end-date 2024-01-01 \
    --symbols-file watchlist.txt

# Ausgabe:
# ================================================================================
# Backtest Results
# ================================================================================
# 
# Total Return: 24.5%
# Sharpe Ratio: 1.8
# Max Drawdown: 12.3%
# Win Rate: 58.2%
# Total Trades: 145
```

## 🔧 Troubleshooting

### "CUDA Out of Memory"

```bash
# In config/sft/news_agent.yaml
model:
  load_in_4bit: true  # ← Setze auf true

training:
  batch_size: 2       # ← Reduziere von 4
  gradient_accumulation_steps: 8  # ← Erhöhe von 4
```

### "API Key Invalid"

```bash
# Prüfe .env
cat .env | grep API_KEY

# Test Anthropic Key
python -c "from anthropic import Anthropic; c = Anthropic(); print('OK')"

# Test Finnhub Key
python -c "import finnhub; c = finnhub.Client(api_key='YOUR_KEY'); print('OK')"
```

### "ModuleNotFoundError"

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## 🎓 Nächste Schritte

### 1. Erweiterte Konfiguration

Passe System-Behavior an:

```yaml
# config/system.yaml
agents:
  news:
    enabled: true
    weight: 0.4
  technical:
    enabled: true
    weight: 0.35
  fundamental:
    enabled: true
    weight: 0.25

risk:
  max_position_size: 0.10  # 10% des Portfolios
  max_drawdown: 0.15       # 15% Max Drawdown
  stop_loss: 0.05          # 5% Stop Loss
```

### 2. Training optimieren

#### Nutze Wandb für Monitoring

```bash
wandb login

# Training mit Tracking
python training/sft/train_news_agent.py \
    --config config/sft/news_agent.yaml

# Dashboard öffnet sich automatisch
```

#### Multi-GPU Training

```yaml
# config/sft/news_agent.yaml
training:
  use_deepspeed: true
  num_gpus: 2
```

### 3. Production Deployment

#### API Server starten

```bash
python main.py --mode serve --port 8000

# Test
curl http://localhost:8000/analyze/AAPL
```

#### Docker Deployment

```bash
docker build -t stock-agent-system .
docker run -p 8000:8000 --gpus all stock-agent-system
```

### 4. Kontinuierliches Learning

#### Experience Library aktivieren

```python
# In main.py automatisch aktiviert
# Alle Analysen werden gespeichert in:
# data/trajectories/experience.db
```

#### Wöchentliche Updates

```bash
# Crontab: Jeden Sonntag um 2 Uhr
0 2 * * 0 cd /path/to/project && python scripts/incremental_update.py
```

## 📚 Weitere Ressourcen

- [Vollständige Dokumentation](docs/ARCHITECTURE.md)
- [Training-Guide](docs/TRAINING.md)
- [API-Referenz](docs/API.md)
- [Deployment-Guide](docs/DEPLOYMENT.md)

## 💡 Beispiel-Use-Cases

### 1. Tägliche Portfolio-Analyse

```bash
# Cron-Job: Jeden Tag um 9:30 (Börsenöffnung)
30 9 * * 1-5 cd /path/to/project && python scripts/daily_analysis.py
```

### 2. Alert-System

```python
# scripts/daily_analysis.py
from orchestration.coordinator import SystemCoordinator

coordinator = SystemCoordinator(load_config('config/system.yaml'))

for symbol in portfolio:
    result = coordinator.analyze_symbol(symbol)
    
    if result['signal'] == 'sell' and result['confidence'] > 0.8:
        send_alert(f"Strong SELL signal for {symbol}")
    
    if result['signal'] == 'buy' and result['confidence'] > 0.8:
        send_alert(f"Strong BUY signal for {symbol}")
```

### 3. Research Dashboard

```bash
# Streamlit Dashboard
pip install streamlit

streamlit run dashboard/app.py
```

## 🤝 Support

- GitHub Issues: [Create Issue](https://github.com/yourusername/stock-agent-system/issues)
- Discord: [Join Community](https://discord.gg/stock-agent)
- Email: support@stock-agent-system.com

---

**Hinweis**: Dieses System ist ausschließlich für Forschungs- und Bildungszwecke. Keine Anlageberatung.
