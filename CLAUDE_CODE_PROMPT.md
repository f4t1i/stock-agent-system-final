# Claude Code Prompt: Self-Improving Stock Analysis Multi-Agent System

## 🎯 Hauptauftrag

Implementiere das komplette Self-Improving Stock Analysis Multi-Agent System basierend auf der vorhandenen Projektstruktur, Dokumentation und Code-Patterns.

## 📋 Arbeitsanweisung

### Phase 1: Analyse & Verständnis (15 Minuten)

1. **Lies die Kerndokumentation:**
   - `CLAUDE_CODE.md` - Deine Haupt-Implementierungsanleitung
   - `ARCHITECTURE.md` - System-Architektur und Datenflüsse
   - `TRAINING.md` - Training-Prozess Details
   - `PROJECT_SUMMARY.md` - Projekt-Übersicht

2. **Analysiere vorhandene Implementierungen:**
   - `agents/base_agent.py` - Base-Pattern für alle Agenten
   - `agents/junior/news_agent.py` - Referenz für Junior-Agenten
   - `agents/junior/technical_agent.py` - Zweites Referenz-Pattern
   - `agents/supervisor/supervisor_agent.py` - Supervisor-Implementierung
   - `training/sft/train_news_agent.py` - SFT Training-Pattern
   - `training/rl/train_strategist_grpo.py` - RL Training-Pattern

3. **Verstehe die Projektstruktur:**
   - Welche Komponenten sind fertig? (✅)
   - Welche fehlen noch? (📝 TODO)
   - Welche Dependencies gibt es?

### Phase 2: Fehlende Core-Komponenten (1-2 Tage)

Implementiere in dieser Reihenfolge:

#### 2.1 Fundamental Agent
**Datei:** `agents/junior/fundamental_agent.py`

**Anforderungen:**
- Nutze `base_agent.py` als Basis (wie News Agent)
- Implementiere SEC Filing Parser (10-Q, 10-K)
- Berechne Key Metrics: P/E, P/B, ROE, ROA, Debt/Equity, EPS Growth
- LLM-basierte Interpretation der Kennzahlen
- Output-Format wie in `ARCHITECTURE.md` Sektion 1.3 definiert

**Referenzen:**
- Schaue dir `news_agent.py` für Code-Pattern an
- Nutze `yfinance` für Finanzdaten
- System-Prompt ähnlich wie News Agent, aber für Fundamentalanalyse

**Test:**
```python
agent = FundamentalAgent("models/fundamental_agent", config)
result = agent.analyze("AAPL", period="Q")
assert "pe_ratio" in result
assert "recommendation" in result
```

#### 2.2 Senior Strategist Agent
**Datei:** `agents/senior/strategist_agent.py`

**Anforderungen:**
- Kombiniert Outputs von allen Junior-Agenten
- Trifft finale Buy/Sell/Hold Entscheidung
- Risiko-Management (Stop-Loss, Position-Sizing)
- Format wie in `ARCHITECTURE.md` Sektion 3 definiert

**Besonderheiten:**
- Wird später mit GRPO trainiert (RL)
- Initial: SFT auf synthetischen Daten
- Muss Portfolio-State managen

#### 2.3 LLM Judge System
**Datei:** `judge/llm_judge.py`

**Anforderungen:**
- Implementiere Single-Judge basierend auf Anthropic Claude
- Lade Rubrics aus `config/judge/rubrics/`
- Score-Berechnung wie in `ARCHITECTURE.md` Sektion 4 definiert

**Datei:** `judge/multi_judge.py`

**Anforderungen:**
- Multi-Judge Consensus (Claude + GPT-4 + DeepSeek)
- Outlier-Filterung
- Inter-Rater Agreement Berechnung

**Erstelle Rubrics:**
- `config/judge/rubrics/news_rubric.yaml`
- `config/judge/rubrics/technical_rubric.yaml`
- `config/judge/rubrics/strategist_rubric.yaml`

Format siehe `ARCHITECTURE.md` Sektion 4.1

#### 2.4 System Coordinator
**Datei:** `orchestration/coordinator.py`

**Anforderungen:**
- Orchestriert alle Agenten
- Nutzt Supervisor für Routing
- Managed Kontext und State
- Implementiert `analyze_symbol()` Methode für `main.py`

**Flow:**
```
1. User Query → Context erstellen
2. Supervisor → Routing-Entscheidung
3. Junior-Agenten → Parallel oder Sequentiell aktivieren
4. Senior Strategist → Finale Entscheidung
5. Logger → Trajektorie speichern
6. Return → Strukturiertes Ergebnis
```

#### 2.5 LangGraph Workflow
**Datei:** `orchestration/langgraph_workflow.py`

**Anforderungen:**
- Definiere State-Schema
- Implementiere Nodes für jeden Agenten
- Conditional Edges basierend auf Supervisor
- Error-Handling und Retries

**Referenz:**
- Siehe `ARCHITECTURE.md` Sektion "Datenfluss"
- Nutze `langgraph` Library

### Phase 3: Data & Training Infrastructure (1 Tag)

#### 3.1 Data Collection Scripts
**Datei:** `scripts/collect_data.py`

**Anforderungen:**
- Download historische Preisdaten via `yfinance`
- Argumente: `--symbols`, `--days`, `--output`
- Speichere als Parquet in `data/raw/`

**Datei:** `scripts/collect_news.py`

**Anforderungen:**
- Hole News via Finnhub/NewsAPI
- Filtere nach Symbolen
- Deduplizierung

**Datei:** `scripts/generate_synthetic_data.py`

**Anforderungen:**
- Nutze GPT-4o/Claude für Synthese
- Generiere SFT-Trainingsdaten
- Format: ChatML Messages
- Argumente: `--agent-type`, `--num-examples`, `--output`

#### 3.2 Experience Library
**Datei:** `training/data_synthesis/experience_library.py`

**Anforderungen:**
- SQLite-basierte Speicherung
- Schema wie in `ARCHITECTURE.md` Sektion 5.1
- Methoden: `add_trajectory()`, `get_top_trajectories()`, `detect_regime_shift()`

**Datei:** `training/data_synthesis/synthesize_trajectories.py`

**Anforderungen:**
- Lade erfolgreiche Trajektorien (reward > threshold)
- Error Healing für fehlerhafte Trajektorien
- Augmentierung für Diversity
- Output: JSONL für SFT Re-Training

#### 3.3 Training Scripts vervollständigen

**Datei:** `training/sft/train_technical_agent.py`
- Analog zu `train_news_agent.py`

**Datei:** `training/sft/train_fundamental_agent.py`
- Analog zu `train_news_agent.py`

**Datei:** `training/supervisor/train_supervisor.py`
- Contextual Bandit Training
- Replay Buffer Management
- Online Learning Loop

**Datei:** `training/rl/train_strategist_ppo.py`
- PPO-Alternative zu GRPO
- Value Network
- Für >24GB VRAM

### Phase 4: Utilities & Infrastructure (4 Stunden)

#### 4.1 Utility Scripts

**Datei:** `utils/news_fetcher.py`
```python
class NewsFetcher:
    def __init__(self, api_key):
        # Finnhub/NewsAPI Client
    
    def get_news(self, symbol, days=7):
        # Hole News
        # Dedupliziere
        # Format standardisieren
        return news_articles
```

**Datei:** `utils/market_data.py`
```python
class MarketDataFetcher:
    def get_historical(self, symbol, period):
        # yfinance wrapper
    
    def get_realtime(self, symbol):
        # Real-time Daten
```

**Datei:** `utils/metrics.py`
```python
def calculate_sharpe_ratio(returns):
    # Sharpe Ratio Berechnung

def calculate_max_drawdown(equity_curve):
    # Max Drawdown

def calculate_win_rate(trades):
    # Win Rate
```

#### 4.2 API Server (Optional, aber empfohlen)

**Datei:** `api/server.py`

**Anforderungen:**
- FastAPI Server
- Endpoints:
  - `POST /analyze` - Einzelne Analyse
  - `POST /batch` - Batch-Analyse
  - `GET /models` - Verfügbare Modelle
  - `GET /health` - Health Check

**Datei:** `api/schemas.py`
- Pydantic Models für Request/Response

### Phase 5: Testing (4 Stunden)

#### 5.1 Unit Tests

**Verzeichnis:** `tests/unit/`

Erstelle Tests für:
- `test_news_agent.py`
- `test_technical_agent.py`
- `test_fundamental_agent.py`
- `test_supervisor.py`
- `test_strategist.py`
- `test_judge.py`

**Pattern:**
```python
def test_news_agent_analyze():
    agent = NewsAgent(model_path, config)
    result = agent.analyze("AAPL", news_articles=mock_news)
    
    assert "sentiment_score" in result
    assert -2 <= result["sentiment_score"] <= 2
    assert result["confidence"] >= 0
```

#### 5.2 Integration Tests

**Verzeichnis:** `tests/integration/`

- `test_full_workflow.py` - End-to-End Test
- `test_coordinator.py` - Orchestration Test

#### 5.3 Backtesting

**Datei:** `training/rl/backtester.py`

**Anforderungen:**
- Simuliere Trades auf historischen Daten
- Berechne Metriken (Sharpe, Drawdown, Win Rate)
- Visualisierung der Equity Curve

### Phase 6: Konfigurationen vervollständigen (2 Stunden)

Erstelle alle fehlenden Configs in `config/`:

#### `config/system.yaml`
```yaml
agents:
  news:
    enabled: true
    weight: 0.4
    model_path: models/news_agent_v1
  
  technical:
    enabled: true
    weight: 0.35
    model_path: models/technical_agent_v1
  
  fundamental:
    enabled: true
    weight: 0.25
    model_path: models/fundamental_agent_v1

supervisor:
  model_path: models/supervisor_v1
  exploration_factor: 0.5

strategist:
  model_path: models/strategist_v1
  
risk:
  max_position_size: 0.10
  max_drawdown: 0.15
  stop_loss: 0.05
```

#### `config/sft/technical_agent.yaml`
- Analog zu `news_agent.yaml`

#### `config/sft/fundamental_agent.yaml`
- Analog zu `news_agent.yaml`

#### `config/rl/ppo_config.yaml`
```yaml
model:
  sft_checkpoint: models/strategist_sft/final
  
training:
  algorithm: ppo
  # PPO-spezifische Parameter
```

#### `config/supervisor/neural_ucb.yaml`
- Wie in `TRAINING.md` beschrieben

### Phase 7: Dokumentation ergänzen (2 Stunden)

#### Erstelle fehlende Docs:

**Datei:** `docs/AGENTS.md`
- Detaillierte Beschreibung jedes Agenten
- API-Referenz
- Konfigurationsoptionen

**Datei:** `docs/JUDGE.md`
- LLM-Judge System
- Rubric-Design
- Multi-Judge Consensus

**Datei:** `docs/DEPLOYMENT.md`
- Docker Deployment
- API-Server Setup
- Production Best Practices
- Monitoring & Alerting

**Datei:** `docs/API.md`
- REST API Dokumentation
- Request/Response Schemas
- Beispiel-Calls

#### Update README.md
- Füge "Installation erfolgreich getestet" Badge hinzu
- Verlinke alle Docs
- Update Feature-Liste mit allen implementierten Komponenten

### Phase 8: Testing & Validation (4 Stunden)

1. **Run all tests:**
```bash
pytest tests/ -v --cov=agents --cov=training --cov=judge
```

2. **Integration Test:**
```bash
python main.py --mode interactive --symbol AAPL
python main.py --mode batch --symbols-file tests/fixtures/test_symbols.txt
```

3. **Training Test:**
```bash
# Kurzes SFT Training (100 steps) zum Testen
python training/sft/train_news_agent.py \
    --config config/sft/news_agent.yaml \
    --max-steps 100
```

4. **Backtesting:**
```bash
python main.py \
    --mode backtest \
    --start-date 2023-01-01 \
    --end-date 2023-03-31
```

### Phase 9: Polish & Documentation (2 Stunden)

1. **Code Quality:**
   - Run `black .` für Formatting
   - Run `isort .` für Import-Sortierung
   - Run `flake8` für Linting
   - Füge Docstrings zu allen Funktionen hinzu

2. **Error Handling:**
   - Überprüfe alle try-except Blöcke
   - Sinnvolle Error-Messages
   - Logging an den richtigen Stellen

3. **Konfiguration:**
   - Validiere alle YAML-Dateien
   - Prüfe Default-Werte
   - Environment Variable Substitution testen

## 📊 Erfolgskriterien

### Muss erfüllt sein:

✅ **Alle Agenten implementiert:**
- [ ] News Agent funktioniert
- [ ] Technical Agent funktioniert
- [ ] Fundamental Agent funktioniert
- [ ] Supervisor funktioniert
- [ ] Senior Strategist funktioniert

✅ **Orchestration funktioniert:**
- [ ] Coordinator kann Analysen durchführen
- [ ] LangGraph Workflow läuft
- [ ] Error-Handling greift

✅ **Training funktioniert:**
- [ ] SFT für alle Junior-Agenten erfolgreich
- [ ] Supervisor Training funktioniert
- [ ] GRPO Training läuft durch

✅ **Tests bestehen:**
- [ ] Alle Unit-Tests grün
- [ ] Integration-Tests erfolgreich
- [ ] Backtesting liefert Metriken

✅ **System lauffähig:**
- [ ] `python main.py --mode interactive --symbol AAPL` funktioniert
- [ ] Output ist strukturiert und korrekt
- [ ] Keine kritischen Fehler

### Sollte erfüllt sein:

⭐ **Performance-Metriken:**
- [ ] Backtested Sharpe Ratio > 1.0 (Ziel: >1.5)
- [ ] Win Rate > 50% (Ziel: >55%)
- [ ] Max Drawdown < 20% (Ziel: <15%)

⭐ **Code Quality:**
- [ ] Test Coverage > 70%
- [ ] Keine Linting-Fehler
- [ ] Alle Funktionen dokumentiert

⭐ **Dokumentation:**
- [ ] Alle Docs vollständig
- [ ] README aktuell
- [ ] Beispiele funktionieren

## 🎯 Arbeitsweise

### 1. Schrittweise vorgehen
- Implementiere eine Komponente komplett, bevor du zur nächsten gehst
- Teste jede Komponente sofort nach Implementierung
- Committe funktionierende Komponenten

### 2. Referenzen nutzen
- Orientiere dich an vorhandenen Implementierungen
- Halte Code-Patterns konsistent
- Nutze gleiche Fehlerbehandlung wie in Beispielen

### 3. Dokumentation lesen
- `ARCHITECTURE.md` für Systemdesign-Entscheidungen
- `TRAINING.md` für Training-Details
- `CLAUDE_CODE.md` für spezifische Implementierungs-Hinweise

### 4. Bei Unsicherheiten
- Prüfe `PROJECT_SUMMARY.md` für Übersicht
- Schaue in bestehenden Code für Patterns
- Halte dich an die Spezifikationen in `ARCHITECTURE.md`

### 5. Best Practices
- Nutze Type Hints (`def analyze(self, symbol: str) -> Dict[str, Any]`)
- Schreibe aussagekräftige Docstrings
- Error-Handling mit spezifischen Exceptions
- Logging statt print()
- Config-driven statt Hardcoding

## 🚨 Wichtige Hinweise

### Memory-Management
```python
# Immer nutzen für Training:
model_config = {
    'load_in_4bit': True,  # Für GPUs <24GB
    'gradient_checkpointing': True
}
```

### Fehlerbehandlung
```python
# Pattern für alle Agents:
try:
    result = self.analyze(...)
except Exception as e:
    logger.error(f"Error in {self.name}: {e}")
    return self._fallback_response()
```

### Config-Zugriff
```python
# Immer via config_loader:
from utils.config_loader import load_config

config = load_config('config/system.yaml')
# Niemals Hardcoding!
```

## 📝 Fragen beantworten

Wenn du Fragen hast:

1. **Architektur-Fragen** → Siehe `ARCHITECTURE.md`
2. **Training-Fragen** → Siehe `TRAINING.md`
3. **Implementierungs-Details** → Siehe bestehenden Code
4. **Config-Fragen** → Siehe `CLAUDE_CODE.md`

## 🎉 Abschluss

Wenn alles implementiert ist:

1. **Final Test:**
```bash
# Full system test
python main.py --mode backtest --start-date 2023-01-01 --end-date 2023-12-31

# Erwartetes Ergebnis:
# - Sharpe Ratio > 1.0
# - Max Drawdown < 20%
# - Keine Errors
```

2. **Dokumentiere:**
   - Update `PROJECT_SUMMARY.md` mit "Status: Production Ready"
   - Liste alle implementierten Features
   - Notiere bekannte Limitationen

3. **Prepare for User:**
   - Erstelle `CHANGELOG.md` mit allen Änderungen
   - Final README-Update
   - Deployment-Guide validieren

---

## ⚡ Los geht's!

Beginne mit Phase 1 (Analyse & Verständnis) und arbeite dich durch alle Phasen.

Viel Erfolg! 🚀
