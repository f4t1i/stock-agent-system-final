# System-Architektur

## Übersicht

Das Self-Improving Stock Analysis Multi-Agent System implementiert eine hierarchische, selbst-lernende Architektur für quantitative Finanzanalyse.

## Kernkomponenten

### 1. Junior-Agenten (Spezialist-Ebene)

Junior-Agenten sind spezialisierte LLMs, die durch Supervised Fine-Tuning (SFT) auf spezifische Aufgaben trainiert wurden.

#### 1.1 News Sentiment Agent

**Aufgabe**: Extraktion und Quantifizierung der Marktstimmung aus unstrukturierten Nachrichtentexten.

**Input**:
```json
{
  "symbol": "AAPL",
  "news_articles": [
    {
      "title": "Apple announces record earnings",
      "content": "...",
      "source": "Reuters",
      "timestamp": "2024-01-15T14:30:00Z"
    }
  ]
}
```

**Output**:
```json
{
  "sentiment_score": 1.8,
  "confidence": 0.92,
  "reasoning": "Strong positive sentiment driven by...",
  "price_impact": "bullish",
  "time_horizon": "short_term"
}
```

**Training-Daten**:
- Synthetically generated via GPT-4o
- Annotierte News-Korpora mit Ground-Truth-Labels
- Historical correlation zwischen Sentiment und tatsächlicher Kursentwicklung

**Modell-Basis**: Llama 3.1 8B mit LoRA Fine-Tuning

#### 1.2 Technical Analysis Agent

**Aufgabe**: Berechnung und Interpretation technischer Indikatoren.

**Implementierte Indikatoren**:
- Momentum: RSI, Stochastic, MACD
- Trend: EMA, SMA, Bollinger Bands
- Volume: OBV, MFI
- Pattern Recognition: Support/Resistance, Chart Patterns

**Output**:
```json
{
  "indicators": {
    "rsi_14": 67.5,
    "macd": {"signal": "bullish", "strength": 0.8},
    "bollinger_position": "upper_band"
  },
  "signals": {
    "momentum": "overbought",
    "trend": "uptrend",
    "reversal_probability": 0.35
  },
  "recommendation": "Hold with caution - approaching overbought"
}
```

#### 1.3 Fundamental Analysis Agent

**Aufgabe**: Analyse von Unternehmenskennzahlen und Finanzberichten.

**Datenquellen**:
- Quarterly/Annual Reports (10-Q, 10-K)
- Earnings Call Transcripts
- SEC Filings

**Key Metrics**:
- Valuation: P/E, P/B, EV/EBITDA
- Profitability: ROE, ROA, Profit Margin
- Growth: YoY Revenue, EPS Growth
- Health: Debt/Equity, Current Ratio

### 2. Supervisor Agent (Routing-Ebene)

Der Supervisor entscheidet dynamisch, welche Junior-Agenten für eine gegebene Anfrage aktiviert werden sollen.

#### 2.1 Contextual Bandit Formulierung

Das Routing-Problem wird als Contextual Multi-Armed Bandit modelliert:

**Zustandsraum** (Context):
```python
context = {
    'market_regime': 'bull' | 'bear' | 'sideways',
    'volatility': vix_value,
    'query_type': 'technical' | 'fundamental' | 'news' | 'mixed',
    'time_horizon': 'intraday' | 'short' | 'medium' | 'long',
    'information_density': news_volume_last_24h
}
```

**Aktionsraum**:
```python
actions = [
    'news_agent_only',
    'technical_agent_only',
    'fundamental_agent_only',
    'news_technical_combo',
    'all_agents_consensus'
]
```

**Reward-Funktion**:
```python
def compute_reward(prediction, actual_outcome, latency):
    # Accuracy reward
    accuracy = 1.0 if prediction_correct else -0.5
    
    # Efficiency penalty (weniger Agenten = schneller)
    efficiency = 1.0 / num_agents_called
    
    # Latency penalty
    time_penalty = -0.01 * latency_seconds
    
    return accuracy + 0.3 * efficiency + time_penalty
```

#### 2.2 NeuralUCB Implementierung

Der Supervisor nutzt ein neuronales Netz zur Schätzung der erwarteten Rewards:

```python
class NeuralUCB:
    def __init__(self, context_dim, num_actions):
        self.network = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
        self.uncertainty_estimates = {}
    
    def select_action(self, context, exploration_factor=1.0):
        # Predicted rewards
        with torch.no_grad():
            q_values = self.network(context)
        
        # Uncertainty bonuses (UCB)
        ucb_values = q_values + exploration_factor * self.get_uncertainty(context)
        
        return torch.argmax(ucb_values).item()
```

### 3. Senior Strategist (Entscheidungs-Ebene)

Der Senior Strategist synthetisiert alle Informationen zu einer finalen Trading-Entscheidung.

#### 3.1 Reinforcement Learning Setup

**State Space**:
```python
state = {
    'portfolio': {
        'cash': float,
        'positions': {symbol: shares},
        'total_value': float
    },
    'market_data': {
        'price': float,
        'volume': int,
        'volatility': float
    },
    'agent_outputs': {
        'news_sentiment': dict,
        'technical_signals': dict,
        'fundamental_metrics': dict
    },
    'risk_metrics': {
        'current_drawdown': float,
        'var_95': float,
        'sharpe_ytd': float
    }
}
```

**Action Space**:
```python
action = {
    'decision': 'buy' | 'sell' | 'hold',
    'size': float,  # Prozent des Portfolios
    'stop_loss': float,  # Prozent
    'take_profit': float  # Prozent
}
```

**Reward-Funktion**:
```python
def compute_strategist_reward(trade_result, reasoning_quality):
    # Financial performance
    returns = (exit_price - entry_price) / entry_price
    
    # Risk-adjusted
    sharpe_contribution = returns / volatility
    
    # Drawdown penalty
    dd_penalty = -max(0, drawdown - 0.10)  # Strafe für >10% Drawdown
    
    # Logic quality (from LLM Judge)
    logic_score = judge_evaluate_reasoning(reasoning)
    
    return (
        0.5 * sharpe_contribution +
        0.3 * logic_score +
        0.2 * dd_penalty
    )
```

#### 3.2 PPO vs. GRPO

**PPO (Proximal Policy Optimization)**:
- Benötigt Value Network (Critic)
- Stabiler, etabliert
- Höherer VRAM-Verbrauch

**GRPO (Group Relative Policy Optimization)**:
- Kein Value Network nötig
- Speichereffizienter
- Besser für LLMs geeignet

**Wann GRPO nutzen**:
- VRAM < 24GB
- Training auf Consumer-GPUs
- Schnelleres Prototyping

### 4. LLM-Judge (Evaluierungs-Ebene)

Der LLM-Judge bewertet die Qualität der Agenten-Outputs qualitativ.

#### 4.1 Bewertungskriterien

**Rubrik für News-Agent**:
```yaml
criteria:
  factual_accuracy:
    weight: 0.4
    description: "Sind extrahierte Fakten korrekt?"
    scale: 0-10
  
  sentiment_calibration:
    weight: 0.3
    description: "Ist Sentiment angemessen?"
    scale: 0-10
  
  completeness:
    weight: 0.2
    description: "Alle relevanten News berücksichtigt?"
    scale: 0-10
  
  format_compliance:
    weight: 0.1
    description: "JSON-Schema eingehalten?"
    scale: 0-10
```

**Rubrik für Senior Strategist**:
```yaml
criteria:
  logical_consistency:
    weight: 0.35
    description: "Folgt Empfehlung aus der Analyse?"
    
  risk_awareness:
    weight: 0.30
    description: "Downside-Risiken identifiziert?"
    
  evidence_quality:
    weight: 0.25
    description: "Konkrete Daten zitiert?"
    
  actionability:
    weight: 0.10
    description: "Klare, umsetzbare Empfehlung?"
```

#### 4.2 Multi-Judge Consensus

Um Bias zu reduzieren, werden mehrere Judge-Instanzen genutzt:

```python
class MultiJudgeConsensus:
    def __init__(self, num_judges=3):
        self.judges = [
            ClaudeJudge(model="claude-sonnet-4"),
            GPT4Judge(model="gpt-4o"),
            DeepSeekJudge(model="deepseek-r1")
        ]
    
    def evaluate(self, output, criteria):
        scores = [judge.score(output, criteria) for judge in self.judges]
        
        # Weighted average mit Outlier-Filtering
        filtered_scores = remove_outliers(scores)
        final_score = np.mean(filtered_scores)
        
        return {
            'score': final_score,
            'variance': np.var(scores),
            'agreement': inter_rater_agreement(scores)
        }
```

### 5. Data Synthesis Pipeline

Die Pipeline generiert automatisch neue Trainingsdaten aus erfolgreichen Trajektorien.

#### 5.1 Experience Library

Alle Interaktionen werden in einer SQLite-Datenbank gespeichert:

```sql
CREATE TABLE trajectories (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    symbol TEXT,
    context_state JSON,
    agent_outputs JSON,
    final_decision JSON,
    actual_outcome FLOAT,
    reward FLOAT,
    judge_score FLOAT
);

CREATE INDEX idx_reward ON trajectories(reward DESC);
CREATE INDEX idx_symbol ON trajectories(symbol);
```

#### 5.2 Synthese-Prozess

```python
def synthesize_training_data(min_reward=0.7, top_k=1000):
    # Hole erfolgreiche Trajektorien
    trajectories = db.query(
        "SELECT * FROM trajectories WHERE reward >= ? ORDER BY reward DESC LIMIT ?",
        (min_reward, top_k)
    )
    
    synthetic_data = []
    
    for traj in trajectories:
        # SFT-Format erstellen
        example = {
            'messages': [
                {
                    'role': 'system',
                    'content': get_agent_system_prompt()
                },
                {
                    'role': 'user',
                    'content': format_input(traj.context_state)
                },
                {
                    'role': 'assistant',
                    'content': traj.agent_outputs
                }
            ]
        }
        synthetic_data.append(example)
    
    # Augmentierung für Diversity
    augmented_data = augment_trajectories(synthetic_data)
    
    return synthetic_data + augmented_data
```

#### 5.3 Error Healing

Trajektorien mit niedrigen Rewards werden nicht verworfen, sondern "geheilt":

```python
def heal_trajectory(failed_traj):
    # Nutze bestes Modell zur Korrektur
    correction_prompt = f"""
    Die folgende Analyse führte zu einem schlechten Ergebnis:
    
    Input: {failed_traj.context_state}
    Output: {failed_traj.agent_outputs}
    Actual Outcome: {failed_traj.actual_outcome}
    
    Was war der Fehler und wie sollte die Analyse lauten?
    """
    
    corrected_output = teacher_model.generate(correction_prompt)
    
    return {
        'messages': [...],  # Mit korrigierter Antwort
        'metadata': {'healed': True, 'original_reward': failed_traj.reward}
    }
```

## Datenfluss

```
1. User Query
   ↓
2. Supervisor (NeuralUCB)
   ├→ News Agent
   ├→ Technical Agent
   └→ Fundamental Agent
   ↓
3. Aggregation
   ↓
4. Senior Strategist (PPO/GRPO)
   ↓
5. Action Execution
   ↓
6. LLM Judge Evaluation
   ↓
7. Experience Library
   ↓
8. Data Synthesis
   ↓
9. Re-Training (SFT + RL)
```

## Selbst-Verbesserungs-Zyklus

```
Woche 1: Initial SFT Training
    ↓
Woche 2-3: Deployment + Data Collection
    ↓
Woche 4: Data Synthesis + Judge Evaluation
    ↓
Woche 5: Incremental SFT Update
    ↓
Woche 6-7: RL Fine-Tuning (PPO/GRPO)
    ↓
Woche 8: Validation + A/B Testing
    ↓
Woche 9+: Repeat Cycle
```

## Skalierbarkeit

### Horizontale Skalierung

- **Junior-Agenten**: Parallel auf mehreren GPUs/Instanzen
- **Supervisor**: Lightweight, CPU-optimiert
- **Senior Strategist**: GPU-intensiv, Queue-basiert

### Vertikale Optimierung

- **Model Quantization**: INT8 für Inference
- **Flash Attention 2**: Schnellere Attention-Berechnung
- **Gradient Checkpointing**: VRAM-Reduktion

## Monitoring

### Key Metrics

**System-Level**:
- Throughput (Analysen/Stunde)
- Latency (P50, P95, P99)
- GPU-Auslastung

**Agent-Level**:
- Accuracy per Agent
- Judge Score Trends
- Reward Distribution

**Financial**:
- Sharpe Ratio
- Max Drawdown
- Win Rate
- Calmar Ratio

## Disaster Recovery

- **Model Checkpoints**: Alle 100 Steps
- **Experience Library Backup**: Täglich
- **Rollback-Mechanismus**: Bei Performance-Degradation
- **A/B Testing**: Neues Modell vs. Production
