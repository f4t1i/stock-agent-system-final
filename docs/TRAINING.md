# Training-Prozess

Detaillierte Anleitung zum Training aller Systemkomponenten.

## Übersicht

Das Training erfolgt in mehreren Phasen:

1. **Supervised Fine-Tuning (SFT)** - Junior-Agenten
2. **Contextual Bandit Training** - Supervisor
3. **Reinforcement Learning (RL)** - Senior Strategist
4. **Data Synthesis** - Kontinuierliche Verbesserung

## Phase 1: Supervised Fine-Tuning

### 1.1 Daten-Vorbereitung

#### News Agent Training Data

Erzeuge synthetische News-Analysen:

```bash
python scripts/generate_synthetic_data.py \
    --agent-type news \
    --num-examples 10000 \
    --output data/processed/news_training.jsonl
```

Format:
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a specialized financial news analyst..."
    },
    {
      "role": "user",
      "content": "Analyze sentiment for AAPL:\n\nArticle 1: Apple announces..."
    },
    {
      "role": "assistant",
      "content": "{\"sentiment_score\": 1.5, \"confidence\": 0.89, ...}"
    }
  ]
}
```

#### Technical Agent Training Data

Nutze historische Preisdaten + Expert-Annotationen:

```bash
python scripts/generate_technical_training.py \
    --symbols AAPL,MSFT,GOOGL,AMZN,TSLA \
    --start-date 2020-01-01 \
    --end-date 2024-01-01 \
    --output data/processed/technical_training.jsonl
```

#### Fundamental Agent Training Data

Parse SEC Filings und erstelle Analysen:

```bash
python scripts/generate_fundamental_training.py \
    --source sec_edgar \
    --output data/processed/fundamental_training.jsonl
```

### 1.2 SFT Training

#### News Agent

```bash
python training/sft/train_news_agent.py \
    --config config/sft/news_agent.yaml
```

Wichtige Hyperparameter:

```yaml
# config/sft/news_agent.yaml
training:
  num_epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  max_steps: 1000
```

**Erwartete Metriken**:
- Training Loss: ~0.5 nach 1000 steps
- Validation Loss: ~0.6
- F1-Score: >0.75

#### Technical Agent

```bash
python training/sft/train_technical_agent.py \
    --config config/sft/technical_agent.yaml
```

**Erwartete Metriken**:
- Signal Accuracy: >70%
- Pattern Recognition F1: >0.80

#### Fundamental Agent

```bash
python training/sft/train_fundamental_agent.py \
    --config config/sft/fundamental_agent.yaml
```

**Erwartete Metriken**:
- Metric Extraction Accuracy: >90%
- Valuation Reasoning Score: >0.75

### 1.3 Evaluation

Test SFT-Modelle auf Hold-out Set:

```bash
python scripts/evaluate_agent.py \
    --agent-type news \
    --model-path models/news_agent_sft/final \
    --test-data data/processed/news_test.jsonl
```

## Phase 2: Supervisor Training

### 2.1 Contextual Bandit Setup

Der Supervisor lernt optimal zu routen via NeuralUCB.

#### Datensammlung

Sammle Routing-Trajektorien:

```bash
python scripts/collect_routing_data.py \
    --num-trajectories 10000 \
    --output data/trajectories/routing.db
```

Format:
```python
{
    'context': {
        'market_regime': 'bull',
        'volatility': 18.5,
        'query_type': 'mixed',
        'time_horizon': 'short'
    },
    'action': 3,  # Index der gewählten Strategie
    'reward': 0.8  # Erfolg der Entscheidung
}
```

#### Training

```bash
python training/supervisor/train_supervisor.py \
    --config config/supervisor/neural_ucb.yaml \
    --data data/trajectories/routing.db
```

**Wichtige Parameter**:

```yaml
# config/supervisor/neural_ucb.yaml
model:
  context_dim: 16
  hidden_dim: 128
  num_actions: 7  # Anzahl Routing-Strategien

training:
  learning_rate: 1e-3
  batch_size: 32
  num_epochs: 100
  exploration_factor: 1.0
  exploration_decay: 0.995
```

**Erwartete Metriken**:
- Routing Accuracy: >85%
- Regret (nach 10k Entscheidungen): <5%

### 2.2 Online Learning

Supervisor lernt kontinuierlich während Betrieb:

```python
# Nach jeder Routing-Entscheidung
supervisor.update(
    context=market_context,
    action=selected_strategy_idx,
    reward=outcome_reward
)
```

## Phase 3: Reinforcement Learning

### 3.1 Experience Collection

Sammle initiale Trajektorien mit SFT-Modell:

```bash
python scripts/collect_rl_trajectories.py \
    --model models/strategist_sft/final \
    --num-trajectories 5000 \
    --mode backtest \
    --start-date 2022-01-01 \
    --end-date 2023-12-31 \
    --output data/trajectories/rl_init.jsonl
```

### 3.2 Reward Function Design

Definiere Reward-Funktion:

```python
def compute_reward(prediction, actual_outcome, context):
    """
    Reward = 0.5 * sharpe + 0.3 * logic_score + 0.2 * drawdown_penalty
    """
    
    # 1. Financial Performance
    returns = actual_outcome['returns']
    volatility = actual_outcome['volatility']
    sharpe = returns / max(volatility, 0.01)
    sharpe_normalized = (sharpe + 2) / 4  # [-2, 2] -> [0, 1]
    
    # 2. Logic Quality (LLM Judge)
    logic_score = judge.evaluate(prediction, context)
    
    # 3. Drawdown Penalty
    max_dd = actual_outcome['max_drawdown']
    if max_dd > 0.10:
        dd_penalty = -(max_dd - 0.10) * 5
    else:
        dd_penalty = 0
    dd_normalized = max(0, min(1, 1 + dd_penalty))
    
    # Combine
    total_reward = (
        0.5 * sharpe_normalized +
        0.3 * logic_score +
        0.2 * dd_normalized
    )
    
    return total_reward
```

### 3.3 GRPO Training

**Warum GRPO?**
- Kein Value-Model nötig → 50% weniger VRAM
- Besser für LLMs geeignet
- Stabiler als PPO

```bash
python training/rl/train_strategist_grpo.py \
    --config config/rl/grpo_config.yaml
```

**Config**:

```yaml
# config/rl/grpo_config.yaml
model:
  sft_checkpoint: models/strategist_sft/final
  max_seq_length: 2048
  load_in_4bit: true

training:
  output_dir: models/strategist_grpo
  num_epochs: 3
  batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 1e-5
  
  # GRPO-specific
  num_generations: 4  # Anzahl Antworten pro Query
  kl_penalty: 0.1  # Verhindert zu starke Abweichung vom SFT
  
reward:
  judge_model: claude-sonnet-4-20250514
  weights:
    sharpe: 0.5
    logic: 0.3
    drawdown: 0.2
  drawdown_threshold: 0.10
```

**Training-Loop**:

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. Generate multiple responses per query
        responses = model.generate(
            batch['queries'],
            num_return_sequences=num_generations
        )
        
        # 2. Compute rewards
        rewards = [
            reward_function(resp, batch['outcomes'][i], batch['contexts'][i])
            for i, resp in enumerate(responses)
        ]
        
        # 3. GRPO update
        loss = grpo_loss(
            responses,
            rewards,
            reference_model,
            kl_penalty
        )
        
        # 4. Backprop
        loss.backward()
        optimizer.step()
```

**Erwartete Metriken**:
- Epoch 1: Avg Reward ~0.5
- Epoch 3: Avg Reward ~0.7
- KL Divergenz: <0.5

### 3.4 PPO Training (Alternative)

Für GPUs mit >24GB VRAM:

```bash
python training/rl/train_strategist_ppo.py \
    --config config/rl/ppo_config.yaml
```

**Unterschiede zu GRPO**:
- Benötigt Value-Model (Critic)
- Höherer VRAM-Verbrauch
- Möglicherweise stabileres Training

## Phase 4: Data Synthesis

### 4.1 Experience Library

Speichere alle erfolgreichen Trajektorien:

```python
# training/data_synthesis/experience_library.py

class ExperienceLibrary:
    def __init__(self, db_path):
        self.db = sqlite3.connect(db_path)
        self._create_tables()
    
    def add_trajectory(
        self,
        context,
        agent_outputs,
        decision,
        actual_outcome,
        reward,
        judge_score
    ):
        self.db.execute("""
            INSERT INTO trajectories VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(),
            json.dumps(context),
            json.dumps(agent_outputs),
            json.dumps(decision),
            actual_outcome,
            reward,
            judge_score
        ))
        self.db.commit()
    
    def get_top_trajectories(self, min_reward=0.7, limit=1000):
        cursor = self.db.execute("""
            SELECT * FROM trajectories
            WHERE reward >= ?
            ORDER BY reward DESC
            LIMIT ?
        """, (min_reward, limit))
        
        return cursor.fetchall()
```

### 4.2 Synthese-Prozess

Generiere neue SFT-Daten aus erfolgreichen Trajektorien:

```bash
python training/data_synthesis/synthesize_trajectories.py \
    --experience-db data/trajectories/experience.db \
    --min-reward 0.7 \
    --output data/processed/synthetic_sft.jsonl \
    --augment  # Error Healing aktivieren
```

**Error Healing**:

```python
def heal_trajectory(failed_traj, teacher_model):
    """
    Korrigiere fehlerhafte Trajektorien
    """
    
    prompt = f"""
    Die folgende Analyse führte zu einem schlechten Ergebnis (Reward: {failed_traj.reward}):
    
    Context: {failed_traj.context}
    Analysis: {failed_traj.decision}
    Actual Outcome: {failed_traj.actual_outcome}
    
    Was war der Fehler? Wie sollte die korrekte Analyse lauten?
    """
    
    corrected = teacher_model.generate(prompt)
    
    return {
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': failed_traj.context},
            {'role': 'assistant', 'content': corrected}
        ],
        'metadata': {
            'healed': True,
            'original_reward': failed_traj.reward
        }
    }
```

### 4.3 Incremental Updates

Regelmäßige Model-Updates:

```bash
# Wöchentlicher Cron-Job
python scripts/incremental_update.py \
    --agent strategist \
    --experience-db data/trajectories/experience.db \
    --base-model models/strategist_grpo/final \
    --output models/strategist_v2
```

## Best Practices

### Memory Management

**Für GPUs <16GB**:
```yaml
model:
  load_in_4bit: true
  use_flash_attention_2: true

training:
  batch_size: 1
  gradient_accumulation_steps: 16
  gradient_checkpointing: true
```

**Für GPUs 16-24GB**:
```yaml
model:
  load_in_4bit: true

training:
  batch_size: 2
  gradient_accumulation_steps: 8
```

**Für GPUs >24GB**:
```yaml
model:
  load_in_4bit: false  # Nutze fp16

training:
  batch_size: 4
  gradient_accumulation_steps: 4
```

### Preventing Reward Hacking

**1. KL-Divergenz Penalty**:
```yaml
training:
  kl_penalty: 0.1  # Verhindert zu starke Abweichung
```

**2. Brevity Penalty**:
```python
if len(output_tokens) > 500:
    reward -= 0.01 * (len(output_tokens) - 500)
```

**3. Multi-Judge Consensus**:
```python
judges = [ClaudeJudge(), GPT4Judge(), DeepSeekJudge()]
scores = [judge.evaluate(output) for judge in judges]
final_score = np.median(scores)  # Robust gegen Outlier
```

### Hyperparameter Tuning

**Learning Rate**:
- SFT: 2e-4 (Standard)
- RL: 1e-5 (niedriger für Stabilität)
- Supervisor: 1e-3 (höher OK für kleines Modell)

**Batch Size**:
- Start klein (2-4)
- Erhöhe gradient_accumulation_steps statt batch_size
- Effective batch = batch_size * grad_accum_steps

**Epochs**:
- SFT: 3 Epochen (mehr kann zu Overfitting führen)
- RL: 1-3 Epochen
- Monitor Validation-Loss

## Monitoring

### Wandb Integration

```python
import wandb

wandb.init(
    project="stock-agent-system",
    name=f"news-agent-{timestamp}",
    config=config
)

# Log metrics
wandb.log({
    'train_loss': loss,
    'val_loss': val_loss,
    'learning_rate': lr,
    'epoch': epoch
})
```

### Key Metrics

**SFT**:
- Training/Validation Loss
- Perplexity
- F1-Score (wenn Labels vorhanden)

**RL**:
- Average Reward
- KL Divergenz (vs. Reference Model)
- Sharpe Ratio (Backtesting)
- Win Rate

**Supervisor**:
- Routing Accuracy
- Cumulative Regret
- Latency

## Troubleshooting

### Loss explodiert

**Symptom**: Loss steigt plötzlich stark an

**Lösung**:
```yaml
training:
  learning_rate: 5e-5  # Reduziere LR
  gradient_clipping: 1.0  # Clip gradients
```

### Keine Verbesserung

**Symptom**: Loss stagniert

**Lösungen**:
1. Erhöhe Learning Rate
2. Prüfe Datenqualität
3. Verlängere Warmup
4. Reduziere Regularization

### CUDA OOM

**Lösungen**:
```python
# 1. Batch Size reduzieren
batch_size: 1

# 2. Gradient Checkpointing
gradient_checkpointing: true

# 3. 4-bit Quantization
load_in_4bit: true

# 4. CPU Offloading
device_map: 'auto'
```

### Reward Hacking

**Symptom**: Reward steigt, aber Qualität sinkt

**Lösungen**:
1. Erhöhe KL-Penalty
2. Nutze Multi-Judge System
3. Füge Diversity-Bonus hinzu
4. Regularisiere Output-Länge
