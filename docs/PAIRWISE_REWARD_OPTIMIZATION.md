## Pairwise Reward Optimization

## Overview

This document describes the **Pairwise Reward Optimization** framework for training the strategist agent using **relative comparisons** instead of **absolute scores**.

---

## Problem: Absolute Scoring is Unstable

Traditional RL training uses **absolute reward scores** (e.g., 1-10):

```
Strategy → Judge → Score: 7.5/10 → Update model
```

**Problems:**

1. **Inconsistent Scoring**: Judge may give 7.5 today, 6.8 tomorrow for same strategy
2. **Score Inflation**: Scores drift upward over time (reward hacking)
3. **Difficult Calibration**: What does 7.5 mean? Hard to define absolute quality
4. **Complex Reasoning**: Hard to assign single score to multi-faceted strategy
5. **Variance**: High variance in scores reduces training stability

**Example:**
```
Strategy A: "Buy AAPL at $180, stop loss $175, take profit $195"
Judge Score: 7.5/10

Same strategy, different day:
Judge Score: 6.8/10  # Inconsistent!
```

---

## Solution: Pairwise Comparisons

Instead of absolute scores, use **relative comparisons**:

```
Strategy A vs Strategy B → Judge → "A is better" → Update model
```

**Advantages:**

1. **Easier for Judge**: Comparing is easier than scoring
2. **More Consistent**: "A > B" is more stable than "A = 7.5, B = 6.8"
3. **Reduces Reward Hacking**: Can't game relative comparisons as easily
4. **Better for Complex Reasoning**: Easier to compare than assign absolute value
5. **Lower Variance**: Binary comparisons have less variance

**Example:**
```
Strategy A: "Buy AAPL at $180, SL $175, TP $195, detailed reasoning"
Strategy B: "Buy AAPL at $180, no SL/TP, minimal reasoning"

Judge: "A is better" (consistent across evaluations)
```

---

## Architecture

### Components

```
Pairwise Reward Optimization
├── PairwiseJudge
│   ├── compare(strategy_a, strategy_b)
│   └── 5 evaluation criteria
├── PairwiseDataGenerator
│   ├── generate_pair(market_state)
│   └── Uses different temperatures
├── PairwiseRewardModel
│   ├── predict_preference(a, b)
│   └── Learns from comparisons
├── PairwiseTrainingDataset
│   ├── save/load comparisons
│   └── Statistics
└── PairwiseRFTTrainer
    ├── Ranking loss
    └── Train on comparisons
```

---

## How It Works

### 1. Generate Strategy Pairs

For each market state, generate **two different strategies**:

```python
# Use different temperatures for diversity
strategy_a = strategist.synthesize(..., temperature=0.7)  # Conservative
strategy_b = strategist.synthesize(..., temperature=0.9)  # Exploratory
```

**Why different temperatures?**
- Lower temp (0.7): More conservative, safer strategies
- Higher temp (0.9): More exploratory, creative strategies
- Creates meaningful differences to compare

---

### 2. Judge Comparison

Judge compares the two strategies:

```python
comparison = judge.compare(market_state, strategy_a, strategy_b)

# Result:
{
    "winner": "A",  # or "B", "tie", "unclear"
    "reasoning": "Strategy A has better risk management...",
    "confidence": 0.85,
    "criteria_scores": {
        "reasoning_quality": "A",
        "risk_management": "A",
        "confidence_calibration": "B",
        "actionability": "A",
        "market_context_awareness": "A"
    }
}
```

**5 Evaluation Criteria:**

1. **Reasoning Quality**: Is reasoning logical, comprehensive, well-supported?
2. **Risk Management**: Are stop-loss, position sizing, risk assessment appropriate?
3. **Confidence Calibration**: Is confidence justified by analysis?
4. **Actionability**: Is strategy clear and executable?
5. **Market Context Awareness**: Does it consider market conditions?

---

### 3. Compute Pairwise Rewards

Convert comparison to rewards:

```python
if winner == "A":
    reward_a = +1.0
    reward_b = -1.0
elif winner == "B":
    reward_a = -1.0
    reward_b = +1.0
elif winner == "tie":
    reward_a = 0.0
    reward_b = 0.0
```

**Simple and stable!**

---

### 4. Train with Ranking Loss

Update model to prefer winner:

```python
# Ranking loss: Encourage winner to score higher
loss = max(0, margin - (score_winner - score_loser))

# Example:
score_a = model(strategy_a)  # 0.6
score_b = model(strategy_b)  # 0.8
margin = 0.5

# A is winner, but B scores higher → Loss!
loss = max(0, 0.5 - (0.6 - 0.8)) = 0.7

# Update model to increase score_a, decrease score_b
```

**Margin**: Minimum difference between winner and loser (e.g., 0.5)

---

## Comparison with Absolute Scoring

| Aspect | Absolute Scoring | Pairwise Comparison |
|--------|------------------|---------------------|
| **Task** | "Score this strategy 1-10" | "Which is better: A or B?" |
| **Consistency** | ⚠️ Variable (7.5 → 6.8) | ✅ Stable ("A > B") |
| **Ease** | ❌ Hard to calibrate | ✅ Easier to compare |
| **Reward Hacking** | ⚠️ Score inflation | ✅ Harder to game |
| **Variance** | ⚠️ High | ✅ Low |
| **Complex Reasoning** | ❌ Hard to score | ✅ Easier to compare |
| **Training Stability** | ⚠️ Moderate | ✅ High |

---

## Implementation

### 1. Generate Comparison Data

```python
from training.pairwise import (
    PairwiseJudge,
    PairwiseDataGenerator,
    PairwiseTrainingDataset
)

# Initialize
judge = PairwiseJudge(config={'model': 'gpt-4.1-mini'})
generator = PairwiseDataGenerator(strategist_agent)
dataset = PairwiseTrainingDataset(save_path="data/comparisons.jsonl")

# Generate comparisons
for market_state, agent_outputs in market_data:
    # Generate pair
    strategy_a, strategy_b = generator.generate_pair(
        market_state,
        agent_outputs
    )
    
    # Judge comparison
    comparison = judge.compare(market_state, strategy_a, strategy_b)
    
    # Add to dataset
    dataset.add_comparison(comparison)

# Save
dataset.save()
```

---

### 2. Train with Pairwise RFT

```python
from training.rl.train_strategist_pairwise_rft import (
    PairwiseRFTTrainer,
    PairwiseRFTConfig
)

# Config
config = PairwiseRFTConfig()
config.num_epochs = 3
config.batch_size = 4
config.margin = 0.5

# Load dataset
dataset = PairwiseTrainingDataset()
dataset.load()

# Split train/eval
train_dataset, eval_dataset = split_dataset(dataset)

# Trainer
trainer = PairwiseRFTTrainer(model, tokenizer, config)

# Train
trainer.train(train_dataset, eval_dataset)
```

---

### 3. Use Reward Model for Inference

```python
from training.pairwise import PairwiseRewardModel

# Load trained reward model
reward_model = PairwiseRewardModel()

# Score strategies
score_a, score_b = reward_model.predict_preference(
    strategy_a,
    strategy_b,
    market_state
)

# Pick better strategy
if score_a > score_b:
    final_strategy = strategy_a
else:
    final_strategy = strategy_b
```

---

## Training Pipeline

### Full Pipeline

```
1. Generate Market States
   ↓
2. Generate Strategy Pairs (temp 0.7 vs 0.9)
   ↓
3. Judge Comparisons (5 criteria)
   ↓
4. Save to Dataset
   ↓
5. Train Reward Model (ranking loss)
   ↓
6. Fine-tune Strategist (prefer winners)
   ↓
7. Evaluate & Iterate
```

### Data Collection

```bash
# Generate 1000 pairwise comparisons
python scripts/generate_pairwise_data.py \
    --num_comparisons 1000 \
    --output_dir data/pairwise_comparisons
```

### Training

```bash
# Train strategist with pairwise RFT
python training/rl/train_strategist_pairwise_rft.py \
    --config config/rl/pairwise_rft.yaml \
    --data_dir data/pairwise_comparisons \
    --output_dir models/strategist_pairwise_rft
```

---

## Evaluation Criteria Details

### 1. Reasoning Quality

**What to evaluate:**
- Is reasoning logical and coherent?
- Are claims supported by evidence?
- Is analysis comprehensive?
- Are multiple factors considered?

**Good Example:**
```
"Based on strong earnings beat (+15% vs expectations), positive analyst 
upgrades (3 in past week), and bullish technical breakout above $180 
resistance, recommend BUY. Risk-reward ratio of 1:2.5 is favorable."
```

**Poor Example:**
```
"Stock looks good. Buy."
```

---

### 2. Risk Management

**What to evaluate:**
- Is stop-loss defined?
- Is take-profit defined?
- Is position sizing appropriate?
- Is risk assessment provided?

**Good Example:**
```
recommendation: buy
position_size: 8%
stop_loss: $175 (3.3% below entry)
take_profit: $195 (8.3% above entry)
risk_assessment: "Moderate risk. R:R = 1:2.5. Max loss = 0.26% of portfolio."
```

**Poor Example:**
```
recommendation: buy
position_size: 15%
stop_loss: None
take_profit: None
risk_assessment: ""
```

---

### 3. Confidence Calibration

**What to evaluate:**
- Is confidence justified by analysis?
- Is confidence appropriate for uncertainty?
- Are caveats mentioned?

**Good Example:**
```
confidence: 0.75
reasoning: "Strong signals, but market volatility is elevated. Confidence 
tempered by macro uncertainty."
```

**Poor Example:**
```
confidence: 0.99
reasoning: "Looks good."
```

---

### 4. Actionability

**What to evaluate:**
- Is strategy clear and executable?
- Are entry/exit points defined?
- Is timing specified?

**Good Example:**
```
recommendation: buy
entry_target: $178.00 (limit order)
stop_loss: $175.00
take_profit: $195.00
timing: "Enter on pullback to $178, or market order if breaks above $182"
```

**Poor Example:**
```
recommendation: buy
entry_target: None
```

---

### 5. Market Context Awareness

**What to evaluate:**
- Does strategy consider current market conditions?
- Are volatility, trend, news mentioned?
- Is sector/macro context considered?

**Good Example:**
```
"In current high-volatility environment (VIX 25), recommend smaller position 
size (8% vs usual 10%). Tech sector rotation favors quality names like AAPL. 
Fed policy uncertainty requires wider stop-loss."
```

**Poor Example:**
```
"Buy AAPL."
```

---

## Ranking Loss

### Formula

```python
loss = max(0, margin - (score_winner - score_loser))
```

**Components:**
- `score_winner`: Model's score for winning strategy
- `score_loser`: Model's score for losing strategy
- `margin`: Minimum desired difference (e.g., 0.5)

### Example

```python
# Comparison: A is better than B
strategy_a: winner
strategy_b: loser

# Model scores (before training)
score_a = 0.6
score_b = 0.8  # Wrong! B scores higher

# Loss
margin = 0.5
loss = max(0, 0.5 - (0.6 - 0.8))
     = max(0, 0.5 - (-0.2))
     = max(0, 0.7)
     = 0.7  # High loss!

# After training
score_a = 0.9  # Increased
score_b = 0.3  # Decreased

# Loss
loss = max(0, 0.5 - (0.9 - 0.3))
     = max(0, 0.5 - 0.6)
     = max(0, -0.1)
     = 0.0  # No loss! Correct ranking.
```

---

## Best Practices

### 1. Use Diverse Strategy Pairs

```python
# Good: Different temperatures
strategy_a = strategist.synthesize(..., temperature=0.7)
strategy_b = strategist.synthesize(..., temperature=0.9)

# Bad: Same temperature
strategy_a = strategist.synthesize(..., temperature=0.8)
strategy_b = strategist.synthesize(..., temperature=0.8)
# May generate very similar strategies
```

### 2. Filter Unclear Comparisons

```python
# Skip unclear comparisons during training
if comparison.winner == ComparisonResult.UNCLEAR:
    continue  # Don't use for training
```

### 3. Monitor Judge Confidence

```python
# Track judge confidence
avg_confidence = np.mean([c.confidence for c in comparisons])

if avg_confidence < 0.6:
    logger.warning("Low judge confidence. Review comparison quality.")
```

### 4. Balance Winner Distribution

```python
# Check winner distribution
stats = dataset.get_statistics()
winner_dist = stats['winner_distribution']

# Ideally: ~50% A, ~50% B (balanced)
# If skewed (e.g., 90% A), review data generation
```

### 5. Use Appropriate Margin

```python
# Margin controls strictness
margin = 0.5  # Standard

# Larger margin: Stricter (winner must be much better)
margin = 1.0

# Smaller margin: More lenient
margin = 0.2
```

---

## Metrics & Monitoring

### Key Metrics

```python
# Dataset metrics
metrics = {
    'total_comparisons': len(dataset.comparisons),
    'winner_distribution': {
        'A_better': count_a,
        'B_better': count_b,
        'tie': count_tie,
        'unclear': count_unclear
    },
    'avg_judge_confidence': np.mean([c.confidence for c in comparisons]),
    'criteria_agreement': {
        'reasoning_quality': agreement_rate,
        'risk_management': agreement_rate,
        ...
    }
}

# Training metrics
metrics = {
    'train/loss': loss,
    'train/ranking_accuracy': accuracy,  # % of correct rankings
    'eval/loss': eval_loss,
    'eval/ranking_accuracy': eval_accuracy
}
```

### Dashboard

```python
# Log to WandB
wandb.log({
    'pairwise/total_comparisons': len(dataset.comparisons),
    'pairwise/avg_confidence': avg_confidence,
    'pairwise/winner_A_rate': winner_a_rate,
    'pairwise/winner_B_rate': winner_b_rate,
    'train/ranking_loss': loss,
    'train/ranking_accuracy': accuracy
})
```

---

## Comparison with Other Methods

### vs. Absolute Scoring (Standard RL)

| Method | Pairwise | Absolute |
|--------|----------|----------|
| **Consistency** | ✅ High | ⚠️ Moderate |
| **Ease** | ✅ Easy | ❌ Hard |
| **Stability** | ✅ Stable | ⚠️ Variable |
| **Reward Hacking** | ✅ Resistant | ⚠️ Vulnerable |

**Verdict:** Pairwise is better for complex reasoning tasks

---

### vs. GRPO (Group Relative Policy Optimization)

| Method | Pairwise | GRPO |
|--------|----------|------|
| **Comparison** | Binary (A vs B) | Group (rank all) |
| **Data Efficiency** | ⚠️ Moderate | ✅ High |
| **Simplicity** | ✅ Simple | ⚠️ Complex |
| **Judge Difficulty** | ✅ Easy | ⚠️ Hard |

**Verdict:** Pairwise is simpler, GRPO is more data-efficient

---

### vs. PPO (Proximal Policy Optimization)

| Method | Pairwise | PPO |
|--------|----------|-----|
| **Reward Type** | Relative | Absolute |
| **Stability** | ✅ High | ⚠️ Moderate |
| **Sample Efficiency** | ⚠️ Moderate | ✅ High |
| **Implementation** | ✅ Simple | ⚠️ Complex |

**Verdict:** Pairwise is more stable, PPO is more sample-efficient

---

## Research Background

Based on:

1. **InstructGPT** (OpenAI, 2022)
   - Pioneered pairwise comparisons for RLHF
   - Showed pairwise is more stable than absolute scoring

2. **Constitutional AI** (Anthropic, 2022)
   - Used pairwise for harmlessness training
   - Demonstrated consistency benefits

3. **RLAIF** (Google, 2023)
   - RL from AI Feedback using pairwise
   - Showed AI judges can replace humans

4. **DPO** (Direct Preference Optimization, 2023)
   - Simplified pairwise training
   - Removed need for separate reward model

---

## Future Enhancements

### 1. Multi-Way Comparisons

Instead of pairs, compare 3+ strategies:

```python
# Rank A, B, C
ranking = judge.rank([strategy_a, strategy_b, strategy_c])
# Result: [A, C, B]  # A best, B worst
```

**Benefit:** More data-efficient

---

### 2. Learned Judge

Train a neural network judge:

```python
# Instead of LLM judge
judge_model = NeuralJudge()
judge_model.train(human_comparisons)

# Use for comparisons
comparison = judge_model.compare(strategy_a, strategy_b)
```

**Benefit:** Faster, cheaper

---

### 3. Active Learning

Select most informative pairs to compare:

```python
# Uncertainty-based selection
uncertainty = model.uncertainty(strategy_a, strategy_b)

if uncertainty > threshold:
    # Get judge comparison (expensive)
    comparison = judge.compare(strategy_a, strategy_b)
else:
    # Skip (model is confident)
    pass
```

**Benefit:** Reduces judge calls

---

### 4. Ensemble Judges

Use multiple judges and aggregate:

```python
judges = [judge_1, judge_2, judge_3]

comparisons = [j.compare(a, b) for j in judges]

# Majority vote
winner = majority_vote([c.winner for c in comparisons])
```

**Benefit:** More robust

---

## Troubleshooting

### Issue: Judge always picks A

**Cause:** Strategies too similar, or judge bias

**Solution:**
- Increase temperature difference (0.6 vs 1.0)
- Review judge prompt for bias
- Check if data generation is correct

---

### Issue: High unclear rate

**Cause:** Strategies are very similar

**Solution:**
- Increase temperature difference
- Use different sampling strategies (top-k, nucleus)
- Generate more diverse market states

---

### Issue: Training loss not decreasing

**Cause:** Model capacity, learning rate, or data quality

**Solution:**
- Increase model size
- Tune learning rate
- Filter low-confidence comparisons
- Check data quality

---

## Summary

**Pairwise Reward Optimization** provides:

- ✅ **More stable training** than absolute scoring
- ✅ **Easier for judge** to compare than score
- ✅ **More consistent** judgments
- ✅ **Resistant to reward hacking**
- ✅ **Better for complex reasoning**

**Key Components:**
- PairwiseJudge (5 criteria)
- PairwiseDataGenerator (diverse pairs)
- PairwiseRewardModel (preference prediction)
- PairwiseRFTTrainer (ranking loss)

**Use Cases:**
- Training strategist agent
- Evaluating strategy quality
- Comparing different models
- Active learning

**This is a critical component for stable, high-quality RL training!** 🚀

---

## References

- `training/pairwise/pairwise_comparison.py` - Main implementation
- `training/rl/train_strategist_pairwise_rft.py` - Training pipeline
- `tests/unit/test_pairwise_comparison.py` - Unit tests
- Research: InstructGPT (OpenAI, 2022)
- Research: Constitutional AI (Anthropic, 2022)
- Research: RLAIF (Google, 2023)

---

## Contact

For questions or issues:
- Open an issue on GitHub
- See `CONTRIBUTING.md`
