## Regime Adaptation & Non-Stationary Market Handling

## Overview

This document describes the system's ability to adapt to **non-stationary market environments** and **regime changes**, a critical capability for real-world trading systems.

---

## Problem: Non-Stationary Markets

Stock markets are **non-stationary**: their statistical properties change over time. Key challenges:

1. **Regime Changes**: Sudden shifts (bull → crash, low vol → high vol)
2. **Agent Performance Variability**: Technical indicators work in trending markets but fail in volatile crashes
3. **Stale Models**: Models trained on past data become obsolete
4. **Exploration-Exploitation Tradeoff**: Need to quickly adapt to new regimes

**Example:**
- **Bull Market**: Technical agent performs well (RSI, MACD reliable)
- **Sudden Crash**: Technical indicators lag, News agent becomes critical
- **Challenge**: Supervisor must detect regime change and re-prioritize agents **quickly**

---

## Solution: Multi-Layer Adaptation

### 1. Enhanced Neural-UCB Supervisor

**File:** `agents/supervisor/enhanced_neural_ucb.py`

**Key Features:**

#### A. Uncertainty Quantification
- **Epistemic Uncertainty**: Model uncertainty (lack of knowledge)
- **Aleatoric Uncertainty**: Data uncertainty (inherent randomness)
- **Bayesian Neural Network**: Quantifies uncertainty explicitly

```python
# Get predictions with uncertainty
mean_rewards, std_rewards = model(context, num_samples=10)

# UCB score = mean + exploration_factor * (uncertainty + count_bonus)
ucb_scores = mean_rewards + exploration_factor * (std_rewards + count_bonus)
```

**Why it matters:**
- High uncertainty → More exploration
- Low uncertainty → More exploitation
- Adapts exploration based on confidence

#### B. Sliding Window for Reward Estimation
- Tracks recent rewards only (window size: 100)
- Handles non-stationarity by forgetting old data
- Focuses on current regime

```python
self.reward_windows = {i: deque(maxlen=100) for i in range(num_agents)}
```

#### C. Change Point Detection (CUSUM)
- Detects when reward distribution changes
- Triggers regime adaptation

```python
class ChangePointDetector:
    def update(self, value: float) -> bool:
        # CUSUM algorithm
        if self.cusum_pos > threshold:
            return True  # Change detected!
```

**When change detected:**
1. Increase exploration factor (2x)
2. Increase learning rate (2x)
3. Clear old reward windows
4. Force re-evaluation of all agents

#### D. Adaptive Learning Rate
- Base learning rate: 1e-3
- After regime change: 2e-3 (temporary boost)
- Enables fast adaptation

---

### 2. Market Regime Detection

**File:** `utils/market_regime_detector.py`

**Detects 3 Regime Dimensions:**

#### A. Volatility Regime
- **LOW** (< 0.25): Stable, predictable
- **MEDIUM** (0.25-0.5): Normal
- **HIGH** (0.5-0.75): Elevated risk
- **EXTREME** (> 0.75): Crisis/crash

#### B. Trend Regime
- **STRONG_BULL** (> 0.6): Strong uptrend
- **BULL** (0.2-0.6): Uptrend
- **SIDEWAYS** (-0.2 to 0.2): Range-bound
- **BEAR** (-0.6 to -0.2): Downtrend
- **STRONG_BEAR** (< -0.6): Strong downtrend

#### C. News Impact Regime
- **LOW** (< 0.3): Fundamentals-driven
- **MEDIUM** (0.3-0.7): Balanced
- **HIGH** (> 0.7): News-driven

**Overall Regime Classification:**
- `crash`: Extreme vol + high news impact
- `news_driven_volatility`: High vol + high news
- `strong_trend_low_vol`: Strong trend + low vol
- `bull_market`, `bear_market`, `range_bound`, etc.

**Agent Recommendations:**

| Regime | Recommended Agents | Reasoning |
|--------|-------------------|-----------|
| Crash | News, Fundamental | News drives market, technicals unreliable |
| High Volatility | News, Fundamental | Volatility makes technicals noisy |
| Strong Trend + Low Vol | Technical, Fundamental | Technicals work well in trends |
| Sideways | Technical, Fundamental | Range trading, support/resistance |
| High News Impact | News, Fundamental | News sentiment is key driver |

---

### 3. Adaptive Agent Router

**File:** `utils/market_regime_detector.py` (class `AdaptiveAgentRouter`)

**Combines:**
- Supervisor's learned preferences
- Regime detector's recommendations
- Regime change detection

**Flow:**
1. Detect current regime
2. Get supervisor's agent selection
3. Check if regime changed
4. If changed: trigger adaptation
5. Combine supervisor + regime recommendations

```python
router = AdaptiveAgentRouter(supervisor, regime_detector)

routing = router.route(
    market_data=market_data,
    portfolio_state=portfolio_state,
    use_regime_detection=True
)

# Returns:
# {
#     'active_agents': ['news', 'fundamental'],
#     'uncertainties': {...},
#     'regime': {...},
#     'regime_changed': True
# }
```

---

## Stress Tests

**File:** `tests/stress/test_regime_adaptation.py`

**15+ Test Cases:**

### 1. Single Regime Tests
- `test_bull_market_routing`: Technical agent preferred in bull market
- `test_crash_routing`: News agent preferred in crash

### 2. Regime Change Tests (CRITICAL!)
- `test_bull_to_crash_adaptation`: Bull → Crash adaptation
- `test_low_to_high_volatility_adaptation`: Low vol → High vol
- `test_trending_to_range_bound_adaptation`: Trending → Sideways

**Success Criteria:**
- Phase 1: Correct agent selected >60%
- Phase 2: Adapts to new optimal agent >60%

### 3. Adaptation Speed Tests
- `test_adaptation_speed`: Measures iterations needed to adapt
- **Target**: Adapt within 10 iterations

### 4. Uncertainty Tests
- `test_uncertainty_increases_after_regime_change`: Uncertainty spikes after change
- Triggers more exploration

### 5. Exploration vs Exploitation Tests
- `test_exploration_in_new_regime`: More exploration in new regimes
- Measures agent diversity

---

## Usage

### 1. Run Stress Tests

```bash
# Run all regime adaptation tests
pytest tests/stress/test_regime_adaptation.py -v

# Run specific test
pytest tests/stress/test_regime_adaptation.py::TestRegimeAdaptation::test_bull_to_crash_adaptation -v
```

**Expected Results:**
- Bull market: Technical selected >60%
- Crash: News selected >60%
- Adaptation: Within 10 iterations
- Uncertainty: Increases after regime change

### 2. Use Enhanced Supervisor

```python
from agents.supervisor.enhanced_neural_ucb import EnhancedSupervisorAgent

config = {
    'context_dim': 16,
    'num_agents': 3,
    'use_bayesian': True,
    'exploration_factor': 2.0,
    'window_size': 100
}

supervisor = EnhancedSupervisorAgent(config)

# Select agents
selected, uncertainties = supervisor.select_agents(
    context=market_data,
    explore=True
)

# Update with reward
supervisor.update(
    context=market_data,
    agent='news',
    reward=0.85
)

# Get statistics
stats = supervisor.get_agent_statistics()
print(f"Exploration factor: {stats['exploration_factor']}")
print(f"Avg uncertainty: {stats['avg_uncertainty']}")
```

### 3. Use Regime Detection

```python
from utils.market_regime_detector import detect_regime

market_data = {
    'volatility': 0.85,
    'trend_strength': -0.7,
    'news_impact': 0.9
}

regime = detect_regime(market_data)

print(f"Overall Regime: {regime['overall_regime']}")
print(f"Recommended Agents: {regime['recommended_agents']}")
print(f"Reasoning: {regime['reasoning']}")
print(f"Confidence: {regime['confidence']:.2%}")
```

**Output:**
```
Overall Regime: crash
Recommended Agents: ['news', 'fundamental']
Reasoning: Extreme volatility detected. Market is likely driven by news/events.
Confidence: 92%
```

### 4. Use Adaptive Router

```python
from utils.market_regime_detector import AdaptiveAgentRouter

router = AdaptiveAgentRouter(supervisor, regime_detector)

routing = router.route(
    market_data=market_data,
    portfolio_state={'cash': 10000},
    use_regime_detection=True
)

print(f"Active Agents: {routing['active_agents']}")
print(f"Regime Changed: {routing['regime_changed']}")
```

---

## Adaptation Mechanisms

### 1. Immediate Adaptation (Change Point Detection)

**Trigger:** CUSUM detects reward distribution change

**Actions:**
1. ✅ Increase exploration factor (2x)
2. ✅ Increase learning rate (2x)
3. ✅ Clear old reward windows
4. ✅ Log regime change event

**Speed:** 1-5 iterations

### 2. Gradual Adaptation (Sliding Window)

**Mechanism:** Only recent 100 rewards used for estimation

**Effect:**
- Old regime data forgotten
- Model focuses on current regime
- Smooth transition

**Speed:** 10-20 iterations

### 3. Regime-Aware Routing

**Mechanism:** Regime detector provides recommendations

**Effect:**
- Supervisor selection weighted with regime recommendations
- If supervisor disagrees with regime, add regime-recommended agent
- Ensemble approach

**Speed:** Immediate (0 iterations)

---

## Performance Metrics

### 1. Adaptation Speed
**Metric:** Iterations needed to switch to optimal agent after regime change

**Target:** ≤ 10 iterations

**Measurement:**
```python
adaptation_point = None
for i in range(len(selections)):
    window = selections[i-5:i]
    if sum(window) / 5 > 0.5:  # 50% selection rate
        adaptation_point = i
        break
```

### 2. Regime Detection Accuracy
**Metric:** % of correct regime classifications

**Target:** > 85%

**Measurement:**
```python
correct = sum(1 for pred, true in zip(predictions, ground_truth) if pred == true)
accuracy = correct / len(predictions)
```

### 3. Agent Selection Accuracy
**Metric:** % of times optimal agent is selected in each regime

**Target:** > 60% (allows for exploration)

**Measurement:**
```python
optimal_selections = sum(1 for agent in selections if agent == optimal_agent)
accuracy = optimal_selections / len(selections)
```

### 4. Uncertainty Calibration
**Metric:** Correlation between uncertainty and prediction error

**Target:** > 0.7 (high correlation)

**Measurement:**
```python
correlation = np.corrcoef(uncertainties, errors)[0, 1]
```

---

## Best Practices

### 1. Monitor Regime Changes

Log regime changes for analysis:

```python
if routing['regime_changed']:
    logger.info(f"Regime change: {old_regime} → {new_regime}")
    wandb.log({
        'regime_change': 1,
        'new_regime': new_regime,
        'exploration_factor': supervisor.exploration_factor
    })
```

### 2. Tune Exploration Factor

Adjust based on market conditions:

```python
# High volatility → More exploration
if market_data['volatility'] > 0.7:
    supervisor.exploration_factor = 3.0
else:
    supervisor.exploration_factor = 2.0
```

### 3. Use Ensemble in Uncertain Regimes

When uncertainty is high, use multiple agents:

```python
if routing['uncertainties']['news']['total'] > 0.5:
    # High uncertainty → Use ensemble
    routing['active_agents'] = ['news', 'technical', 'fundamental']
```

### 4. Backtest on Historical Regime Changes

Test on real historical events:
- 2020 COVID crash
- 2022 Fed rate hikes
- 2023 banking crisis

```python
# Load historical data with known regime changes
for event in historical_regime_changes:
    test_adaptation_on_event(event)
```

---

## Research Background

This implementation is based on research on:

1. **Contextual Bandits for Non-Stationary Environments** (2021)
   - Sliding window approaches
   - Change point detection
   - Adaptive exploration

2. **Neural-UCB** (2020)
   - Uncertainty quantification via neural networks
   - Bayesian deep learning
   - Exploration bonuses

3. **Regime Detection in Financial Markets** (2019)
   - Hidden Markov Models
   - Volatility regimes
   - Adaptive trading strategies

---

## Troubleshooting

### Issue: Slow Adaptation

**Symptom:** Takes >15 iterations to adapt

**Solutions:**
1. Increase learning rate: `learning_rate=2e-3`
2. Reduce window size: `window_size=50`
3. Increase exploration: `exploration_factor=3.0`
4. Lower CUSUM threshold: `threshold=3.0`

### Issue: Too Much Exploration

**Symptom:** Random agent selection, no convergence

**Solutions:**
1. Reduce exploration factor: `exploration_factor=1.0`
2. Increase min_exploration: `min_exploration=0.2`
3. Use regime detection to guide exploration

### Issue: False Regime Change Detections

**Symptom:** CUSUM triggers too often

**Solutions:**
1. Increase CUSUM threshold: `threshold=7.0`
2. Increase drift parameter: `drift=1.0`
3. Use longer confirmation window

---

## Future Enhancements

1. **Meta-Learning for Fast Adaptation**
   - Learn to adapt quickly from few examples
   - Transfer knowledge across regimes

2. **Hierarchical Regime Detection**
   - Macro regimes (bull/bear)
   - Micro regimes (intraday patterns)

3. **Multi-Timescale Adaptation**
   - Fast adaptation (seconds)
   - Slow adaptation (days)

4. **Causal Regime Detection**
   - Identify causal drivers of regime changes
   - Predict regime changes before they happen

---

## References

- `tests/stress/test_regime_adaptation.py` - Stress tests
- `agents/supervisor/enhanced_neural_ucb.py` - Enhanced supervisor
- `utils/market_regime_detector.py` - Regime detection
- Research papers on contextual bandits and non-stationary environments

---

## Metrics Dashboard

Track these metrics in production:

```python
# Log to WandB/MLflow
wandb.log({
    'regime/current': regime['overall_regime'],
    'regime/volatility': regime['volatility_regime'],
    'regime/trend': regime['trend_regime'],
    'regime/confidence': regime['confidence'],
    'supervisor/exploration_factor': supervisor.exploration_factor,
    'supervisor/avg_uncertainty': stats['avg_uncertainty'],
    'supervisor/total_steps': stats['total_steps'],
    'agents/news_selection_rate': stats['news']['count'] / total_steps,
    'agents/technical_selection_rate': stats['technical']['count'] / total_steps,
    'agents/fundamental_selection_rate': stats['fundamental']['count'] / total_steps
})
```

---

## Contact

For questions or issues related to regime adaptation:
- Open an issue on GitHub
- See `CONTRIBUTING.md` for contribution guidelines
