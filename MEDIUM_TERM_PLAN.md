# Medium-Term Plan (1-3 Months)

**Comprehensive guide for achieving medium-term performance goals**

---

## 🎯 Goals

### 1. Accumulate 10,000+ Trajectories ✅
- **Current:** 0
- **Target:** 10,000
- **Strategy:** Continuous collection across diverse market conditions
- **Timeline:** 1-2 months

### 2. Achieve Sharpe Ratio > 1.5 ✅
- **Current:** ~0.5-0.7 (after SFT)
- **Target:** >1.5
- **Strategy:** Iterative RL training with diverse trajectories
- **Timeline:** 2-3 months

### 3. Achieve Win Rate > 55% ✅
- **Current:** ~48-52% (baseline)
- **Target:** >55%
- **Strategy:** Error correction, reward shaping, model improvements
- **Timeline:** 2-3 months

### 4. Publish Performance Benchmarks ✅
- **Deliverable:** Public performance report
- **Content:** Sharpe, Win Rate, Returns, Drawdown, Methodology
- **Timeline:** End of 3-month period

---

## 📋 Implementation Strategy

### Phase 1: Continuous Data Collection (Weeks 1-8)

**Goal:** Accumulate 10,000+ diverse trajectories

**Strategy:**
1. **Diversify Symbol Coverage**
   - 40+ symbols across sectors (Tech, Finance, Consumer, Healthcare, Energy, Industrial)
   - Equal representation from different market caps
   - Include both trending and mean-reverting stocks

2. **Diversify Market Conditions**
   - Bull markets (strong uptrends)
   - Bear markets (downtrends)
   - Sideways/choppy markets
   - High volatility periods
   - Low volatility periods

3. **Collection Schedule**
   - Collect 1,000 trajectories per week
   - Run backtests across multiple time periods
   - Store all trajectories in experience library

**Expected Output:**
- Week 4: 4,000 trajectories
- Week 8: 8,000+ trajectories
- Week 10: 10,000+ trajectories

---

### Phase 2: Iterative Training & Optimization (Weeks 3-12)

**Goal:** Improve Sharpe to >1.5 and Win Rate to >55%

**Strategy:**
1. **Incremental Training**
   - Train every 5 collection batches (~5,000 trajectories)
   - 50-100 RL iterations per training session
   - Use GRPO for memory efficiency

2. **Reward Engineering**
   - **Current:** Simple P&L-based rewards
   - **Improvements:**
     - Sharpe-weighted rewards
     - Risk-adjusted returns
     - Drawdown penalties
     - Position sizing bonuses

3. **Model Improvements**
   - Fine-tune hyperparameters (learning rate, KL coefficient)
   - Experiment with different architectures
   - Ensemble best models
   - A/B test different strategies

4. **Error Analysis**
   - Identify common failure patterns
   - Generate corrective training data
   - Add failure cases to experience library
   - Retrain on corrected examples

**Training Schedule:**
```
Week 3: Training iteration 1 (on 3,000 trajectories)
Week 5: Training iteration 2 (on 5,000 trajectories)
Week 7: Training iteration 3 (on 7,000 trajectories)
Week 9: Training iteration 4 (on 9,000 trajectories)
Week 11: Training iteration 5 (on 10,000+ trajectories)
```

**Expected Progress:**
- Week 3: Sharpe 0.7-0.9, Win Rate 50-52%
- Week 6: Sharpe 1.0-1.2, Win Rate 52-54%
- Week 9: Sharpe 1.3-1.5, Win Rate 54-56%
- Week 12: Sharpe >1.5, Win Rate >55%

---

### Phase 3: Performance Validation (Weeks 10-12)

**Goal:** Validate and document performance

**Strategy:**
1. **Out-of-Sample Testing**
   - Test on recent 90-day period (not in training)
   - Test on different symbols (not in training)
   - Test on different market regimes

2. **Robustness Testing**
   - Stress test with extreme market conditions
   - Test with different position sizes
   - Test with varying risk parameters

3. **Statistical Validation**
   - Bootstrap confidence intervals
   - Monte Carlo simulation
   - Statistical significance tests

4. **Benchmark Comparison**
   - Compare vs. SPY (S&P 500)
   - Compare vs. Buy & Hold
   - Compare vs. other strategies (SMA crossover, etc.)

---

## 🚀 Quick Start

### Option 1: Continuous Training (Recommended)

**Run the continuous training pipeline:**

```bash
# Start continuous training
python scripts/continuous_training.py \
  --target-trajectories 10000 \
  --target-sharpe 1.5 \
  --target-win-rate 0.55 \
  --batch-size 1000 \
  --training-interval 5 \
  --evaluation-interval 10
```

This will:
1. Continuously collect trajectories (1,000 per batch)
2. Train every 5 batches (~5,000 trajectories)
3. Evaluate every 10 batches
4. Auto-publish benchmarks when goals achieved
5. Run until all goals are met

**Monitoring:**

```bash
# In a separate terminal
python scripts/monitor_performance.py \
  --target-trajectories 10000 \
  --target-sharpe 1.5 \
  --target-win-rate 0.55 \
  --refresh 5
```

This provides a live dashboard with:
- Current performance metrics
- Progress towards goals
- Recent performance history
- Status alerts

---

### Option 2: Manual Step-by-Step

#### Step 1: Initial Data Collection

```bash
# Collect first 5,000 trajectories
python -c "
from scripts.continuous_training import ContinuousTrainingPipeline
pipeline = ContinuousTrainingPipeline()
for _ in range(5):
    pipeline.collect_trajectories_batch(batch_size=1000)
"
```

#### Step 2: First Training Iteration

```bash
# Train on collected data
python scripts/run_rl_training.py \
  --num-trajectories 5000 \
  --num-iterations 100 \
  --algorithm grpo
```

#### Step 3: Evaluate Performance

```bash
python scripts/evaluate_models.py \
  --models rl:training_runs/rl_grpo_run_*/models \
  --test-days 90
```

#### Step 4: Repeat Until Goals Met

```bash
# Continue collecting + training + evaluating
# Monitor progress in continuous_training/training_state.json
```

---

## 📊 Performance Tracking

### Key Metrics to Monitor

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Total Trajectories | 10,000 | 0 | ⏳ |
| Sharpe Ratio | >1.5 | ~0.7 | ⏳ |
| Win Rate | >55% | ~50% | ⏳ |
| Total Return | >50% | - | - |
| Max Drawdown | <15% | - | - |
| Profit Factor | >2.0 | - | - |

### Progress Milestones

**Week 4:**
- ✅ 4,000+ trajectories
- 🎯 Sharpe 0.8-1.0
- 🎯 Win Rate 51-53%

**Week 8:**
- ✅ 8,000+ trajectories
- 🎯 Sharpe 1.2-1.4
- 🎯 Win Rate 53-55%

**Week 12:**
- ✅ 10,000+ trajectories
- ✅ Sharpe >1.5
- ✅ Win Rate >55%
- ✅ Benchmarks published

---

## 📈 Expected Results

### Conservative Estimate

| Metric | Value |
|--------|-------|
| Sharpe Ratio | 1.5-1.8 |
| Win Rate | 55-58% |
| Annual Return | 50-70% |
| Max Drawdown | 10-15% |
| Profit Factor | 2.0-2.5 |

### Optimistic Estimate

| Metric | Value |
|--------|-------|
| Sharpe Ratio | 1.8-2.2 |
| Win Rate | 58-62% |
| Annual Return | 70-100% |
| Max Drawdown | 8-12% |
| Profit Factor | 2.5-3.0 |

---

## 🔧 Optimization Strategies

### 1. Data Quality

**Improve trajectory quality:**
- Filter low-quality trajectories (reward < 0.3)
- Prioritize high-Sharpe-ratio periods
- Balance across market regimes
- Remove outliers and anomalies

**Implementation:**
```python
# In continuous_training.py
trajectories = experience_lib.get_top_trajectories(
    n=5000,
    min_reward=0.5,  # Only use successful trajectories
    market_regime='balanced'  # Ensure diversity
)
```

### 2. Reward Engineering

**Improve reward function:**
```python
def compute_reward(pnl, sharpe, drawdown, win_rate):
    # Base reward from P&L
    reward = 0.5 + pnl / 2000

    # Sharpe bonus
    if sharpe > 1.5:
        reward *= 1.3
    elif sharpe > 1.0:
        reward *= 1.15

    # Win rate bonus
    if win_rate > 0.55:
        reward *= 1.2

    # Drawdown penalty
    if drawdown > 0.15:
        reward *= 0.7

    return np.clip(reward, 0, 1)
```

### 3. Training Optimization

**Hyperparameter tuning:**
```yaml
# config/rl/optimized_grpo.yaml
model:
  learning_rate: 5e-6  # Lower for stability
  lora_rank: 16        # Higher for capacity

training:
  batch_size: 32       # Larger for better gradients
  kl_coef: 0.05        # Lower for more exploration
  vf_coef: 0.15        # Higher for better value estimation
  num_iterations: 150  # More iterations
```

### 4. Ensemble Methods

**Combine multiple models:**
```python
# Use ensemble of best 3 models
models = [
    'models/rl_iter_1/final',
    'models/rl_iter_3/final',
    'models/rl_iter_5/final'
]

# Average predictions
decisions = [model.analyze(symbol) for model in models]
final_decision = ensemble_vote(decisions)
```

---

## 🎓 Best Practices

### Data Collection

✅ **Do:**
- Collect across all market conditions
- Use diverse symbols (40+ stocks)
- Include different time periods
- Store all metadata (volatility, regime, etc.)

❌ **Don't:**
- Only collect in bull markets
- Focus on single sector
- Ignore failed trajectories
- Skip data validation

### Training

✅ **Do:**
- Train incrementally (every 5k trajectories)
- Monitor validation metrics
- Use early stopping
- Save all checkpoints

❌ **Don't:**
- Wait until all 10k trajectories collected
- Ignore overfitting signs
- Train without evaluation
- Discard intermediate models

### Evaluation

✅ **Do:**
- Test on out-of-sample data
- Use multiple metrics (Sharpe, Win Rate, etc.)
- Compare with benchmarks
- Validate statistical significance

❌ **Don't:**
- Test on training data
- Cherry-pick best results
- Ignore drawdown
- Skip robustness tests

---

## 🚨 Common Issues & Solutions

### Issue 1: Sharpe Plateau (~1.0-1.2)

**Symptoms:**
- Sharpe stuck at 1.0-1.2 for multiple iterations
- Win rate improves but returns don't

**Solutions:**
1. Increase training data diversity
2. Adjust reward function (emphasize Sharpe)
3. Try different RL algorithms (GRPO → PPO)
4. Ensemble multiple models

### Issue 2: High Win Rate but Low Sharpe

**Symptoms:**
- Win rate >55% but Sharpe <1.5
- Small wins, large losses

**Solutions:**
1. Improve position sizing logic
2. Tighten stop-loss rules
3. Increase risk-reward ratio
4. Filter low-confidence trades

### Issue 3: Overfitting

**Symptoms:**
- Great training performance, poor test performance
- Sharpe drops significantly on new data

**Solutions:**
1. Reduce model complexity (lower LoRA rank)
2. Increase regularization (higher dropout)
3. Use more diverse training data
4. Implement early stopping

### Issue 4: Slow Progress

**Symptoms:**
- Taking >3 months to reach goals
- Marginal improvements per iteration

**Solutions:**
1. Increase batch size
2. More aggressive learning rate
3. Collect higher-quality trajectories
4. Improve reward function

---

## 📊 Benchmark Publishing Template

When goals are achieved, publish benchmarks using this template:

```markdown
# Stock Agent System - Performance Benchmarks

**Published:** YYYY-MM-DD
**Period:** 3 months (MM/DD/YYYY - MM/DD/YYYY)

## Summary

- **Total Trajectories:** 10,000+
- **Training Iterations:** 15+
- **Sharpe Ratio:** 1.65
- **Win Rate:** 57.3%
- **Annual Return:** 68.2%
- **Max Drawdown:** 11.8%

## Methodology

- **Data:** 40+ stocks across 6 sectors
- **Period:** 12 months historical data
- **Training:** GRPO with LoRA (8B Llama 3.1)
- **Evaluation:** 90-day out-of-sample test

## Performance Metrics

| Metric | Value | Benchmark (SPY) |
|--------|-------|-----------------|
| Sharpe Ratio | 1.65 | 0.92 |
| Win Rate | 57.3% | - |
| Annual Return | 68.2% | 24.1% |
| Max Drawdown | -11.8% | -18.3% |
| Profit Factor | 2.34 | - |

## Reproducibility

Code and models available at: [GitHub URL]
```

---

## ✅ Completion Checklist

### Data Collection
- [ ] 10,000+ trajectories collected
- [ ] 40+ unique symbols
- [ ] Multiple market regimes
- [ ] Data quality validated

### Training
- [ ] 10+ training iterations completed
- [ ] Checkpoints saved
- [ ] W&B tracking enabled
- [ ] No overfitting detected

### Performance
- [ ] Sharpe Ratio >1.5 achieved
- [ ] Win Rate >55% achieved
- [ ] Out-of-sample validated
- [ ] Benchmarks compared

### Documentation
- [ ] Performance report written
- [ ] Methodology documented
- [ ] Results published
- [ ] Code released

---

## 🎯 Success Criteria

All of the following must be achieved:

1. ✅ **Accumulated 10,000+ trajectories**
   - Verified in experience library
   - Diverse across symbols and regimes

2. ✅ **Sharpe Ratio >1.5**
   - On 90-day out-of-sample test
   - Statistically significant

3. ✅ **Win Rate >55%**
   - On out-of-sample test
   - Consistent across symbols

4. ✅ **Benchmarks Published**
   - Full performance report
   - Methodology documented
   - Results reproducible

---

**Status:** 🚀 **Ready to Execute**

**Estimated Time:** 2-3 months with continuous training

**Next Steps:**
1. Run `python scripts/continuous_training.py`
2. Monitor with `python scripts/monitor_performance.py`
3. Iterate until all goals achieved
4. Publish benchmarks

---

**Last Updated:** 2026-01-04
