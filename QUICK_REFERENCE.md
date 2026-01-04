# Quick Reference Guide

**Fast access to common commands and workflows**

---

## 🚀 Quick Start Commands

### Short-Term (1-2 Weeks)

```bash
# Complete 1-2 week pipeline
python scripts/run_complete_training.py \
  --sft-examples 100 \
  --rl-trajectories 500 \
  --auto-deploy

# Monitor with W&B
wandb ui
```

### Medium-Term (1-3 Months)

```bash
# Start continuous training
python scripts/continuous_training.py \
  --target-trajectories 10000 \
  --target-sharpe 1.5 \
  --target-win-rate 0.55

# Monitor in separate terminal
python scripts/monitor_performance.py
```

---

## 📊 Training Workflows

### SFT Training Only

```bash
# Generate data + train all agents
python scripts/run_sft_training.py \
  --num-examples 100 \
  --max-steps 500

# Or train individual agent
python training/sft/train_news_agent.py --config config/sft/news_agent.yaml
```

### RL Training Only

```bash
# Collect + train strategist
python scripts/run_rl_training.py \
  --num-trajectories 500 \
  --num-iterations 100 \
  --algorithm grpo
```

### Evaluation

```bash
# Evaluate models
python scripts/evaluate_models.py \
  --models \
    baseline:models/baseline \
    sft:training_runs/sft_run_*/models \
  --test-days 90
```

### Deployment

```bash
# Deploy validated models
python scripts/deploy_models.py \
  --models \
    news:path/to/news/final \
    technical:path/to/technical/final \
    fundamental:path/to/fundamental/final \
    strategist:path/to/strategist/final \
  --min-sharpe 0.5
```

---

## 💡 Common Tasks

### Run Analysis

```python
from orchestration.coordinator import SystemCoordinator

coordinator = SystemCoordinator()
result = coordinator.analyze_symbol('AAPL')

print(f"Recommendation: {result['recommendation']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### Check Progress

```bash
# View training state
cat continuous_training/training_state.json | jq '.'

# View benchmarks
cat continuous_training/performance_benchmarks.md
```

### View Logs

```bash
# Training logs
tail -f logs/training.log

# System logs
tail -f logs/system.log
```

---

## 🔍 Troubleshooting

### Out of Memory

```bash
# Use GRPO instead of PPO
--algorithm grpo

# Reduce batch size
# Edit config file:
training:
  batch_size: 2  # Reduce from 4
```

### Training Slow

```bash
# Reduce max steps
--sft-steps 300  # Instead of 500

# Reduce trajectories
--rl-trajectories 300  # Instead of 500
```

### Poor Performance

```bash
# Check data quality
python -c "
from training.data_synthesis.experience_library import ExperienceLibrary
lib = ExperienceLibrary()
stats = lib.get_statistics()
print(stats)
"

# Re-evaluate
python scripts/evaluate_models.py --test-days 90
```

---

## 📁 Important Directories

```
training_runs/           # All training outputs
├── sft_run_*/          # SFT training results
│   ├── data/           # Generated training data
│   └── models/         # Trained models
└── rl_grpo_run_*/      # RL training results
    ├── models/         # Trained strategist
    └── experience_library.db

continuous_training/     # Continuous training state
├── training_state.json # Current progress
├── experience_library.db
└── performance_benchmarks.md

models/
├── production/         # Deployed models
├── backups/           # Model backups
└── baseline/          # Baseline for comparison

evaluation_results/     # Evaluation outputs
└── eval_*/
    ├── evaluation_report.md
    ├── comparison_plots.png
    └── evaluation_results.json
```

---

## 🎯 Goal Tracking

### Short-Term (1-2 Weeks)

| Goal | Command | Check |
|------|---------|-------|
| SFT Training | `run_sft_training.py` | Check `training_runs/sft_run_*/` |
| RL Training | `run_rl_training.py` | Check `training_runs/rl_grpo_run_*/` |
| Evaluation | `evaluate_models.py` | Check `evaluation_results/` |
| Deployment | `deploy_models.py` | Check `models/production/` |

### Medium-Term (1-3 Months)

| Goal | Current | Target | Check |
|------|---------|--------|-------|
| Trajectories | - | 10,000 | `training_state.json` |
| Sharpe | - | >1.5 | `performance_benchmarks.md` |
| Win Rate | - | >55% | `performance_benchmarks.md` |

---

## 📦 Environment Setup

### Initial Setup

```bash
# Clone repo
git clone <repo-url>
cd stock-agent-system-final

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.template .env

# Edit .env with your API keys
nano .env
```

### API Keys Required

```bash
# .env file
OPENAI_API_KEY=sk-...          # For data generation
ANTHROPIC_API_KEY=sk-ant-...   # For evaluation
FINNHUB_API_KEY=...            # For news
NEWSAPI_KEY=...                # For news
WANDB_API_KEY=...              # For tracking (optional)
```

---

## 🔧 Configuration

### Quick Config Edits

**SFT Training:**
```yaml
# config/sft/news_agent.yaml
training:
  max_steps: 500      # Adjust training length
  batch_size: 4       # Adjust for memory
  learning_rate: 2e-4 # Adjust learning speed
```

**RL Training:**
```yaml
# config/rl/grpo_config.yaml
training:
  num_iterations: 100  # Adjust training length
  batch_size: 16       # Adjust for memory
  kl_coef: 0.1        # Adjust exploration
```

---

## 📈 Monitoring

### W&B Dashboard

```bash
# Login once
wandb login

# View in browser
wandb ui
# or visit: https://wandb.ai/your-project
```

### Performance Monitor

```bash
# Real-time dashboard
python scripts/monitor_performance.py

# Custom refresh rate
python scripts/monitor_performance.py --refresh 10
```

---

## 🆘 Get Help

### Documentation

- Training Guide: `TRAINING_GUIDE.md`
- Medium-Term Plan: `MEDIUM_TERM_PLAN.md`
- Implementation Status: `IMPLEMENTATION_STATUS.md`
- API Docs: Start server and visit `/docs`

### Check System Status

```python
# Quick health check
from orchestration.coordinator import SystemCoordinator
coordinator = SystemCoordinator()

# Check loaded agents
print(coordinator.agents.keys())

# Test analysis
result = coordinator.analyze_symbol('AAPL')
print(result)
```

### Common Commands

```bash
# Check Python environment
python --version
pip list | grep -E "torch|transformers|anthropic"

# Check GPU
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"

# Check disk space
df -h

# Check process
ps aux | grep python
```

---

## 💾 Backup & Recovery

### Backup Models

```bash
# Automatic backup before deployment
python scripts/deploy_models.py  # Auto-creates backup

# Manual backup
cp -r models/production models/backups/manual_$(date +%Y%m%d)
```

### Rollback

```python
from scripts.deploy_models import ModelDeployer
deployer = ModelDeployer()
deployer.rollback('models/backups/backup_YYYYMMDD_HHMMSS')
```

---

## 🎓 Example Workflows

### Full Training Pipeline

```bash
# 1. Generate data
python scripts/generate_synthetic_data.py \
  --agent-type news --num-examples 100 \
  --output data/training/news.jsonl

# 2. Train SFT
python training/sft/train_news_agent.py

# 3. Collect trajectories
python scripts/run_rl_training.py \
  --num-trajectories 500

# 4. Evaluate
python scripts/evaluate_models.py \
  --models trained:models/news/final

# 5. Deploy
python scripts/deploy_models.py \
  --models news:models/news/final
```

### Continuous Improvement Loop

```bash
# Run in background with nohup
nohup python scripts/continuous_training.py \
  --target-trajectories 10000 \
  --target-sharpe 1.5 \
  > continuous_training.log 2>&1 &

# Monitor progress
tail -f continuous_training.log

# Check status
python scripts/monitor_performance.py
```

---

## 📊 Performance Targets

### Short-Term (1-2 Weeks)
- ✅ Models trained
- ✅ Sharpe > 0.5
- ✅ Win Rate > 50%

### Medium-Term (1-3 Months)
- ⏳ 10,000+ trajectories
- ⏳ Sharpe > 1.5
- ⏳ Win Rate > 55%

### Long-Term (3-6 Months)
- ⏳ Sharpe > 2.0
- ⏳ Win Rate > 60%
- ⏳ Production deployment

---

**Last Updated:** 2026-01-04

**Quick Links:**
- [Training Guide](TRAINING_GUIDE.md)
- [Medium-Term Plan](MEDIUM_TERM_PLAN.md)
- [Implementation Status](IMPLEMENTATION_STATUS.md)
