# Training Guide - Stock Analysis Multi-Agent System

**Complete guide for training, evaluation, and deployment**

---

## 🎯 Short-Term Training Plan (1-2 Weeks)

### Week 1: SFT Training

**Goal:** Train all junior agents with 100+ synthetic trajectories

1. **Generate Synthetic Data** (Day 1)
2. **Train News Agent** (Day 2-3)
3. **Train Technical Agent** (Day 3-4)
4. **Train Fundamental Agent** (Day 4-5)
5. **Evaluate SFT Models** (Day 6-7)

### Week 2: RL Training

**Goal:** Train strategist with 500+ real trajectories

1. **Collect Trajectories** (Day 1-2)
2. **Train Strategist (GRPO/PPO)** (Day 3-5)
3. **Evaluate RL Models** (Day 6)
4. **Deploy Best Models** (Day 7)

---

## 📋 Prerequisites

### 1. Environment Setup

```bash
# Ensure all dependencies installed
pip install -r requirements.txt

# Install optional training dependencies
pip install unsloth wandb
```

### 2. API Keys

Set up required API keys in `.env`:

```bash
# For data generation
OPENAI_API_KEY=sk-...          # GPT-4o for synthetic data
ANTHROPIC_API_KEY=sk-ant-...   # Claude for evaluation

# For market data
FINNHUB_API_KEY=your_key       # News data
NEWSAPI_KEY=your_key           # News data
SERPER_API_KEY=your_key        # News data

# For tracking (optional)
WANDB_API_KEY=your_key         # Weights & Biases
```

### 3. Hardware Requirements

| Training Type | Minimum RAM | Recommended GPU | VRAM |
|--------------|-------------|-----------------|------|
| SFT (with Unsloth) | 16 GB | RTX 3090 | 24 GB |
| GRPO | 16 GB | RTX 3090 | 24 GB |
| PPO | 32 GB | A100 | 40 GB |

**Note:** Use GRPO for systems with <32GB RAM

---

## 🚀 Quick Start - Complete Pipeline

### Option 1: Run Everything at Once

```bash
# Complete 1-2 week training pipeline
python scripts/run_complete_training.py \
  --sft-examples 100 \
  --rl-trajectories 500 \
  --rl-iterations 100 \
  --algorithm grpo
```

This will:
1. Generate 100 synthetic examples per agent
2. Train all SFT models (news, technical, fundamental)
3. Collect 500 RL trajectories
4. Train strategist with GRPO
5. Evaluate all models
6. Deploy best models

**Estimated time:** 24-48 hours (depending on hardware)

---

## 📖 Step-by-Step Guide

### Step 1: SFT Training (Week 1)

#### 1.1 Generate Synthetic Training Data

```bash
# Run complete SFT pipeline (data + training)
python scripts/run_sft_training.py \
  --num-examples 100 \
  --max-steps 500 \
  --provider openai \
  --model gpt-4o
```

**Output:**
- `training_runs/sft_run_YYYYMMDD_HHMMSS/data/` - Training datasets
- `training_runs/sft_run_YYYYMMDD_HHMMSS/models/` - Trained models

**Expected duration:** 6-12 hours

**W&B tracking:** Monitor at https://wandb.ai/your-project

#### 1.2 Alternative - Generate Data Only

```bash
# Generate data for specific agent
python scripts/generate_synthetic_data.py \
  --agent-type news \
  --num-examples 100 \
  --output data/training/news_train.jsonl \
  --provider openai \
  --model gpt-4o

# Repeat for technical and fundamental
```

#### 1.3 Alternative - Train Specific Agent

```bash
# Train individual agent
python training/sft/train_news_agent.py \
  --config config/sft/news_agent.yaml

python training/sft/train_technical_agent.py \
  --config config/sft/technical_agent.yaml

python training/sft/train_fundamental_agent.py \
  --config config/sft/fundamental_agent.yaml
```

---

### Step 2: RL Training (Week 2)

#### 2.1 Run Complete RL Pipeline

```bash
# Complete RL pipeline (collection + training)
python scripts/run_rl_training.py \
  --num-trajectories 500 \
  --num-iterations 100 \
  --algorithm grpo \
  --symbols AAPL MSFT GOOGL AMZN TSLA
```

**Output:**
- `training_runs/rl_grpo_run_YYYYMMDD_HHMMSS/experience_library.db` - Trajectory database
- `training_runs/rl_grpo_run_YYYYMMDD_HHMMSS/models/` - Trained strategist

**Expected duration:** 12-24 hours

#### 2.2 Alternative - Collect Trajectories Only

```bash
# Collect using backtesting
python training/rl/backtester.py \
  --symbols AAPL MSFT GOOGL AMZN TSLA \
  --start-date 2023-01-01 \
  --end-date 2023-12-31
```

#### 2.3 Alternative - Train Strategist Only

```bash
# GRPO (recommended for <32GB RAM)
python training/rl/train_strategist_grpo.py \
  --config config/rl/grpo_config.yaml \
  --iterations 100

# PPO (for >32GB RAM)
python training/rl/train_strategist_ppo.py \
  --config config/rl/ppo_config.yaml \
  --iterations 100
```

---

### Step 3: Evaluation

#### 3.1 Evaluate All Models

```bash
python scripts/evaluate_models.py \
  --models \
    baseline:models/baseline \
    sft:training_runs/sft_run_YYYYMMDD_HHMMSS/models \
    rl:training_runs/rl_grpo_run_YYYYMMDD_HHMMSS/models \
  --test-symbols AAPL MSFT GOOGL AMZN TSLA \
  --test-days 90
```

**Output:**
- `evaluation_results/eval_YYYYMMDD_HHMMSS/evaluation_report.md` - Full report
- `evaluation_results/eval_YYYYMMDD_HHMMSS/comparison_plots.png` - Performance plots
- `evaluation_results/eval_YYYYMMDD_HHMMSS/evaluation_results.json` - JSON results

#### 3.2 Review Results

```bash
# View report
cat evaluation_results/eval_*/evaluation_report.md

# View plots
open evaluation_results/eval_*/comparison_plots.png
```

**Key Metrics to Check:**
- **Sharpe Ratio** > 0.5 (minimum for deployment)
- **Max Drawdown** < 20%
- **Win Rate** > 50%
- **Profit Factor** > 1.5

---

### Step 4: Deployment

#### 4.1 Deploy Best Models

```bash
python scripts/deploy_models.py \
  --models \
    news:training_runs/sft_run_YYYYMMDD_HHMMSS/models/news_agent/final \
    technical:training_runs/sft_run_YYYYMMDD_HHMMSS/models/technical_agent/final \
    fundamental:training_runs/sft_run_YYYYMMDD_HHMMSS/models/fundamental_agent/final \
    strategist:training_runs/rl_grpo_run_YYYYMMDD_HHMMSS/models/final \
  --min-sharpe 0.5
```

**Safety Features:**
- Automatic backup before deployment
- Model validation (Sharpe > 0.5)
- Deployment verification
- Rollback on failure

**Output:**
- `models/production/` - Deployed models
- `models/backups/backup_YYYYMMDD_HHMMSS/` - Backup
- `models/production/deployment_YYYYMMDD_HHMMSS.json` - Deployment record

#### 4.2 Verify Deployment

```bash
# Test deployed system
python -c "
from orchestration.coordinator import SystemCoordinator
coordinator = SystemCoordinator()
result = coordinator.analyze_symbol('AAPL')
print(f'Recommendation: {result[\"recommendation\"]}')
print(f'Confidence: {result[\"confidence\"]:.2f}')
"
```

#### 4.3 Rollback (if needed)

```bash
# Rollback to previous version
python -c "
from scripts.deploy_models import ModelDeployer
deployer = ModelDeployer()
deployer.rollback('models/backups/backup_YYYYMMDD_HHMMSS')
"
```

---

## 📊 Monitoring Training

### Weights & Biases (Recommended)

```bash
# Initialize W&B
wandb login

# View training
wandb ui
# or visit: https://wandb.ai/your-project
```

**Metrics to Monitor:**
- **Loss:** Should decrease steadily
- **Perplexity:** Should decrease
- **Reward (RL):** Should increase
- **Validation metrics:** Track overfitting

### TensorBoard (Alternative)

```bash
# Launch TensorBoard
tensorboard --logdir training_runs/

# View at http://localhost:6006
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Solutions:**
- Use GRPO instead of PPO (lower memory)
- Reduce batch size in config
- Enable gradient checkpointing
- Use 4-bit quantization (already default)

```yaml
# In config file
training:
  batch_size: 2  # Reduce from 4
  gradient_accumulation_steps: 8  # Increase to compensate
```

#### 2. Training Too Slow

**Solutions:**
- Reduce max_steps
- Use smaller model (7B instead of 13B)
- Enable mixed precision (fp16/bf16)
- Reduce num_examples

#### 3. Poor Performance

**Indicators:**
- Sharpe Ratio < 0.5
- Win Rate < 50%
- Max Drawdown > 20%

**Solutions:**
- Collect more diverse trajectories
- Increase training iterations
- Adjust reward function
- Check data quality

#### 4. Data Generation Fails

**Check:**
- API keys are set correctly
- Sufficient API credits
- Network connectivity
- Rate limits

---

## 📈 Advanced Configuration

### Custom Training Config

```yaml
# config/sft/custom_agent.yaml
model:
  base_model: "unsloth/llama-3.1-8b-bnb-4bit"
  max_seq_length: 2048
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.05

training:
  output_dir: "models/custom_agent"
  batch_size: 4
  gradient_accumulation_steps: 4
  max_steps: 1000
  learning_rate: 2e-4
  warmup_steps: 100
  save_steps: 100

data:
  dataset_path: "data/training/custom_train.jsonl"
```

### Custom RL Config

```yaml
# config/rl/custom_grpo.yaml
model:
  sft_checkpoint: "models/strategist/sft/final"
  lora_rank: 8
  lora_alpha: 16
  learning_rate: 1e-5

training:
  output_dir: "models/custom_strategist"
  num_iterations: 200
  batch_size: 16
  save_interval: 20
  kl_coef: 0.1
  vf_coef: 0.1
```

---

## 💾 Saving & Loading Models

### Save Training Checkpoint

```python
# Models auto-save during training
# Checkpoints: models/agent_name/checkpoint-XXXX
# Final: models/agent_name/final
```

### Load for Inference

```python
from agents.junior.news_agent import NewsAgent

agent = NewsAgent(
    model_path="models/production/news",
    config={'fp16': True}
)

result = agent.analyze('AAPL', lookback_days=7)
```

### Export for Serving

```bash
# Convert to GGUF (for llama.cpp)
python scripts/convert_to_gguf.py \
  --model-path models/production/news \
  --output-path models/export/news.gguf

# Convert to ONNX (for production serving)
python scripts/convert_to_onnx.py \
  --model-path models/production/news \
  --output-path models/export/news.onnx
```

---

## 🎓 Best Practices

### 1. Data Quality

✅ **Do:**
- Generate diverse examples
- Include edge cases
- Validate generated data
- Use multiple LLMs for diversity

❌ **Don't:**
- Use only bull market examples
- Ignore failed trajectories
- Skip data validation
- Use outdated data

### 2. Training

✅ **Do:**
- Monitor validation metrics
- Save checkpoints frequently
- Use early stopping
- Track experiments with W&B

❌ **Don't:**
- Train without validation
- Ignore overfitting signs
- Skip hyperparameter tuning
- Train on test data

### 3. Deployment

✅ **Do:**
- Always backup before deployment
- Validate model performance
- Test thoroughly
- Monitor in production

❌ **Don't:**
- Deploy without evaluation
- Skip validation checks
- Ignore performance degradation
- Deploy without rollback plan

---

## 📚 Additional Resources

- **API Documentation:** `/docs` endpoint when server running
- **Code Examples:** `examples/` directory
- **Configuration Templates:** `config/` directory
- **W&B Documentation:** https://docs.wandb.ai

---

## 🆘 Getting Help

**Issues:**
- GitHub Issues: https://github.com/your-repo/issues
- Discussions: https://github.com/your-repo/discussions

**Documentation:**
- System Architecture: `docs/ARCHITECTURE.md`
- Agent Details: `docs/AGENTS.md`
- API Reference: `docs/API.md`

---

## ✅ Training Checklist

### Week 1 - SFT
- [ ] Environment setup complete
- [ ] API keys configured
- [ ] Generated 100+ examples per agent
- [ ] Trained news agent
- [ ] Trained technical agent
- [ ] Trained fundamental agent
- [ ] Evaluated SFT models
- [ ] Sharpe Ratio > 0.5

### Week 2 - RL
- [ ] Collected 500+ trajectories
- [ ] Trained strategist (GRPO/PPO)
- [ ] Evaluated RL models
- [ ] Comparison with baseline
- [ ] Models deployed to production
- [ ] Deployment verified
- [ ] Monitoring enabled

---

**Status:** Ready for Production Training 🚀

**Last Updated:** 2026-01-04
