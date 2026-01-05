# Stock Agent System - Pending Tasks

**Phase A0 (Proof-of-Value): ✅ 100% COMPLETE** (12/12 tasks)

**Phase A1 Week 3-4: ✅ 100% COMPLETE** (4/4 tasks)

---

## Phase A1 Week 3-4: Closed Loop v1 (4 tasks) ✅

- [x] **Task #13: 🔄 Auto Data-Synthesis Pipeline** (post-backtest)
  - Experience Store with multi-format storage (JSON, JSONL, SQLite, Parquet)
  - Query and filter capabilities (symbol, reward, judge approval)
  - Statistics tracking and export functionality

- [x] **Task #14: 📚 Dataset Registry System** (versioning)
  - Dataset versioning with semantic version support (1.0.0)
  - Metadata tracking (dataset_id, source_experiences, approval_rate)
  - Dataset configuration preservation

- [x] **Task #15: 🎓 Judge-Approved Filtering** für SFT Datasets
  - Batch judge filtering with configurable thresholds
  - Integration with JudgeRunner for quality assessment
  - Statistics reporting (pass rate, rejection reasons)

- [x] **Task #16: ✅ Acceptance Tests** für Auto-Synthesis
  - 15/15 test cases passing
  - Tests for all 4 synthesis strategies
  - Tests for all 3 output formats
  - End-to-end pipeline validation

## Phase A1 Week 5-6: SFT Training (4 tasks)

- [x] **Task #17: 🧠 SFT Training Pipeline** für Junior Agents (LoRA/QLoRA)
  - SFT config with 5 base models and 3 presets
  - LoRA/QLoRA trainer with 4-bit quantization
  - Unified training script for all agents
  - Model registry with versioning and promotion
  - Makefile commands (9 new targets)
  - Acceptance tests (19 test cases passing)

- [x] **Task #18: 🚧 Eval Gates** (Holdout-Performance Checking)
  - Standalone evaluation system with configurable gates
  - Holdout dataset evaluation (785 lines)
  - Performance drift detection
  - Historical tracking database
  - CLI interface with report generation

- [x] **Task #19: 🔒 Regression Guards** (neue Modelle ≥ alte Modelle)
  - Comprehensive regression testing framework (847 lines)
  - Multi-metric comparison with configurable tolerances
  - Automated blocking of degraded models
  - Override capability for exceptional cases
  - Registry-based and holdout-based testing

- [x] **Task #20: ✅ Acceptance Tests** für SFT Training + Gates
  - Complete pipeline integration tests (721 lines)
  - 7/7 test cases passing
  - Tests for eval gates and regression guards
  - End-to-end workflow validation
  - Makefile target: acceptance-test-sft

## Phase A1 Week 7-8: RL Training (4 tasks)

- [x] **Task #21: 🎮 GRPO Trainer** für Senior Strategist
  - Group Relative Policy Optimization implementation (635 lines)
  - Group sampling with K responses per prompt
  - Relative advantage computation
  - PPO-style policy updates with KL penalty
  - Integration framework with experience store

- [x] **Task #22: 🧭 Supervisor v2** mit Contextual Bandit Routing
  - Multi-armed bandit agent selection (519 lines)
  - 3 algorithms: Thompson Sampling, UCB, Epsilon-Greedy
  - Per-agent performance tracking
  - Regime-aware routing
  - SQLite tracking database

- [x] **Task #23: 🌍 Regime-Features** für Routing (391 lines)
  - Volatility calculation (historical, realized)
  - Trend detection (SMA crossovers, strength)
  - Market regime classification (6 regimes)
  - Sentiment integration hooks

- [x] **Task #24: ✅ Acceptance Tests** für RL Training + Routing
  - 4/4 test cases passing
  - GRPO configuration tests
  - Supervisor bandit algorithm tests
  - Regime feature extraction tests
  - End-to-end integration tests
  - Makefile target: acceptance-test-rl

## Phase A2 Week 9-10: Learning Track (4 tasks)

- [ ] 🔁 Multi-Iteration Training Script (10 Iterations)
- [ ] 📈 Convergence Tracking System
- [ ] 🌦️ Regime-spezifische Modelle (Bull/Bear Strategists)
- [ ] ✅ Acceptance Tests für iteratives Training

## Phase A3 Week 11-12: Produktisierung (6 tasks)

- [ ] 🎨 UI Dashboard mit Explainability Cards
- [ ] ⚡ Alerts & Watchlists implementieren
- [ ] 🛡️ Trading Policies & Guardrails (Risk Gates)
- [ ] 🎯 Confidence Calibration System
- [ ] 📦 Release v1.0.0 mit Semver + Changelog
- [ ] ✅ Final Acceptance Tests für v1.0.0

---

## Progress Summary

**Total Completed: 24 tasks**
- Phase A0 (Proof-of-Value): 12/12 ✅
- Phase A1 Week 3-4 (Closed Loop v1): 4/4 ✅
- Phase A1 Week 5-6 (SFT Training): 4/4 ✅
- Phase A1 Week 7-8 (RL Training): 4/4 ✅

**Total Pending: 10 tasks**
- Phase A2 Week 9-10 (Learning Track): 4 tasks
- Phase A3 Week 11-12 (Produktisierung): 6 tasks

**Overall Progress: 71% (24/34 tasks)**

---

## Next Steps

**Ready to start:** Tasks #25-28 - Learning Track (Multi-Iteration Training)

**Task #25:** Multi-Iteration Training Script (10 Iterations)
- Automated training loop with feedback
- Model versioning across iterations
- Convergence monitoring

**Task #26:** Convergence Tracking System
- Performance metrics over iterations
- Early stopping criteria
- Visualization dashboards

**Task #27:** Regime-spezifische Modelle (Bull/Bear Strategists)
- Train separate models per market regime
- Regime detection and model switching
- Performance comparison

**Task #28:** Acceptance Tests für iteratives Training
- Multi-iteration workflow tests
- Convergence validation
- Regime-specific model tests

---

Last updated: 2026-01-05 (Tasks #21-24 complete - Phase A1 Week 7-8: 100%)
