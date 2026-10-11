# Project Status & Honest Assessment

_Last reviewed: 2026-10-11 (commit `6de97ed`)_

This document describes what the Stock Agent System **actually** does today,
what does not work yet, and how to approach the repository if you want to use
or extend it. It supersedes the older `*_SUMMARY.md` / `*_STATUS*.md` files in
the repository root where they disagree.

**TL;DR:** A well-designed **research prototype** with a strong architecture and
broad scope. It is **not** production-ready, it does **not** run end-to-end out
of the box, and there is **no evidence yet** that it produces profitable trading
decisions. Do not trade real money with it.

---

## 1. Scorecard

| Area | Rating | Notes |
|------|--------|-------|
| Concept / architecture | 8/10 | Junior agents → Senior strategist, bandit supervisor, LLM judge feeding SFT/RL loops |
| Scope / ambition | 9/10 | Training, risk, calibration, explainability, alerts, dashboard |
| End-to-end runnability | 2/10 | Agents need fine-tuned local models that are not in the repo |
| Scientific validity (backtest / eval) | 2/10 | Main backtester has look-ahead bias, no baseline comparison |
| Production readiness | 1–2/10 | Placeholders in the core path, no CI, no broker integration |

---

## 2. What is solid

- **Architecture.** Specialised junior agents (`agents/junior/`: news, technical,
  fundamental), a senior strategist (`agents/senior/`), a supervisor with
  contextual-bandit routing (`agents/supervisor/enhanced_neural_ucb.py`) and an
  LLM judge (`judge/`, `training/judge/`) providing rewards for SFT, GRPO, PPO
  and DPO training (`training/`).
- **Self-contained modules that can be used on their own:**
  - `risk_management/` – risk gates, policies, position validation
  - `calibration/confidence_calibrator.py` – isotonic-regression calibration
  - `training/rl/backtester_v2.py` – backtester with commission, slippage,
    corporate actions and survivorship-bias handling
  - `utils/market_regime_detector.py`, `ml/` – feature engineering, anomaly
    detection, ensembles
- **Modular layout** that makes it a good base for experimenting with new ideas.

---

## 3. Known issues (verified in code)

### 3.1 The system does not start end-to-end
Every agent loads a locally fine-tuned model via
`AutoModelForCausalLM.from_pretrained(model_path)`. `config/system.yaml` points
to `models/news_agent_v1`, `models/technical_agent_v1`,
`models/fundamental_agent_v1`, `models/strategist_v1` – **none of these exist
in the repository**, and no trained checkpoints are published. As a result:

- `python main.py --symbol AAPL` fails while initialising `SystemCoordinator`.
- `api/server.py` creates the coordinator in its startup hook, so the API server
  fails to start as well.

### 3.2 The main backtest is not valid (look-ahead bias)
`training/rl/backtester.py` calls `coordinator.analyze_symbol(symbol)` for each
historical day **without passing the date**. The agents therefore fetch
*today's* news and market data while "trading" in the past. Any Sharpe ratio,
return or win rate produced by `main.py --mode backtest` is meaningless.
`analyze_symbol` has no `as_of` parameter yet.

### 3.3 Placeholders in the core decision path
| Location | Issue |
|----------|-------|
| `orchestration/coordinator.py:314-315` | Market regime hard-coded to `'bull'`, volatility to `20.0` |
| `orchestration/coordinator.py:399` | Trade execution is a TODO (no broker API) |
| `risk_management/risk_engine.py:279` | Volatility falls back to a mock value `0.2` |
| `agents/supervisor_v2.py:522` | Reward is random noise (`np.random.randn() * 0.5`) |

### 3.4 DeepMaster / Epistemic Shield / Memory are not integrated
`deepmaster_supervisor.py`, `deepmaster_orchestration.py`, `epistemic_shield.py`,
`meta_learning_cycle.py` and `memory_knowledge/` (commits "BLOCK 1–3") live
alongside the main system but are **not imported by it**. When their
dependencies are missing they fall back to mock implementations.

### 3.5 Web dashboard is not wired to the Python API
`web-dashboard/` is a separate Node/tRPC application (with OAuth, LLM and
storage helpers from a template). No calls to the FastAPI backend on port 8000
were found.

### 3.6 Test suite
Running `pytest tests test_deepmaster_orchestration.py` with lightweight
dependencies only (no `torch`, `transformers`, `pandas_ta`, no internet):

```
118 passed, 10 failed, 56 collection errors
```

- Collection errors: missing heavy dependencies (`torch`, `pandas_ta`, …).
- Failures: network access for `yfinance`, missing `pytest-asyncio` for
  WebSocket tests, and the deterministic-backtest `make` target.
- Passing: error taxonomy, judge & reports, Phase A3 acceptance, corporate
  actions, survivorship bias, SFT/RL pipeline acceptance, DeepMaster
  orchestration.

There is no CI (`.github/workflows` is missing), so "100% passing" was never
verified automatically.

### 3.7 Documentation drift
- Claims such as "production-ready", "enterprise-grade", "100% passing" and
  "5 AI agents" are not accurate (there is no standalone Sentiment agent;
  sentiment is part of the news agent).
- ~20 status/summary markdown files in the root contradict each other.
- Performance numbers in the README are not backed by reproducible benchmarks.

### 3.8 No proof of edge
There are no trained models, no evaluation reports and no clean
out-of-sample backtest. Whether the agents add any alpha over simple
baselines is **completely open**.

---

## 4. How to approach this repository

### If you want to explore the code
1. Read `docs/ARCHITECTURE.md`, then `orchestration/coordinator.py` – it shows
   how agents are wired together.
2. Look at one agent (e.g. `agents/junior/news_agent.py`) to understand the
   prompt → JSON → parsed result pattern.
3. Run the tests that work without GPU / internet:
   ```bash
   pip install pytest loguru python-dotenv pyyaml scipy scikit-learn \
               yfinance anthropic fastapi httpx pydantic sqlalchemy
   pytest tests/unit/test_error_taxonomy.py \
          tests/acceptance/test_phase_a3_complete.py \
          tests/acceptance/test_corporate_actions.py \
          tests/acceptance/test_survivorship_bias.py \
          test_deepmaster_orchestration.py -q
   ```

### If you want to use individual components
The risk engine, calibration, backtester v2, regime detector and `ml/` modules
can be imported directly without the LLM agents. This is currently the most
useful way to get value out of the repository.

### If you want to make the full system work – recommended order
1. **Runnable end-to-end path.** Replace the local `from_pretrained` loading
   with an API-based LLM backend (e.g. Claude or OpenAI) behind a common
   interface, so the system runs without fine-tuned checkpoints. Fine-tuning
   can come later.
2. **Fix the backtest.** Add `as_of` to `analyze_symbol`, pass point-in-time
   prices and date-filtered news to all agents, and use `backtester_v2` (costs,
   slippage) instead of `backtester`.
3. **Measure against baselines.** Buy & hold, SMA crossover, on an
   out-of-sample period. Only then is it possible to say whether the agents
   help.
4. **Remove placeholders** listed in 3.3 (regime detection and volatility from
   real data, a real reward signal for `supervisor_v2`).
5. **Decide on DeepMaster:** integrate it into the coordinator or move it to
   `experimental/`.
6. **Housekeeping:** add CI (GitHub Actions) with tests that run offline,
   split requirements into `core` / `training` / `dashboard`, consolidate the
   root-level status files.

### Do not
- Do not trade real money based on this system's output.
- Do not trust the numbers from `main.py --mode backtest` (see 3.2).
- Do not assume `docker-compose up` gives you a working system – it hits the
  same missing-model problem.
