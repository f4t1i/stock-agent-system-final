# Makefile for Stock Agent Trading System
# One-click commands for backtest, test, report, and more

.PHONY: help install test backtest report clean lint format docker-up docker-down

# Default target
.DEFAULT_GOAL := help

##@ General

help: ## Display this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Installation

install: ## Install dependencies
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Installation complete"

install-dev: ## Install development dependencies
	@echo "📦 Installing development dependencies..."
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy
	@echo "✅ Development installation complete"

##@ Testing

test: ## Run all tests
	@echo "🧪 Running tests..."
	python -m pytest tests/ -v --tb=short
	@echo "✅ Tests complete"

test-cov: ## Run tests with coverage
	@echo "🧪 Running tests with coverage..."
	python -m pytest tests/ -v --cov=. --cov-report=html --cov-report=term
	@echo "📊 Coverage report: htmlcov/index.html"

test-contracts: ## Test signal contracts
	@echo "📋 Testing signal contracts..."
	python -m pytest tests/test_signal_validator.py -v
	python contracts/signal_validator.py
	@echo "✅ Contract tests complete"

test-backtest: ## Test backtest harness
	@echo "🎯 Testing backtest harness..."
	python -m pytest tests/test_backtester.py -v
	@echo "✅ Backtest tests complete"

##@ Backtesting

backtest: ## Run backtest (default config)
	@echo "🚀 Running backtest..."
	python scripts/run_backtest.py
	@echo "✅ Backtest complete - Check backtest_results/"

backtest-quick: ## Run quick backtest (3 months, 3 symbols)
	@echo "⚡ Running quick backtest..."
	python scripts/run_backtest.py --symbols AAPL,MSFT,GOOGL --start 2023-10-01 --end 2023-12-31
	@echo "✅ Quick backtest complete"

backtest-full: ## Run full backtest (1 year, 10 symbols)
	@echo "🔥 Running full backtest..."
	python scripts/run_backtest.py --symbols AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA,JPM,V,JNJ --start 2023-01-01 --end 2023-12-31
	@echo "✅ Full backtest complete"

backtest-validate: ## Run backtest with signal validation enabled
	@echo "✅ Running backtest with signal validation..."
	python scripts/run_backtest.py --validate-signals
	@echo "✅ Validated backtest complete"

##@ Reporting

report: ## Generate backtest report (latest results)
	@echo "📊 Generating backtest report..."
	python scripts/generate_report.py
	@echo "✅ Report generated - Check backtest_results/"

report-html: ## Generate HTML report
	@echo "🌐 Generating HTML report..."
	python scripts/generate_report.py --format html
	@echo "✅ HTML report generated"

report-pdf: ## Generate PDF report
	@echo "📄 Generating PDF report..."
	python scripts/generate_report.py --format pdf
	@echo "✅ PDF report generated"

##@ Training

##@ SFT Training

train-news-agent: ## Train News Agent (SFT)
	@echo "📰 Training News Agent..."
	python scripts/train_agent_sft.py \
		--agent news_agent \
		--dataset data/datasets/sft_v1 \
		--output models/sft/news_agent_v1.0.0
	@echo "✅ News Agent training complete"

train-technical-agent: ## Train Technical Agent (SFT)
	@echo "📊 Training Technical Agent..."
	python scripts/train_agent_sft.py \
		--agent technical_agent \
		--dataset data/datasets/sft_v1 \
		--output models/sft/technical_agent_v1.0.0
	@echo "✅ Technical Agent training complete"

train-fundamental-agent: ## Train Fundamental Agent (SFT)
	@echo "💰 Training Fundamental Agent..."
	python scripts/train_agent_sft.py \
		--agent fundamental_agent \
		--dataset data/datasets/sft_v1 \
		--output models/sft/fundamental_agent_v1.0.0
	@echo "✅ Fundamental Agent training complete"

train-all-agents: ## Train all three agents (News, Technical, Fundamental)
	@echo "🚀 Training all agents..."
	python scripts/train_agent_sft.py \
		--agent all \
		--dataset data/datasets/sft_v1 \
		--output-dir models/sft
	@echo "✅ All agents training complete"

train-quick-test: ## Quick test training (1 epoch, 100 samples)
	@echo "⚡ Quick test training..."
	python scripts/train_agent_sft.py \
		--agent news_agent \
		--dataset data/datasets/sft_v1 \
		--output models/sft/news_agent_test \
		--preset quick_test
	@echo "✅ Quick test complete"

train-production: ## Production training with optimized settings
	@echo "🏭 Production training..."
	python scripts/train_agent_sft.py \
		--agent all \
		--dataset data/datasets/sft_v1 \
		--output-dir models/sft \
		--preset production
	@echo "✅ Production training complete"

##@ Model Registry

models-list: ## List all registered models
	@echo "📋 Listing registered models..."
	python training/sft/model_registry.py --list

models-best: ## Show best model for agent (usage: make models-best AGENT=news_agent)
	@echo "🏆 Best model for $(or $(AGENT),news_agent)..."
	python training/sft/model_registry.py --best --agent $(or $(AGENT),news_agent) --metric eval_loss

models-promote: ## Promote model to production (usage: make models-promote MODEL_ID=xxx)
	@echo "⬆️  Promoting model $(MODEL_ID) to production..."
	python training/sft/model_registry.py --promote $(MODEL_ID) --to-stage production
	@echo "✅ Model promoted"

##@ Eval Gates & Regression Guards

eval-model: ## Evaluate model on holdout dataset (usage: make eval-model MODEL=xxx DATASET=xxx)
	@echo "📊 Evaluating model on holdout dataset..."
	python training/sft/eval_gates.py \
		--model $(MODEL) \
		--dataset $(DATASET)
	@echo "✅ Evaluation complete"

eval-with-drift: ## Evaluate with drift detection (usage: make eval-with-drift MODEL=xxx DATASET=xxx BASELINE='{"eval_loss":0.45}')
	@echo "📊 Evaluating with drift detection..."
	python training/sft/eval_gates.py \
		--model $(MODEL) \
		--dataset $(DATASET) \
		--baseline-metrics '$(BASELINE)' \
		--drift-threshold 5.0
	@echo "✅ Evaluation complete"

eval-history: ## View evaluation history
	@echo "📜 Evaluation history:"
	python training/sft/eval_gates.py --history --limit 20

regression-test: ## Run regression test (usage: make regression-test BASELINE=xxx CANDIDATE=xxx)
	@echo "🔍 Running regression test..."
	python training/sft/regression_guards.py \
		--baseline $(BASELINE) \
		--candidate $(CANDIDATE) \
		--metrics eval_loss eval_accuracy eval_f1
	@echo "✅ Regression test complete"

regression-test-holdout: ## Regression test with holdout (usage: make regression-test-holdout BASELINE_PATH=xxx CANDIDATE_PATH=xxx HOLDOUT=xxx)
	@echo "🔍 Running regression test with holdout re-evaluation..."
	python training/sft/regression_guards.py \
		--baseline-path $(BASELINE_PATH) \
		--candidate-path $(CANDIDATE_PATH) \
		--holdout $(HOLDOUT) \
		--metrics eval_loss eval_accuracy eval_f1
	@echo "✅ Regression test complete"

regression-history: ## View regression test history
	@echo "📜 Regression test history:"
	python training/sft/regression_guards.py --history --limit 20

regression-override: ## Override blocked model (usage: make regression-override TEST_ID=xxx REASON="explanation")
	@echo "⚠️  Applying override to test $(TEST_ID)..."
	python training/sft/regression_guards.py \
		--override $(TEST_ID) \
		--reason "$(REASON)"
	@echo "✅ Override applied"

##@ RL Training

train-rl: ## Train RL model with GRPO (usage: make train-rl POLICY=xxx EXPERIENCES=xxx OUTPUT=xxx)
	@echo "🎮 Training RL model with GRPO..."
	python scripts/train_rl.py \
		--policy $(POLICY) \
		--experience-store $(EXPERIENCES) \
		--output $(OUTPUT) \
		--iterations 100
	@echo "✅ RL training complete"

train-rl-quick: ## Quick RL test (10 iterations)
	@echo "⚡ Quick RL test training..."
	python scripts/train_rl.py \
		--policy models/sft/strategist_v1.0.0 \
		--experience-store data/experiences \
		--output models/rl/strategist_test \
		--preset quick_test
	@echo "✅ Quick RL test complete"

supervisor-stats: ## Show supervisor agent routing statistics
	@echo "📊 Supervisor routing statistics:"
	python agents/supervisor_v2.py --stats

supervisor-demo: ## Demo supervisor agent selection
	@echo "🎯 Supervisor demo:"
	python agents/supervisor_v2.py

regime-features-demo: ## Demo regime feature extraction
	@echo "🌍 Regime features demo:"
	python agents/regime_features.py --demo

##@ Data Synthesis

synthesize-sft: ## Synthesize SFT dataset (judge-approved, chat format)
	@echo "🔄 Synthesizing SFT dataset..."
	python scripts/synthesize_dataset.py --preset sft_v1
	@echo "✅ SFT dataset synthesized"

synthesize-preference: ## Synthesize preference learning dataset (contrastive pairs)
	@echo "🔄 Synthesizing preference learning dataset..."
	python scripts/synthesize_dataset.py --preset preference_v1
	@echo "✅ Preference dataset synthesized"

synthesize-rl: ## Synthesize RL training dataset (full spectrum)
	@echo "🔄 Synthesizing RL dataset..."
	python scripts/synthesize_dataset.py --preset rl_v1
	@echo "✅ RL dataset synthesized"

synthesize-eval: ## Synthesize evaluation benchmark (gold standard)
	@echo "🔄 Synthesizing evaluation benchmark..."
	python scripts/synthesize_dataset.py --preset eval_benchmark
	@echo "✅ Evaluation benchmark synthesized"

synthesize-custom: ## Synthesize custom dataset (specify --strategy, --format, etc.)
	@echo "🔄 Synthesizing custom dataset..."
	@echo "Usage: make synthesize-custom STRATEGY=judge_approved FORMAT=chat MIN_REWARD=0.5"
	python scripts/synthesize_dataset.py \
		--strategy $(or $(STRATEGY),judge_approved) \
		--format $(or $(FORMAT),chat) \
		--min-reward $(or $(MIN_REWARD),0.0)
	@echo "✅ Custom dataset synthesized"

judge-filter: ## Apply judge filtering to experience store
	@echo "⚖️  Applying judge filter to experiences..."
	python training/data_synthesis/judge_filter.py --storage-dir data/experiences --min-score 6.0
	@echo "✅ Judge filtering complete"

experience-stats: ## Show experience store statistics
	@echo "📊 Experience Store Statistics:"
	python training/data_synthesis/experience_store.py --storage-dir data/experiences --stats

experience-query: ## Query experiences (usage: make experience-query SYMBOL=AAPL MIN_REWARD=0.5)
	@echo "🔍 Querying experiences..."
	python training/data_synthesis/experience_store.py \
		--storage-dir data/experiences \
		--query \
		$(if $(SYMBOL),--symbol $(SYMBOL),) \
		$(if $(MIN_REWARD),--min-reward $(MIN_REWARD),) \
		$(if $(JUDGE_ONLY),--judge-approved-only,)
	@echo "✅ Query complete"

##@ Code Quality

lint: ## Run linters
	@echo "🔍 Running linters..."
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	@echo "✅ Linting complete"

format: ## Format code with black
	@echo "🎨 Formatting code..."
	black . --line-length 100
	@echo "✅ Code formatted"

type-check: ## Run type checker
	@echo "🔎 Running type checker..."
	mypy . --ignore-missing-imports
	@echo "✅ Type checking complete"

##@ Docker

docker-up: ## Start all services with Docker Compose
	@echo "🐳 Starting Docker services..."
	docker compose up -d
	@echo "✅ Services started"
	@echo "  API: http://localhost:8000"
	@echo "  Grafana: http://localhost:3000"
	@echo "  Prometheus: http://localhost:9090"

docker-down: ## Stop all Docker services
	@echo "🛑 Stopping Docker services..."
	docker compose down
	@echo "✅ Services stopped"

docker-logs: ## View Docker logs
	docker compose logs -f

docker-build: ## Build Docker images
	@echo "🔨 Building Docker images..."
	docker compose build
	@echo "✅ Build complete"

docker-backtest: ## Run backtest in Docker
	@echo "🐳 Running backtest in Docker..."
	docker compose run --rm api python scripts/run_backtest.py
	@echo "✅ Docker backtest complete"

##@ API

api-start: ## Start FastAPI server (development)
	@echo "🚀 Starting API server..."
	uvicorn api.server:app --reload --host 0.0.0.0 --port 8000

api-test: ## Test API endpoints
	@echo "🧪 Testing API endpoints..."
	python scripts/test_api.py
	@echo "✅ API tests complete"

##@ Database

db-init: ## Initialize PostgreSQL database
	@echo "🗄️  Initializing database..."
	python scripts/init_database.py
	@echo "✅ Database initialized"

db-migrate: ## Run database migrations
	@echo "🔄 Running database migrations..."
	python scripts/migrate_database.py
	@echo "✅ Migrations complete"

db-backup: ## Backup database
	@echo "💾 Backing up database..."
	python scripts/backup_database.py
	@echo "✅ Backup complete"

##@ Validation

validate-signals: ## Validate all signal examples
	@echo "✅ Validating signal examples..."
	python -c "from contracts.signal_validator import validate_signal_file; \
		print('Testing valid signal...'); \
		is_valid, errors = validate_signal_file('contracts/examples/valid_buy_signal.json'); \
		print(f'Valid: {is_valid}'); \
		print('\nTesting invalid signal...'); \
		is_valid, errors = validate_signal_file('contracts/examples/invalid_signal.json', strict=False); \
		print(f'Valid: {is_valid}, Errors: {len(errors)}')"
	@echo "✅ Signal validation complete"

validate-config: ## Validate system configuration
	@echo "⚙️  Validating configuration..."
	python scripts/validate_config.py
	@echo "✅ Configuration valid"

##@ Cleanup

clean: ## Clean temporary files and caches
	@echo "🧹 Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage 2>/dev/null || true
	@echo "✅ Cleanup complete"

clean-data: ## Clean generated data (backtest results, logs)
	@echo "🗑️  Cleaning generated data..."
	rm -rf backtest_results/*.json logs/*.log 2>/dev/null || true
	@echo "⚠️  Data cleaned"

clean-all: clean clean-data ## Clean everything

##@ Development

dev-setup: install-dev validate-config ## Setup development environment
	@echo "🎉 Development environment ready!"

dev-check: lint type-check test ## Run all development checks
	@echo "✅ All checks passed!"

ci: lint type-check test backtest-quick ## Run CI pipeline
	@echo "✅ CI pipeline complete"

##@ Production

prod-deploy: ## Deploy to production (with checks)
	@echo "🚀 Deploying to production..."
	@make lint
	@make type-check
	@make test
	@make docker-build
	@make docker-up
	@echo "✅ Production deployment complete"

prod-rollback: ## Rollback production deployment
	@echo "⏪ Rolling back production..."
	docker compose down
	git checkout main
	@make docker-build
	@make docker-up
	@echo "✅ Rollback complete"

##@ Monitoring

logs: ## View application logs
	@echo "📜 Viewing logs..."
	tail -f logs/app.log

logs-error: ## View error logs only
	@echo "❌ Viewing error logs..."
	grep -i error logs/app.log | tail -100

monitor: ## Start monitoring dashboard
	@echo "📊 Opening monitoring dashboard..."
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"

##@ Benchmarks

benchmark: ## Run performance benchmarks
	@echo "⚡ Running benchmarks..."
	python scripts/benchmark.py
	@echo "✅ Benchmarks complete"

benchmark-backtest: ## Benchmark backtest performance
	@echo "⚡ Benchmarking backtest..."
	time make backtest-quick
	@echo "✅ Backtest benchmark complete"

##@ Acceptance Tests (Phase A0)

acceptance-test: ## Run Phase A0 acceptance tests
	@echo "✅ Running Phase A0 acceptance tests..."
	@echo "\n1. Testing Signal Contract..."
	@make validate-signals
	@echo "\n2. Testing Backtest Determinism..."
	python tests/acceptance/test_deterministic_backtest.py
	@echo "\n3. Testing Survivorship Bias Guard..."
	python tests/acceptance/test_survivorship_bias.py
	@echo "\n4. Testing Corporate Actions..."
	python tests/acceptance/test_corporate_actions.py
	@echo "\n✅ All acceptance tests passed!"

acceptance-test-quick: validate-signals ## Quick acceptance test (contracts only)
	@echo "✅ Quick acceptance test complete"

acceptance-test-sft: ## Run SFT training pipeline acceptance tests (Tasks #17-20)
	@echo "✅ Running SFT training pipeline acceptance tests..."
	@echo "\n1. Testing SFT Training..."
	python tests/acceptance/test_sft_training.py
	@echo "\n2. Testing Eval Gates & Regression Guards..."
	python tests/acceptance/test_sft_pipeline_complete.py
	@echo "\n✅ All SFT tests passed!"

acceptance-test-rl: ## Run RL training pipeline acceptance tests (Tasks #21-24)
	@echo "✅ Running RL training pipeline acceptance tests..."
	python tests/acceptance/test_rl_training.py
	@echo "\n✅ All RL tests passed!"

acceptance-test-iteration: ## Run multi-iteration training acceptance tests (Phase A2)
	@echo "✅ Running multi-iteration training acceptance tests..."
	python tests/acceptance/test_multi_iteration_training.py
	@echo "\n✅ All iteration tests passed!"

##@ Multi-Iteration Training (Phase A2)

train-iteration: ## Run multi-iteration training (default: 10 iterations)
	@echo "🔄 Starting multi-iteration training..."
	python scripts/train_rl.py --mode iteration --iterations 10 --output models/iteration
	@echo "✅ Multi-iteration training complete"

train-iteration-quick: ## Quick iteration test (3 iterations)
	@echo "⚡ Quick iteration training..."
	python scripts/train_rl.py --mode iteration --iterations 3 --output models/iteration_test
	@echo "✅ Quick iteration complete"

train-regime-specific: ## Train regime-specific models (bull/bear/sideways)
	@echo "📊 Training regime-specific models..."
	python scripts/train_rl.py --mode regime --regimes bull,bear,sideways --output models/regime
	@echo "✅ Regime-specific training complete"

train-with-convergence: ## Train with convergence tracking
	@echo "📈 Training with convergence tracking..."
	python scripts/train_rl.py --mode iteration --iterations 20 --convergence-threshold 0.01 --patience 3 --output models/converged
	@echo "✅ Convergence training complete"

test-supervisor-v2: ## Test Supervisor v2 routing
	@echo "🎯 Testing Supervisor v2..."
	python -c "from agents.supervisor_v2 import SupervisorV2, SupervisorConfig; import numpy as np; config = SupervisorConfig(num_agents=4, context_dim=10, hidden_dim=64); supervisor = SupervisorV2(config); context = np.random.randn(10); agent_id, confidence = supervisor.select_agent(context); print(f'Selected agent: {agent_id}, Confidence: {confidence:.3f}')"
	@echo "✅ Supervisor v2 test complete"

test-regime-features: ## Test regime feature extraction
	@echo "📊 Testing regime features..."
	python -c "from agents.regime_features import RegimeFeatureExtractor; import pandas as pd; import numpy as np; from datetime import datetime; extractor = RegimeFeatureExtractor(); dates = pd.date_range(end=datetime.now(), periods=100, freq='D'); prices = pd.DataFrame({'date': dates, 'open': 100 + np.cumsum(np.random.randn(100) * 0.5), 'high': 100 + np.cumsum(np.random.randn(100) * 0.5 + 0.1), 'low': 100 + np.cumsum(np.random.randn(100) * 0.5 - 0.1), 'close': 100 + np.cumsum(np.random.randn(100) * 0.5), 'volume': np.random.randint(1000000, 10000000, 100)}); features = extractor.extract('AAPL', prices); print(f'Regime: {features[\"regime\"]}, Trend: {features[\"trend\"]}, Volatility: {features[\"volatility\"]}')"
	@echo "✅ Regime features test complete"

iteration-status: ## Show iteration training status
	@echo "📊 Iteration Training Status:"
	@echo "Models:"
	@ls -lh models/iteration/*.pt 2>/dev/null || echo "  No iteration models found"
	@echo "\nLogs:"
	@ls -lh models/iteration/logs/*.log 2>/dev/null || echo "  No logs found"
	@echo "\nMetrics:"
	@ls -lh models/iteration/metrics/*.json 2>/dev/null || echo "  No metrics found"

##@ Acceptance Tests (All)

acceptance-test-all: acceptance-test-phase-a0 acceptance-test-sft acceptance-test-rl acceptance-test-iteration ## Run all acceptance tests
	@echo "\n🎉 ALL ACCEPTANCE TESTS PASSED!"

##@ Documentation

docs: ## Generate documentation
	@echo "📚 Generating documentation..."
	python scripts/generate_docs.py
	@echo "✅ Documentation generated"

docs-serve: ## Serve documentation locally
	@echo "🌐 Serving documentation at http://localhost:8080"
	python -m http.server 8080 -d docs/

##@ Version

version: ## Show current version
	@echo "📌 Stock Agent Trading System"
	@python -c "import json; print('Version:', json.load(open('package.json'))['version'])" 2>/dev/null || echo "Version: 1.0.0"

##@ Quick Commands

all: install test backtest report ## Run complete workflow
	@echo "🎉 Complete workflow finished!"

quick: backtest-quick report ## Quick workflow (3 months)
	@echo "⚡ Quick workflow complete!"

full: backtest-full report ## Full workflow (1 year)
	@echo "🔥 Full workflow complete!"
