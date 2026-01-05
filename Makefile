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

train-sft: ## Train SFT models for junior agents
	@echo "🧠 Training SFT models..."
	python scripts/train_sft.py
	@echo "✅ SFT training complete"

train-rl: ## Train RL model for strategist
	@echo "🎮 Training RL model..."
	python scripts/train_rl.py
	@echo "✅ RL training complete"

synthesize-data: ## Synthesize training data from experience library
	@echo "🔄 Synthesizing training data..."
	python scripts/synthesize_training_data.py
	@echo "✅ Data synthesis complete"

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
