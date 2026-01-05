# Phase A3 Full-Stack Integration TODO

**Goal:** Integrate Python Backend (stock-agent-system-final) with React Frontend (web-dashboard)

**Total Tasks:** ~60 implementation tasks across 6 features

---

## 📋 TASK #29: UI Dashboard mit Explainability Cards

### Backend (Python) - 8 Tasks

#### 1. Explainability API Module (`api/explainability.py`)
- [ ] Create FastAPI router for explainability endpoints
- [ ] GET /api/explainability/agent/{agent_name}/decision/{decision_id}
- [ ] POST /api/explainability/analyze - Generate explanation for decision
- [ ] Response schema: ExplainabilityResponse (reasoning, confidence, factors, alternatives)

#### 2. Reasoning Extractor (`agents/reasoning_extractor.py`)
- [ ] Extract reasoning from agent outputs
- [ ] Parse confidence scores and contributing factors
- [ ] Identify key decision drivers
- [ ] Generate human-readable explanations

#### 3. Decision Logger (`agents/decision_logger.py`)
- [ ] Log all agent decisions with metadata
- [ ] Store reasoning chains
- [ ] Track confidence evolution
- [ ] SQLite database schema for decisions

#### 4. Explainability Config (`config/explainability.yaml`)
- [ ] Configure explanation verbosity levels
- [ ] Define factor importance thresholds
- [ ] Set confidence display ranges

### Frontend (React) - 7 Tasks

#### 5. ExplainabilityCard Component (`client/src/components/explainability/ExplainabilityCard.tsx`)
- [ ] Card layout with agent icon and name
- [ ] Recommendation display (Buy/Sell/Hold)
- [ ] Confidence meter with color coding
- [ ] Reasoning text with formatting
- [ ] Key factors list with importance weights
- [ ] Alternative scenarios section
- [ ] Expand/collapse functionality

#### 6. ReasoningVisualization Component (`client/src/components/explainability/ReasoningVisualization.tsx`)
- [ ] Factor importance bar chart
- [ ] Confidence timeline graph
- [ ] Decision tree visualization
- [ ] Interactive tooltips

#### 7. ConfidenceGauge Component (`client/src/components/explainability/ConfidenceGauge.tsx`)
- [ ] Circular gauge with percentage
- [ ] Color coding (red/yellow/green)
- [ ] Confidence level labels
- [ ] Animation on value change

#### 8. tRPC Procedures (`server/routers.ts`)
- [ ] explainability.getDecision(decisionId)
- [ ] explainability.analyze(symbol, agentName)
- [ ] explainability.listRecent(limit)

#### 9. Explainability Page (`client/src/pages/Explainability.tsx`)
- [ ] Page layout with filters
- [ ] Agent selector dropdown
- [ ] Symbol search
- [ ] Decision history timeline
- [ ] ExplainabilityCard grid

#### 10. Integration with Dashboard (`client/src/pages/Dashboard.tsx`)
- [ ] Add Explainability section
- [ ] Show recent decisions
- [ ] Link to full Explainability page

#### 11. Database Schema (`drizzle/schema.ts`)
- [ ] decisions table (id, symbol, agentName, recommendation, confidence, reasoning, factors, timestamp)
- [ ] decision_factors table (id, decisionId, factor, importance, value)

---

## 📋 TASK #30: Alerts & Watchlists

### Backend (Python) - 7 Tasks

#### 12. Alerts System (`api/alerts.py`)
- [ ] POST /api/alerts/create - Create new alert
- [ ] GET /api/alerts/list - List user alerts
- [ ] PUT /api/alerts/{id}/update - Update alert
- [ ] DELETE /api/alerts/{id} - Delete alert
- [ ] POST /api/alerts/{id}/trigger - Manually trigger alert
- [ ] Alert types: price_threshold, confidence_change, recommendation_change, technical_signal

#### 13. Alert Evaluator (`monitoring/alert_evaluator.py`)
- [ ] Evaluate price threshold alerts
- [ ] Check confidence changes
- [ ] Monitor recommendation changes
- [ ] Detect technical signals (RSI, MACD crossovers)
- [ ] Background task scheduler (every 5 minutes)

#### 14. Notification Dispatcher (`monitoring/notification_dispatcher.py`)
- [ ] Send email notifications
- [ ] Send push notifications (via Manus notification API)
- [ ] Send webhook notifications
- [ ] Notification templates
- [ ] Rate limiting and deduplication

#### 15. Watchlist Manager (`api/watchlist.py`)
- [ ] POST /api/watchlist/create - Create watchlist
- [ ] GET /api/watchlist/list - List watchlists
- [ ] POST /api/watchlist/{id}/add_symbol - Add symbol
- [ ] DELETE /api/watchlist/{id}/remove_symbol - Remove symbol
- [ ] GET /api/watchlist/{id}/status - Get watchlist status

#### 16. Watchlist Monitor (`monitoring/watchlist_monitor.py`)
- [ ] Monitor all symbols in watchlists
- [ ] Fetch latest prices and metrics
- [ ] Trigger alerts based on conditions
- [ ] Update watchlist status

#### 17. Alert Config (`config/alerts.yaml`)
- [ ] Default alert thresholds
- [ ] Notification channels configuration
- [ ] Alert evaluation frequency
- [ ] Rate limiting rules

#### 18. Database Schema
- [ ] alerts table (id, userId, symbol, alertType, condition, threshold, isActive, lastTriggered)
- [ ] watchlists table (id, userId, name, symbols, createdAt)
- [ ] alert_history table (id, alertId, triggeredAt, value, notificationSent)

### Frontend (React) - 8 Tasks

#### 19. AlertsPanel Component (`client/src/components/alerts/AlertsPanel.tsx`)
- [ ] Alert list with status indicators
- [ ] Create alert button
- [ ] Edit/Delete alert actions
- [ ] Alert history view
- [ ] Filter by status (active/triggered/disabled)

#### 20. CreateAlertDialog Component (`client/src/components/alerts/CreateAlertDialog.tsx`)
- [ ] Symbol input with autocomplete
- [ ] Alert type selector
- [ ] Condition builder (above/below/crosses)
- [ ] Threshold input
- [ ] Notification channel selection
- [ ] Form validation

#### 21. WatchlistPanel Component (`client/src/components/watchlist/WatchlistPanel.tsx`)
- [ ] Watchlist tabs
- [ ] Symbol list with current prices
- [ ] Add/Remove symbol buttons
- [ ] Quick analysis link per symbol
- [ ] Drag-and-drop reordering

#### 22. WatchlistCard Component (`client/src/components/watchlist/WatchlistCard.tsx`)
- [ ] Symbol name and ticker
- [ ] Current price with change percentage
- [ ] Mini chart (sparkline)
- [ ] Latest recommendation badge
- [ ] Quick action buttons

#### 23. NotificationToast Component (`client/src/components/alerts/NotificationToast.tsx`)
- [ ] Toast notification for triggered alerts
- [ ] Auto-dismiss after 5 seconds
- [ ] Click to view details
- [ ] Sound notification option

#### 24. tRPC Procedures
- [ ] alerts.create(alertData)
- [ ] alerts.list()
- [ ] alerts.update(id, data)
- [ ] alerts.delete(id)
- [ ] watchlist.create(name, symbols)
- [ ] watchlist.list()
- [ ] watchlist.addSymbol(id, symbol)
- [ ] watchlist.removeSymbol(id, symbol)

#### 25. Alerts Page (`client/src/pages/Alerts.tsx`)
- [ ] Page layout with tabs (Alerts / Watchlists)
- [ ] AlertsPanel integration
- [ ] WatchlistPanel integration
- [ ] Create buttons

#### 26. Real-time Updates (WebSocket)
- [ ] Subscribe to alert triggers
- [ ] Subscribe to watchlist updates
- [ ] Update UI on new notifications

---

## 📋 TASK #31: Trading Policies & Guardrails

### Backend (Python) - 6 Tasks

#### 27. Risk Gates System (`risk/risk_gates.py`)
- [ ] Position size limits (max % of portfolio)
- [ ] Daily loss limits (max drawdown per day)
- [ ] Concentration limits (max % in single symbol)
- [ ] Leverage limits
- [ ] Volatility gates (block trades in high volatility)
- [ ] Evaluate before trade execution

#### 28. Trading Policies Engine (`risk/trading_policies.py`)
- [ ] Define policy rules (YAML-based)
- [ ] Policy evaluation logic
- [ ] Policy violation detection
- [ ] Override mechanism with approval
- [ ] Policy audit log

#### 29. Position Validator (`risk/position_validator.py`)
- [ ] Validate proposed trades against policies
- [ ] Check portfolio constraints
- [ ] Calculate risk metrics
- [ ] Return validation result with reasons

#### 30. Risk API (`api/risk.py`)
- [ ] POST /api/risk/validate_trade - Validate trade
- [ ] GET /api/risk/policies - List active policies
- [ ] PUT /api/risk/policies/{id}/update - Update policy
- [ ] POST /api/risk/override - Request policy override
- [ ] GET /api/risk/violations - List policy violations

#### 31. Risk Config (`config/risk_policies.yaml`)
- [ ] Default risk limits
- [ ] Policy definitions
- [ ] Override approval workflow
- [ ] Violation severity levels

#### 32. Database Schema
- [ ] risk_policies table (id, name, rules, isActive, createdAt)
- [ ] policy_violations table (id, policyId, symbol, violationType, severity, timestamp)
- [ ] policy_overrides table (id, violationId, approvedBy, reason, timestamp)

### Frontend (React) - 6 Tasks

#### 33. RiskGuardrailsPanel Component (`client/src/components/risk/RiskGuardrailsPanel.tsx`)
- [ ] Active policies list
- [ ] Policy status indicators
- [ ] Enable/Disable toggles
- [ ] Edit policy button
- [ ] Violation history

#### 34. PolicyEditor Component (`client/src/components/risk/PolicyEditor.tsx`)
- [ ] Policy name input
- [ ] Rule builder interface
- [ ] Limit value inputs
- [ ] Severity level selector
- [ ] Save/Cancel buttons

#### 35. TradeValidationWidget Component (`client/src/components/risk/TradeValidationWidget.tsx`)
- [ ] Trade input form (symbol, quantity, side)
- [ ] Validate button
- [ ] Validation result display
- [ ] Policy violations list
- [ ] Override request button

#### 36. RiskMetricsCard Component (`client/src/components/risk/RiskMetricsCard.tsx`)
- [ ] Current portfolio risk metrics
- [ ] Position concentration chart
- [ ] Daily P&L vs limit
- [ ] Volatility gauge

#### 37. tRPC Procedures
- [ ] risk.validateTrade(tradeData)
- [ ] risk.listPolicies()
- [ ] risk.updatePolicy(id, rules)
- [ ] risk.requestOverride(violationId, reason)
- [ ] risk.getViolations()

#### 38. Risk Management Page (`client/src/pages/RiskManagement.tsx`)
- [ ] Page layout with sections
- [ ] RiskGuardrailsPanel
- [ ] RiskMetricsCard
- [ ] TradeValidationWidget
- [ ] Violation history table

---

## 📋 TASK #32: Confidence Calibration System

### Backend (Python) - 5 Tasks

#### 39. Calibration Engine (`calibration/calibration_engine.py`)
- [ ] Collect predicted confidences and actual outcomes
- [ ] Calculate calibration metrics (ECE, MCE, Brier score)
- [ ] Generate calibration curves
- [ ] Identify over/under-confident regions
- [ ] Suggest confidence adjustments

#### 40. Calibration Tracker (`calibration/calibration_tracker.py`)
- [ ] Track predictions with timestamps
- [ ] Store actual outcomes when available
- [ ] Update calibration statistics
- [ ] Generate calibration reports

#### 41. Uncertainty Quantification (`calibration/uncertainty.py`)
- [ ] Epistemic uncertainty (model uncertainty)
- [ ] Aleatoric uncertainty (data uncertainty)
- [ ] Combined uncertainty estimation
- [ ] Confidence interval calculation

#### 42. Calibration API (`api/calibration.py`)
- [ ] GET /api/calibration/metrics - Get calibration metrics
- [ ] GET /api/calibration/curve - Get calibration curve data
- [ ] POST /api/calibration/adjust - Adjust confidence scores
- [ ] GET /api/calibration/history - Calibration history

#### 43. Database Schema
- [ ] predictions table (id, agentName, symbol, predictedConf, actualOutcome, timestamp)
- [ ] calibration_metrics table (id, agentName, ece, mce, brierScore, sampleSize, timestamp)

### Frontend (React) - 5 Tasks

#### 44. CalibrationDashboard Component (`client/src/components/calibration/CalibrationDashboard.tsx`)
- [ ] Calibration metrics cards (ECE, MCE, Brier)
- [ ] Per-agent calibration comparison
- [ ] Time-series calibration evolution
- [ ] Confidence distribution histogram

#### 45. CalibrationCurve Component (`client/src/components/calibration/CalibrationCurve.tsx`)
- [ ] Reliability diagram (predicted vs actual)
- [ ] Perfect calibration line
- [ ] Confidence bins
- [ ] Interactive tooltips with sample counts

#### 46. UncertaintyWidget Component (`client/src/components/calibration/UncertaintyWidget.tsx`)
- [ ] Uncertainty breakdown (epistemic/aleatoric)
- [ ] Confidence interval display
- [ ] Uncertainty trend over time

#### 47. tRPC Procedures
- [ ] calibration.getMetrics(agentName)
- [ ] calibration.getCurve(agentName)
- [ ] calibration.getHistory(agentName, days)

#### 48. Calibration Page (`client/src/pages/Calibration.tsx`)
- [ ] Page layout
- [ ] CalibrationDashboard
- [ ] Agent selector
- [ ] Time range selector
- [ ] Export calibration report button

---

## 📋 TASK #33: Release v1.0.0

### Documentation & Release - 6 Tasks

#### 49. CHANGELOG.md
- [ ] Version 1.0.0 header
- [ ] Added features list (all Phase A0-A3 tasks)
- [ ] Breaking changes (if any)
- [ ] Migration guide
- [ ] Contributors

#### 50. VERSION File
- [ ] Create VERSION file with 1.0.0
- [ ] Semantic versioning format

#### 51. Release Documentation (`docs/RELEASE_v1.0.0.md`)
- [ ] Release highlights
- [ ] Feature overview
- [ ] Installation instructions
- [ ] Quick start guide
- [ ] Known issues
- [ ] Roadmap

#### 52. API Documentation (`docs/API.md`)
- [ ] All API endpoints documented
- [ ] Request/response schemas
- [ ] Example requests
- [ ] Error codes

#### 53. User Guide (`docs/USER_GUIDE.md`)
- [ ] Getting started
- [ ] Dashboard overview
- [ ] Explainability features
- [ ] Alerts & watchlists
- [ ] Risk management
- [ ] Calibration monitoring

#### 54. Git Tag & GitHub Release
- [ ] Create git tag v1.0.0
- [ ] Push tag to GitHub
- [ ] Create GitHub release with CHANGELOG
- [ ] Attach release artifacts (if any)

---

## 📋 TASK #34: Final Acceptance Tests

### End-to-End Tests - 6 Tasks

#### 55. Explainability E2E Test (`tests/e2e/test_explainability_e2e.py`)
- [ ] Create decision via API
- [ ] Fetch explanation
- [ ] Verify reasoning extraction
- [ ] Test frontend rendering (Playwright)
- [ ] Verify confidence display

#### 56. Alerts E2E Test (`tests/e2e/test_alerts_e2e.py`)
- [ ] Create alert
- [ ] Trigger alert condition
- [ ] Verify notification sent
- [ ] Test alert history
- [ ] Test watchlist monitoring

#### 57. Risk Guardrails E2E Test (`tests/e2e/test_risk_e2e.py`)
- [ ] Create risk policy
- [ ] Submit trade violating policy
- [ ] Verify rejection
- [ ] Request override
- [ ] Verify override approval flow

#### 58. Calibration E2E Test (`tests/e2e/test_calibration_e2e.py`)
- [ ] Submit predictions
- [ ] Update with actual outcomes
- [ ] Verify calibration metrics
- [ ] Test calibration curve generation
- [ ] Verify frontend visualization

#### 59. Performance Benchmarks (`tests/benchmarks/test_performance.py`)
- [ ] API response time benchmarks
- [ ] Database query performance
- [ ] Frontend rendering performance
- [ ] WebSocket latency
- [ ] Memory usage profiling

#### 60. Production Readiness Checklist (`tests/production_readiness.md`)
- [ ] All acceptance tests passing
- [ ] Performance benchmarks met
- [ ] Security audit complete
- [ ] Documentation complete
- [ ] Error handling tested
- [ ] Logging configured
- [ ] Monitoring setup
- [ ] Backup strategy defined

---

## 📊 Summary

**Total Tasks: 60**

- Task #29 (Explainability): 11 tasks
- Task #30 (Alerts & Watchlists): 15 tasks
- Task #31 (Risk Guardrails): 12 tasks
- Task #32 (Calibration): 10 tasks
- Task #33 (Release): 6 tasks
- Task #34 (Acceptance Tests): 6 tasks

**Estimated Implementation Time:**
- Backend: ~30 tasks × 2 hours = 60 hours
- Frontend: ~24 tasks × 1.5 hours = 36 hours
- Testing & Documentation: ~6 tasks × 3 hours = 18 hours
- **Total: ~114 hours (14-15 days of work)**

---

## 🚀 Implementation Order

### Week 1: Core Features
1. Task #29: Explainability (Days 1-2)
2. Task #30: Alerts & Watchlists (Days 3-4)
3. Task #31: Risk Guardrails (Days 5-6)

### Week 2: Calibration & Release
4. Task #32: Calibration (Days 7-8)
5. Task #34: Acceptance Tests (Days 9-10)
6. Task #33: Release v1.0.0 (Days 11-12)

### Buffer: Days 13-15 for bug fixes and polish

---

Last updated: 2026-01-05
