#!/usr/bin/env python3
"""
Database Models (SQLAlchemy ORM)

Defines all database tables for:
- Stock analyses
- Training runs and model versions
- Experiences and datasets
- Alerts and watchlists
- Decisions and risk violations
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, JSON, ForeignKey, Index
from sqlalchemy.orm import relationship
from database.config import Base


# ============================================================================
# Analysis & Results
# ============================================================================

class Analysis(Base):
    """Stock analysis results"""
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    recommendation = Column(String(10), nullable=False)  # buy, sell, hold
    confidence = Column(Float, nullable=False)
    position_size = Column(Float, nullable=False)

    # Risk management
    entry_target = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    risk_assessment = Column(Text)

    # Agent outputs
    news_output = Column(JSON)
    technical_output = Column(JSON)
    fundamental_output = Column(JSON)
    strategist_output = Column(JSON)

    # Metadata
    use_supervisor = Column(Boolean, default=False)
    reasoning = Column(Text)
    errors = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Indexes
    __table_args__ = (
        Index('ix_analyses_symbol_created', 'symbol', 'created_at'),
    )


# ============================================================================
# Training & Models
# ============================================================================

class TrainingRun(Base):
    """Training run records"""
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(50), unique=True, nullable=False, index=True)
    agent_type = Column(String(20), nullable=False)  # news, technical, fundamental, strategist
    training_type = Column(String(20), nullable=False)  # sft, grpo, iteration

    # Configuration
    config = Column(JSON)
    base_model = Column(String(100))
    dataset_id = Column(String(50))

    # Metrics
    train_loss = Column(Float)
    eval_loss = Column(Float)
    sharpe_ratio = Column(Float)
    num_epochs = Column(Integer)
    num_iterations = Column(Integer)

    # Status
    status = Column(String(20), default="running")  # running, completed, failed
    error_message = Column(Text)

    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime)

    # Relationships
    model_versions = relationship("ModelVersion", back_populates="training_run")


class ModelVersion(Base):
    """Model version registry"""
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String(100), unique=True, nullable=False, index=True)
    agent_type = Column(String(20), nullable=False)
    version = Column(String(20), nullable=False)  # Semantic version (1.0.0)

    # Metrics
    performance_metrics = Column(JSON)
    eval_metrics = Column(JSON)

    # Metadata
    training_run_id = Column(Integer, ForeignKey("training_runs.id"))
    model_path = Column(String(500))
    config = Column(JSON)
    promoted = Column(Boolean, default=False)  # Promoted to production

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    training_run = relationship("TrainingRun", back_populates="model_versions")


# ============================================================================
# Experiences & Datasets
# ============================================================================

class ExperienceRecord(Base):
    """Experience store records"""
    __tablename__ = "experiences"

    id = Column(Integer, primary_key=True, index=True)
    experience_id = Column(String(50), unique=True, nullable=False, index=True)
    symbol = Column(String(10), nullable=False, index=True)

    # Data
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    reward = Column(Float, nullable=False)

    # Judge evaluation
    judge_score = Column(Float)
    judge_feedback = Column(Text)
    judge_approved = Column(Boolean, default=False, index=True)

    # Metadata
    agent_type = Column(String(20))
    market_regime = Column(String(20))
    metadata = Column(JSON)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Indexes
    __table_args__ = (
        Index('ix_experiences_reward', 'reward'),
        Index('ix_experiences_approved_created', 'judge_approved', 'created_at'),
    )


# ============================================================================
# Alerts & Watchlists
# ============================================================================

class Alert(Base):
    """Price alerts and notifications"""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(String(50), unique=True, nullable=False, index=True)

    # Alert configuration
    symbol = Column(String(10), nullable=False, index=True)
    alert_type = Column(String(30), nullable=False)  # price_threshold, confidence_change, etc.
    condition = Column(String(20), nullable=False)   # above, below, crosses
    threshold = Column(Float, nullable=False)

    # Notification
    notification_channels = Column(JSON)  # ["email", "push", "webhook"]
    message = Column(String(500))

    # Status
    is_active = Column(Boolean, default=True, index=True)
    triggered_count = Column(Integer, default=0)
    last_triggered = Column(DateTime)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Watchlist(Base):
    """User watchlists"""
    __tablename__ = "watchlists"

    id = Column(Integer, primary_key=True, index=True)
    watchlist_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    symbols = Column(JSON, nullable=False)  # List of symbols

    # Metadata
    description = Column(String(500))
    is_active = Column(Boolean, default=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ============================================================================
# Decisions & Risk
# ============================================================================

class Decision(Base):
    """AI decision audit trail"""
    __tablename__ = "decisions"

    id = Column(Integer, primary_key=True, index=True)
    decision_id = Column(String(50), unique=True, nullable=False, index=True)

    # Decision
    symbol = Column(String(10), nullable=False, index=True)
    agent_name = Column(String(30), nullable=False)
    recommendation = Column(String(10), nullable=False)
    confidence = Column(Float, nullable=False)

    # Reasoning
    reasoning = Column(Text, nullable=False)
    factors = Column(JSON)  # Contributing factors
    alternatives = Column(JSON)  # Alternative scenarios

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class RiskViolation(Base):
    """Risk policy violations"""
    __tablename__ = "risk_violations"

    id = Column(Integer, primary_key=True, index=True)
    violation_id = Column(String(50), unique=True, nullable=False, index=True)

    # Violation
    symbol = Column(String(10), nullable=False, index=True)
    policy_id = Column(String(50), nullable=False)
    violation_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)  # low, medium, high, critical

    # Details
    rule_violated = Column(String(200))
    expected_value = Column(Float)
    actual_value = Column(Float)
    message = Column(Text)

    # Override
    override_approved = Column(Boolean, default=False)
    override_reason = Column(Text)
    approved_by = Column(String(100))

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    resolved_at = Column(DateTime)


# ============================================================================
# Helper Functions
# ============================================================================

def create_all_tables(engine):
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)


def drop_all_tables(engine):
    """Drop all database tables - USE WITH CAUTION!"""
    Base.metadata.drop_all(bind=engine)


if __name__ == "__main__":
    from database.config import engine

    print("Database Models")
    print("Tables:", [table.name for table in Base.metadata.sorted_tables])

    # Create tables
    create_all_tables(engine)
    print("✅ All tables created")
