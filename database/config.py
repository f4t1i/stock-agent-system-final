#!/usr/bin/env python3
"""
Database Configuration

Supports both SQLite (development) and PostgreSQL (production).
Uses SQLAlchemy for ORM and Alembic for migrations.
"""

import os
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from loguru import logger

# Database URL from environment or default to SQLite
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./data/stock_agent.db"  # Development default
)

# For PostgreSQL in production:
# DATABASE_URL=postgresql://user:password@localhost:5432/stock_agent

# Create engine
if DATABASE_URL.startswith("postgresql"):
    # PostgreSQL configuration
    engine = create_engine(
        DATABASE_URL,
        pool_size=20,
        max_overflow=40,
        pool_pre_ping=True,  # Verify connections before using
        pool_recycle=3600,   # Recycle connections after 1 hour
        echo=os.getenv("SQL_ECHO", "false").lower() == "true"
    )
    logger.info(f"Using PostgreSQL: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'localhost'}")
else:
    # SQLite configuration
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},  # SQLite specific
        echo=os.getenv("SQL_ECHO", "false").lower() == "true"
    )
    logger.info("Using SQLite (development mode)")

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


# Dependency for FastAPI
def get_db() -> Generator[Session, None, None]:
    """
    Get database session for FastAPI dependency injection

    Usage in FastAPI:
        @app.get("/users")
        def get_users(db: Session = Depends(get_db)):
            return db.query(User).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Initialize database
def init_db():
    """Create all tables"""
    from database.models import (
        Analysis,
        ExperienceRecord,
        TrainingRun,
        ModelVersion,
        Alert,
        Watchlist,
        Decision,
        RiskViolation
    )

    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database tables created")


# Drop all tables (use with caution!)
def drop_db():
    """Drop all tables - USE WITH CAUTION!"""
    logger.warning("⚠️  Dropping all database tables...")
    Base.metadata.drop_all(bind=engine)
    logger.info("✅ All tables dropped")


if __name__ == "__main__":
    print("Database Configuration")
    print(f"URL: {DATABASE_URL}")
    print(f"Engine: {engine}")

    # Test connection
    try:
        with engine.connect() as conn:
            print("✅ Database connection successful")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
