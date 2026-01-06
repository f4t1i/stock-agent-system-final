#!/usr/bin/env python3
"""
Database Initialization Script

Creates all tables and initial data for production deployment.

Usage:
    python database/migrations/init_db.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from database.config import engine, init_db, DATABASE_URL
from database.models import Base
from loguru import logger


def main():
    """Initialize database"""
    logger.info("=" * 80)
    logger.info("DATABASE INITIALIZATION")
    logger.info("=" * 80)

    logger.info(f"\nDatabase URL: {DATABASE_URL}")

    # Create tables
    logger.info("\nCreating tables...")
    init_db()

    # Verify tables
    logger.info("\nVerifying tables...")
    tables = [table.name for table in Base.metadata.sorted_tables]
    logger.info(f"Created {len(tables)} tables:")
    for table in tables:
        logger.info(f"  ✅ {table}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ DATABASE INITIALIZED SUCCESSFULLY")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
