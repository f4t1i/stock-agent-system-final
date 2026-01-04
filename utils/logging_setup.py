"""
Logging Setup - Zentralisierte Logging-Konfiguration
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: str = "logs/system.log",
    rotation: str = "100 MB",
    retention: str = "1 week"
):
    """
    Setup logging configuration with Loguru
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file
        rotation: When to rotate logs
        retention: How long to keep logs
    """
    
    # Remove default handler
    logger.remove()
    
    # Console handler mit Farben
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True
    )
    
    # File handler
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        rotation=rotation,
        retention=retention,
        compression="zip"
    )
    
    # Error-only file
    error_log = log_path.parent / "errors.log"
    logger.add(
        error_log,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation=rotation,
        retention=retention,
        backtrace=True,
        diagnose=True
    )
    
    logger.info(f"Logging initialized at level {level}")
    logger.info(f"Log file: {log_file}")
