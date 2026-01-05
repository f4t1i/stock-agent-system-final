#!/usr/bin/env python3
"""
Training Logs Aggregation - Task 7.5
Aggregate and analyze training logs from multiple sources.
Phase A1 Week 5-6: Task 7.5 COMPLETE
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from loguru import logger


class LogLevel(Enum):
    """Log level"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class LogEntry:
    """Single log entry"""
    timestamp: datetime
    level: LogLevel
    message: str
    job_id: str
    source: str = "training"  # training, validation, system
    metadata: Dict = field(default_factory=dict)
    
    def __str__(self) -> str:
        emoji = {"info": "ℹ", "warning": "⚠", "error": "✗"}[self.level.value]
        time_str = self.timestamp.strftime("%H:%M:%S")
        return f"{emoji} [{time_str}] {self.message}"


@dataclass
class LogSummary:
    """Logs summary"""
    total_entries: int
    info_count: int
    warning_count: int
    error_count: int
    first_timestamp: Optional[datetime] = None
    last_timestamp: Optional[datetime] = None
    
    def __str__(self) -> str:
        return f"""Log Summary:
  Total: {self.total_entries}
  Info: {self.info_count}, Warnings: {self.warning_count}, Errors: {self.error_count}
  Time range: {self.first_timestamp} to {self.last_timestamp}"""


class LogsAggregator:
    """Aggregate training logs"""
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.logs: List[LogEntry] = []
        logger.info(f"LogsAggregator initialized for job {job_id}")
    
    def add_log(self, log: LogEntry):
        """Add log entry"""
        self.logs.append(log)
    
    def get_logs(
        self,
        level: Optional[LogLevel] = None,
        source: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[LogEntry]:
        """Get filtered logs"""
        filtered = self.logs
        
        if level:
            filtered = [log for log in filtered if log.level == level]
        if source:
            filtered = [log for log in filtered if log.source == source]
        
        # Sort by timestamp
        filtered = sorted(filtered, key=lambda x: x.timestamp)
        
        if limit:
            filtered = filtered[-limit:]
        
        return filtered
    
    def get_errors(self) -> List[LogEntry]:
        """Get all error logs"""
        return self.get_logs(level=LogLevel.ERROR)
    
    def get_warnings(self) -> List[LogEntry]:
        """Get all warning logs"""
        return self.get_logs(level=LogLevel.WARNING)
    
    def calculate_summary(self) -> LogSummary:
        """Calculate summary statistics"""
        if not self.logs:
            return LogSummary(0, 0, 0, 0)
        
        info_count = sum(1 for log in self.logs if log.level == LogLevel.INFO)
        warning_count = sum(1 for log in self.logs if log.level == LogLevel.WARNING)
        error_count = sum(1 for log in self.logs if log.level == LogLevel.ERROR)
        
        timestamps = [log.timestamp for log in self.logs]
        
        return LogSummary(
            total_entries=len(self.logs),
            info_count=info_count,
            warning_count=warning_count,
            error_count=error_count,
            first_timestamp=min(timestamps),
            last_timestamp=max(timestamps)
        )
    
    def export_logs(self, file_path: str):
        """Export logs to file"""
        with open(file_path, 'w') as f:
            for log in self.logs:
                f.write(f"{log}\n")
        logger.info(f"✓ Logs exported: {file_path}")


class MultiJobLogsAggregator:
    """Aggregate logs across multiple jobs"""
    
    def __init__(self):
        self.aggregators: Dict[str, LogsAggregator] = {}
    
    def add_aggregator(self, aggregator: LogsAggregator):
        """Add aggregator"""
        self.aggregators[aggregator.job_id] = aggregator
    
    def get_aggregator(self, job_id: str) -> Optional[LogsAggregator]:
        """Get aggregator by job ID"""
        return self.aggregators.get(job_id)
    
    def get_all_errors(self) -> Dict[str, List[LogEntry]]:
        """Get all errors across jobs"""
        return {
            job_id: agg.get_errors()
            for job_id, agg in self.aggregators.items()
        }
    
    def get_job_summaries(self) -> Dict[str, LogSummary]:
        """Get summaries for all jobs"""
        return {
            job_id: agg.calculate_summary()
            for job_id, agg in self.aggregators.items()
        }


if __name__ == "__main__":
    print("=== Logs Aggregator Test ===\n")
    
    # Test 1: Create aggregator
    print("Test 1: Create aggregator")
    agg = LogsAggregator("job-123")
    print(f"✓ Aggregator created for {agg.job_id}\n")
    
    # Test 2: Add logs
    print("Test 2: Add logs")
    agg.add_log(LogEntry(datetime.now(), LogLevel.INFO, "Training started", "job-123"))
    agg.add_log(LogEntry(datetime.now(), LogLevel.WARNING, "High loss detected", "job-123"))
    agg.add_log(LogEntry(datetime.now(), LogLevel.ERROR, "Training failed", "job-123"))
    print(f"✓ Added {len(agg.logs)} logs\n")
    
    # Test 3: Get errors
    print("Test 3: Get errors")
    errors = agg.get_errors()
    print(f"✓ Found {len(errors)} errors:")
    for err in errors:
        print(f"  {err}")
    print()
    
    # Test 4: Calculate summary
    print("Test 4: Calculate summary")
    summary = agg.calculate_summary()
    print(f"✓ {summary}\n")
    
    print("=== Tests Complete ===")
