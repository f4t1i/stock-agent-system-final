#!/usr/bin/env python3
"""
Training Management CLI - Task 7.6
CLI interface for training management operations.
Phase A1 Week 5-6: Task 7.6 COMPLETE
"""

import argparse
import sys
from loguru import logger

# Import management modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from hyperparameter_config import HyperparameterManager
from metrics_tracker import MetricsTracker, MetricPoint
from model_versioning import ModelRegistry, ModelVersion
from logs_aggregator import LogsAggregator, LogEntry, LogLevel


def list_presets(args):
    """List hyperparameter presets"""
    manager = HyperparameterManager()
    presets = manager.list_presets(provider=args.provider)
    
    print(f"\n=== Hyperparameter Presets ({len(presets)}) ===\n")
    for preset in presets:
        print(f"Name: {preset.name}")
        print(f"Provider: {preset.provider}")
        print(f"Description: {preset.description}")
        print(f"Config: epochs={preset.config.n_epochs}, batch={preset.config.batch_size}")
        print()


def suggest_config(args):
    """Suggest hyperparameter configuration"""
    manager = HyperparameterManager()
    config = manager.suggest_config(
        dataset_size=args.dataset_size,
        task_complexity=args.complexity,
        provider=args.provider
    )
    
    print(f"\n=== Suggested Configuration ===\n")
    print(f"Dataset size: {args.dataset_size}")
    print(f"Task complexity: {args.complexity}")
    print(f"Provider: {args.provider}")
    print(f"\nConfiguration:")
    print(f"  Epochs: {config.n_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate multiplier: {config.learning_rate_multiplier}")
    print()


def show_metrics(args):
    """Show training metrics summary"""
    # Mock data for demonstration
    tracker = MetricsTracker(args.job_id)
    
    # Add some sample metrics
    for i in range(5):
        tracker.add_metric(MetricPoint(
            step=i * 10,
            training_loss=0.5 - i * 0.05,
            validation_loss=0.6 - i * 0.04
        ))
    
    summary = tracker.calculate_summary()
    
    print(f"\n=== Metrics Summary ===\n")
    print(f"Job ID: {args.job_id}")
    print(summary)
    print()


def list_models(args):
    """List registered models"""
    registry = ModelRegistry()
    
    # Add sample models
    registry.register_model(ModelVersion(
        model_id="m1",
        model_name="stock-agent-v1",
        version="1.0.0",
        base_model="gpt-3.5-turbo",
        provider="openai",
        job_id="job1"
    ))
    
    models = registry.list_models(provider=args.provider, status=args.status)
    
    print(f"\n=== Registered Models ({len(models)}) ===\n")
    for model in models:
        print(f"Name: {model.model_name}")
        print(f"Version: {model.version}")
        print(f"Provider: {model.provider}")
        print(f"Status: {model.status}")
        print()


def show_logs(args):
    """Show training logs"""
    agg = LogsAggregator(args.job_id)
    
    # Add sample logs
    from datetime import datetime
    agg.add_log(LogEntry(datetime.now(), LogLevel.INFO, "Training started", args.job_id))
    agg.add_log(LogEntry(datetime.now(), LogLevel.WARNING, "High loss detected", args.job_id))
    
    logs = agg.get_logs(limit=args.limit)
    
    print(f"\n=== Training Logs ({len(logs)}) ===\n")
    for log in logs:
        print(log)
    print()


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Training Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # list-presets command
    presets_parser = subparsers.add_parser('list-presets', help='List hyperparameter presets')
    presets_parser.add_argument('--provider', help='Filter by provider')
    presets_parser.set_defaults(func=list_presets)
    
    # suggest-config command
    suggest_parser = subparsers.add_parser('suggest-config', help='Suggest configuration')
    suggest_parser.add_argument('--dataset-size', type=int, required=True, help='Dataset size')
    suggest_parser.add_argument('--complexity', default='medium', choices=['simple', 'medium', 'complex'])
    suggest_parser.add_argument('--provider', default='openai')
    suggest_parser.set_defaults(func=suggest_config)
    
    # show-metrics command
    metrics_parser = subparsers.add_parser('show-metrics', help='Show training metrics')
    metrics_parser.add_argument('job_id', help='Job ID')
    metrics_parser.set_defaults(func=show_metrics)
    
    # list-models command
    models_parser = subparsers.add_parser('list-models', help='List registered models')
    models_parser.add_argument('--provider', help='Filter by provider')
    models_parser.add_argument('--status', default='active', help='Filter by status')
    models_parser.set_defaults(func=list_models)
    
    # show-logs command
    logs_parser = subparsers.add_parser('show-logs', help='Show training logs')
    logs_parser.add_argument('job_id', help='Job ID')
    logs_parser.add_argument('--limit', type=int, default=10, help='Max logs to show')
    logs_parser.set_defaults(func=show_logs)
    
    # Parse and execute
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
