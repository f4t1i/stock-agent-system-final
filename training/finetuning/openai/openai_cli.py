#!/usr/bin/env python3
"""
OpenAI Fine-Tuning CLI - Task 5.6

Command-line interface for OpenAI fine-tuning operations.

Commands:
- upload: Upload dataset
- create-job: Create training job
- monitor: Monitor job status
- list-jobs: List training jobs
- test-model: Test model inference
- compare: Compare models
- cancel: Cancel job
- list-files: List uploaded files
- delete-file: Delete uploaded file

Phase A1 Week 5-6: Task 5.6 COMPLETE
"""

import os
import sys
import json
import argparse
from typing import Optional
from loguru import logger

try:
    from .openai_client import OpenAIClient, create_client
    from .dataset_upload import DatasetUploader, validate_dataset
    from .training_job import TrainingJobManager, TrainingConfig, HyperParameters
    from .job_monitor import JobMonitor
    from .model_deployment import ModelDeployer, InferenceRequest
except ImportError:
    from training.finetuning.openai.openai_client import OpenAIClient, create_client
    from training.finetuning.openai.dataset_upload import DatasetUploader, validate_dataset
    from training.finetuning.openai.training_job import TrainingJobManager, TrainingConfig, HyperParameters
    from training.finetuning.openai.job_monitor import JobMonitor
    from training.finetuning.openai.model_deployment import ModelDeployer, InferenceRequest


class OpenAICLI:
    """OpenAI Fine-Tuning CLI"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize CLI
        
        Args:
            api_key: OpenAI API key (uses env var if None)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            logger.error("OPENAI_API_KEY not set")
            sys.exit(1)
        
        self.client = create_client(self.api_key)
        self.uploader = DatasetUploader(self.client)
        self.job_manager = TrainingJobManager(self.client)
        self.monitor = JobMonitor(self.client)
        self.deployer = ModelDeployer(self.client)
        
        logger.info("OpenAI CLI initialized")
    
    def upload_dataset(self, args):
        """Upload dataset command"""
        print(f"Uploading dataset: {args.file}")
        
        try:
            # Validate if requested
            if not args.skip_validation:
                print("Validating dataset...")
                result = validate_dataset(args.file)
                
                if not result.is_valid:
                    print(f"✗ Validation failed:")
                    for error in result.errors:
                        print(f"  - {error}")
                    sys.exit(1)
                
                print(f"✓ Validation passed: {result.example_count} examples")
                
                if result.warnings:
                    print("Warnings:")
                    for warning in result.warnings:
                        print(f"  - {warning}")
            
            # Upload
            print("Uploading to OpenAI...")
            upload_result = self.uploader.upload_file(
                args.file,
                validate=False  # Already validated
            )
            
            print(f"\n✓ Upload complete!")
            print(f"File ID: {upload_result.file_id}")
            print(f"Filename: {upload_result.filename}")
            print(f"Size: {upload_result.bytes} bytes")
            print(f"Status: {upload_result.status}")
            
        except Exception as e:
            print(f"✗ Upload failed: {e}")
            sys.exit(1)
    
    def create_job(self, args):
        """Create training job command"""
        print(f"Creating training job...")
        print(f"Model: {args.model}")
        print(f"Training file: {args.training_file}")
        
        try:
            # Create config
            hyperparameters = HyperParameters(
                n_epochs=args.n_epochs,
                batch_size=args.batch_size,
                learning_rate_multiplier=args.learning_rate_multiplier
            )
            
            config = TrainingConfig(
                training_file=args.training_file,
                model=args.model,
                validation_file=args.validation_file,
                hyperparameters=hyperparameters,
                suffix=args.suffix
            )
            
            # Create job
            job = self.job_manager.create_job(config)
            
            print(f"\n✓ Job created!")
            print(f"Job ID: {job.job_id}")
            print(f"Status: {job.status}")
            print(f"Model: {job.model}")
            
            # Monitor if requested
            if args.monitor:
                print(f"\nMonitoring job...")
                self.monitor_job_internal(job.job_id, args.poll_interval)
            
        except Exception as e:
            print(f"✗ Job creation failed: {e}")
            sys.exit(1)
    
    def monitor_job(self, args):
        """Monitor job command"""
        self.monitor_job_internal(args.job_id, args.poll_interval)
    
    def monitor_job_internal(self, job_id: str, poll_interval: int):
        """Internal monitor job implementation"""
        print(f"Monitoring job: {job_id}")
        
        def on_status_change(job):
            print(f"\nStatus: {job.status}")
            if job.fine_tuned_model:
                print(f"Fine-tuned model: {job.fine_tuned_model}")
        
        def on_event(event):
            print(f"{event}")
        
        try:
            job = self.monitor.monitor_job(
                job_id=job_id,
                poll_interval=poll_interval,
                on_status_change=on_status_change,
                on_event=on_event
            )
            
            print(f"\n{'='*60}")
            if job.is_completed:
                print(f"✓ Job completed successfully!")
                print(f"Fine-tuned model: {job.fine_tuned_model}")
                if job.trained_tokens:
                    print(f"Trained tokens: {job.trained_tokens:,}")
            elif job.is_failed:
                print(f"✗ Job failed!")
                if job.error:
                    print(f"Error: {job.error.get('message', 'Unknown error')}")
            elif job.is_cancelled:
                print(f"⊗ Job cancelled")
            
        except Exception as e:
            print(f"✗ Monitoring failed: {e}")
            sys.exit(1)
    
    def list_jobs(self, args):
        """List jobs command"""
        print(f"Listing training jobs (limit: {args.limit})...")
        
        try:
            jobs = self.job_manager.list_jobs(limit=args.limit)
            
            if not jobs:
                print("No jobs found")
                return
            
            print(f"\nFound {len(jobs)} jobs:\n")
            
            for job in jobs:
                status_emoji = {
                    'succeeded': '✓',
                    'failed': '✗',
                    'cancelled': '⊗',
                    'running': '⟳',
                    'queued': '⋯',
                    'validating_files': '⋯'
                }.get(job.status, '?')
                
                print(f"{status_emoji} {job.job_id}")
                print(f"  Status: {job.status}")
                print(f"  Model: {job.model}")
                if job.fine_tuned_model:
                    print(f"  Fine-tuned: {job.fine_tuned_model}")
                print()
            
        except Exception as e:
            print(f"✗ Failed to list jobs: {e}")
            sys.exit(1)
    
    def test_model(self, args):
        """Test model command"""
        print(f"Testing model: {args.model}")
        
        try:
            # Load messages from file or use prompt
            if args.messages_file:
                with open(args.messages_file, 'r') as f:
                    messages = json.load(f)
            else:
                messages = [{"role": "user", "content": args.prompt}]
            
            # Create request
            request = InferenceRequest(
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
            
            # Test inference
            print("Running inference...")
            response = self.deployer.test_inference(args.model, request)
            
            print(f"\n{'='*60}")
            print(f"Response:")
            print(f"{response.content}")
            print(f"\n{'='*60}")
            print(f"Model: {response.model}")
            print(f"Tokens: {response.total_tokens} ({response.prompt_tokens} prompt + {response.completion_tokens} completion)")
            print(f"Latency: {response.latency_ms:.0f}ms")
            print(f"Cost: ${response.cost_estimate:.6f}")
            print(f"Finish reason: {response.finish_reason}")
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            sys.exit(1)
    
    def compare_models(self, args):
        """Compare models command"""
        print(f"Comparing models:")
        print(f"  Base: {args.base_model}")
        print(f"  Fine-tuned: {args.fine_tuned_model}")
        
        try:
            # Load test cases
            with open(args.test_file, 'r') as f:
                test_data = json.load(f)
            
            # Create requests
            test_cases = [
                InferenceRequest(messages=test['messages'])
                for test in test_data
            ]
            
            print(f"\nRunning {len(test_cases)} test cases...")
            
            # Compare
            result = self.deployer.compare_models(
                args.base_model,
                args.fine_tuned_model,
                test_cases
            )
            
            print(f"\n{'='*60}")
            print(f"Comparison Results:")
            print(f"  Test cases: {result.test_cases}")
            print(f"  Latency improvement: {result.avg_latency_improvement:+.1f}%")
            print(f"  Token difference: {result.avg_token_difference:+.1f}")
            
            # Cost comparison
            base_costs = self.deployer.estimate_costs(result.base_responses)
            ft_costs = self.deployer.estimate_costs(result.fine_tuned_responses)
            
            print(f"\nCost Comparison:")
            print(f"  Base model: ${base_costs['total_cost_usd']:.6f}")
            print(f"  Fine-tuned: ${ft_costs['total_cost_usd']:.6f}")
            print(f"  Difference: ${ft_costs['total_cost_usd'] - base_costs['total_cost_usd']:+.6f}")
            
        except Exception as e:
            print(f"✗ Comparison failed: {e}")
            sys.exit(1)
    
    def cancel_job(self, args):
        """Cancel job command"""
        print(f"Cancelling job: {args.job_id}")
        
        try:
            job = self.job_manager.cancel_job(args.job_id)
            print(f"✓ Job cancelled")
            print(f"Status: {job.status}")
            
        except Exception as e:
            print(f"✗ Cancellation failed: {e}")
            sys.exit(1)
    
    def list_files(self, args):
        """List files command"""
        print("Listing uploaded files...")
        
        try:
            files = self.uploader.list_files(purpose=args.purpose)
            
            if not files:
                print("No files found")
                return
            
            print(f"\nFound {len(files)} files:\n")
            
            for file_obj in files:
                print(f"• {file_obj.id}")
                print(f"  Filename: {file_obj.filename}")
                print(f"  Size: {file_obj.bytes} bytes")
                print(f"  Purpose: {file_obj.purpose}")
                print(f"  Status: {file_obj.status}")
                print()
            
        except Exception as e:
            print(f"✗ Failed to list files: {e}")
            sys.exit(1)
    
    def delete_file(self, args):
        """Delete file command"""
        print(f"Deleting file: {args.file_id}")
        
        try:
            success = self.uploader.delete_file(args.file_id)
            if success:
                print("✓ File deleted")
            else:
                print("✗ Deletion failed")
                sys.exit(1)
            
        except Exception as e:
            print(f"✗ Deletion failed: {e}")
            sys.exit(1)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="OpenAI Fine-Tuning CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--api-key',
        help='OpenAI API key (or set OPENAI_API_KEY env var)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Upload command
    upload_parser = subparsers.add_parser('upload', help='Upload dataset')
    upload_parser.add_argument('file', help='Dataset file (JSONL)')
    upload_parser.add_argument('--skip-validation', action='store_true', help='Skip validation')
    
    # Create job command
    create_parser = subparsers.add_parser('create-job', help='Create training job')
    create_parser.add_argument('training_file', help='Training file ID')
    create_parser.add_argument('--model', default='gpt-3.5-turbo', help='Base model')
    create_parser.add_argument('--validation-file', help='Validation file ID')
    create_parser.add_argument('--n-epochs', type=int, help='Number of epochs')
    create_parser.add_argument('--batch-size', type=int, help='Batch size')
    create_parser.add_argument('--learning-rate-multiplier', type=float, help='Learning rate multiplier')
    create_parser.add_argument('--suffix', help='Model name suffix')
    create_parser.add_argument('--monitor', action='store_true', help='Monitor after creation')
    create_parser.add_argument('--poll-interval', type=int, default=60, help='Poll interval (seconds)')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor job')
    monitor_parser.add_argument('job_id', help='Job ID')
    monitor_parser.add_argument('--poll-interval', type=int, default=60, help='Poll interval (seconds)')
    
    # List jobs command
    list_parser = subparsers.add_parser('list-jobs', help='List training jobs')
    list_parser.add_argument('--limit', type=int, default=20, help='Maximum jobs to list')
    
    # Test model command
    test_parser = subparsers.add_parser('test-model', help='Test model')
    test_parser.add_argument('model', help='Model ID')
    test_parser.add_argument('--prompt', help='Test prompt')
    test_parser.add_argument('--messages-file', help='JSON file with messages')
    test_parser.add_argument('--temperature', type=float, default=0.7, help='Temperature')
    test_parser.add_argument('--max-tokens', type=int, help='Max tokens')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare models')
    compare_parser.add_argument('base_model', help='Base model ID')
    compare_parser.add_argument('fine_tuned_model', help='Fine-tuned model ID')
    compare_parser.add_argument('test_file', help='JSON file with test cases')
    
    # Cancel command
    cancel_parser = subparsers.add_parser('cancel', help='Cancel job')
    cancel_parser.add_argument('job_id', help='Job ID')
    
    # List files command
    files_parser = subparsers.add_parser('list-files', help='List uploaded files')
    files_parser.add_argument('--purpose', help='Filter by purpose')
    
    # Delete file command
    delete_parser = subparsers.add_parser('delete-file', help='Delete file')
    delete_parser.add_argument('file_id', help='File ID')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Create CLI
    cli = OpenAICLI(api_key=args.api_key)
    
    # Route to command
    if args.command == 'upload':
        cli.upload_dataset(args)
    elif args.command == 'create-job':
        cli.create_job(args)
    elif args.command == 'monitor':
        cli.monitor_job(args)
    elif args.command == 'list-jobs':
        cli.list_jobs(args)
    elif args.command == 'test-model':
        cli.test_model(args)
    elif args.command == 'compare':
        cli.compare_models(args)
    elif args.command == 'cancel':
        cli.cancel_job(args)
    elif args.command == 'list-files':
        cli.list_files(args)
    elif args.command == 'delete-file':
        cli.delete_file(args)


if __name__ == "__main__":
    main()
