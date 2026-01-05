#!/usr/bin/env python3
"""
Training Job Creation & Configuration - Task 5.3

Create and configure fine-tuning jobs for OpenAI.

Features:
- Fine-tuning job creation
- Hyperparameter configuration
- Model selection
- Training options
- Validation split
- Job metadata tracking

Phase A1 Week 5-6: Task 5.3 COMPLETE
"""

import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from loguru import logger

try:
    from .openai_client import OpenAIClient
except ImportError:
    from training.finetuning.openai.openai_client import OpenAIClient


@dataclass
class HyperParameters:
    """Fine-tuning hyperparameters"""
    n_epochs: Optional[int] = None  # Number of epochs (auto if None)
    batch_size: Optional[int] = None  # Batch size (auto if None)
    learning_rate_multiplier: Optional[float] = None  # Learning rate multiplier (auto if None)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, excluding None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class TrainingConfig:
    """Training job configuration"""
    training_file: str  # File ID of training data
    model: str = "gpt-3.5-turbo"  # Base model
    validation_file: Optional[str] = None  # File ID of validation data
    hyperparameters: Optional[HyperParameters] = None  # Hyperparameters
    suffix: Optional[str] = None  # Model name suffix (max 40 chars)
    
    def __post_init__(self):
        if self.hyperparameters is None:
            self.hyperparameters = HyperParameters()


@dataclass
class TrainingJob:
    """Training job information"""
    job_id: str
    model: str
    status: str
    created_at: int
    training_file: str
    validation_file: Optional[str] = None
    fine_tuned_model: Optional[str] = None
    finished_at: Optional[int] = None
    trained_tokens: Optional[int] = None
    error: Optional[Dict[str, Any]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    result_files: Optional[List[str]] = None
    
    @classmethod
    def from_openai_job(cls, job_obj) -> 'TrainingJob':
        """Create from OpenAI job object"""
        return cls(
            job_id=job_obj.id,
            model=job_obj.model,
            status=job_obj.status,
            created_at=job_obj.created_at,
            training_file=job_obj.training_file,
            validation_file=getattr(job_obj, 'validation_file', None),
            fine_tuned_model=getattr(job_obj, 'fine_tuned_model', None),
            finished_at=getattr(job_obj, 'finished_at', None),
            trained_tokens=getattr(job_obj, 'trained_tokens', None),
            error=getattr(job_obj, 'error', None),
            hyperparameters=getattr(job_obj, 'hyperparameters', None),
            result_files=getattr(job_obj, 'result_files', None)
        )
    
    @property
    def is_completed(self) -> bool:
        """Check if job is completed"""
        return self.status == 'succeeded'
    
    @property
    def is_failed(self) -> bool:
        """Check if job failed"""
        return self.status == 'failed'
    
    @property
    def is_running(self) -> bool:
        """Check if job is running"""
        return self.status in ['validating_files', 'queued', 'running']
    
    @property
    def is_cancelled(self) -> bool:
        """Check if job was cancelled"""
        return self.status == 'cancelled'
    
    def __str__(self) -> str:
        status_emoji = {
            'succeeded': '✓',
            'failed': '✗',
            'cancelled': '⊗',
            'running': '⟳',
            'queued': '⋯',
            'validating_files': '⋯'
        }.get(self.status, '?')
        
        return (
            f"{status_emoji} Job {self.job_id}\n"
            f"Status: {self.status}\n"
            f"Model: {self.model}\n"
            f"Fine-tuned: {self.fine_tuned_model or 'N/A'}"
        )


class TrainingJobManager:
    """
    Manage OpenAI fine-tuning jobs
    
    Handles:
    - Job creation
    - Job configuration
    - Job listing
    - Job cancellation
    """
    
    def __init__(self, client: OpenAIClient):
        """
        Initialize job manager
        
        Args:
            client: OpenAI client
        """
        self.client = client
        logger.info("TrainingJobManager initialized")
    
    def create_job(
        self,
        config: TrainingConfig,
        wait_for_completion: bool = False,
        poll_interval: int = 60
    ) -> TrainingJob:
        """
        Create fine-tuning job
        
        Args:
            config: Training configuration
            wait_for_completion: Wait for job to complete
            poll_interval: Polling interval in seconds
        
        Returns:
            Training job
        """
        try:
            logger.info(f"Creating fine-tuning job: {config.model}")
            
            # Prepare job parameters
            params = {
                'training_file': config.training_file,
                'model': config.model
            }
            
            # Add optional parameters
            if config.validation_file:
                params['validation_file'] = config.validation_file
            
            if config.suffix:
                params['suffix'] = config.suffix[:40]  # Max 40 chars
            
            # Add hyperparameters
            if config.hyperparameters:
                hp_dict = config.hyperparameters.to_dict()
                if hp_dict:
                    params['hyperparameters'] = hp_dict
            
            # Create job
            job_obj = self.client.client.fine_tuning.jobs.create(**params)
            
            job = TrainingJob.from_openai_job(job_obj)
            
            logger.info(
                f"✓ Job created: {job.job_id} "
                f"(status: {job.status})"
            )
            
            # Wait for completion if requested
            if wait_for_completion:
                logger.info(f"Waiting for job completion (polling every {poll_interval}s)...")
                job = self.wait_for_completion(job.job_id, poll_interval)
            
            return job
            
        except Exception as e:
            logger.error(f"Failed to create job: {e}")
            raise
    
    def get_job(self, job_id: str) -> TrainingJob:
        """
        Get job details
        
        Args:
            job_id: Job ID
        
        Returns:
            Training job
        """
        try:
            job_obj = self.client.client.fine_tuning.jobs.retrieve(job_id)
            job = TrainingJob.from_openai_job(job_obj)
            
            logger.info(f"Retrieved job: {job_id} (status: {job.status})")
            return job
            
        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {e}")
            raise
    
    def list_jobs(self, limit: int = 20) -> List[TrainingJob]:
        """
        List fine-tuning jobs
        
        Args:
            limit: Maximum number of jobs to return
        
        Returns:
            List of training jobs
        """
        try:
            jobs_obj = self.client.client.fine_tuning.jobs.list(limit=limit)
            
            jobs = [
                TrainingJob.from_openai_job(job_obj)
                for job_obj in jobs_obj.data
            ]
            
            logger.info(f"Listed {len(jobs)} jobs")
            return jobs
            
        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            raise
    
    def cancel_job(self, job_id: str) -> TrainingJob:
        """
        Cancel running job
        
        Args:
            job_id: Job ID
        
        Returns:
            Updated training job
        """
        try:
            job_obj = self.client.client.fine_tuning.jobs.cancel(job_id)
            job = TrainingJob.from_openai_job(job_obj)
            
            logger.info(f"Cancelled job: {job_id}")
            return job
            
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            raise
    
    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 60,
        timeout: Optional[int] = None
    ) -> TrainingJob:
        """
        Wait for job to complete
        
        Args:
            job_id: Job ID
            poll_interval: Polling interval in seconds
            timeout: Timeout in seconds (None for no timeout)
        
        Returns:
            Completed training job
        
        Raises:
            TimeoutError: If timeout is reached
        """
        start_time = time.time()
        
        while True:
            job = self.get_job(job_id)
            
            if job.is_completed:
                logger.info(f"✓ Job completed: {job.job_id}")
                return job
            
            if job.is_failed:
                error_msg = job.error.get('message', 'Unknown error') if job.error else 'Unknown error'
                logger.error(f"✗ Job failed: {error_msg}")
                return job
            
            if job.is_cancelled:
                logger.warning(f"⊗ Job cancelled: {job.job_id}")
                return job
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")
            
            # Log progress
            logger.info(f"Job {job_id} status: {job.status}")
            
            # Wait before next poll
            time.sleep(poll_interval)
    
    def list_job_events(self, job_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        List job events (logs)
        
        Args:
            job_id: Job ID
            limit: Maximum number of events to return
        
        Returns:
            List of event dicts
        """
        try:
            events_obj = self.client.client.fine_tuning.jobs.list_events(
                fine_tuning_job_id=job_id,
                limit=limit
            )
            
            events = [
                {
                    'created_at': event.created_at,
                    'level': event.level,
                    'message': event.message
                }
                for event in events_obj.data
            ]
            
            logger.info(f"Listed {len(events)} events for job {job_id}")
            return events
            
        except Exception as e:
            logger.error(f"Failed to list events for job {job_id}: {e}")
            raise
    
    def get_job_metrics(self, job_id: str) -> Dict[str, Any]:
        """
        Get job training metrics
        
        Args:
            job_id: Job ID
        
        Returns:
            Metrics dict
        """
        job = self.get_job(job_id)
        
        metrics = {
            'job_id': job.job_id,
            'status': job.status,
            'trained_tokens': job.trained_tokens,
            'hyperparameters': job.hyperparameters
        }
        
        # Get result files if available
        if job.result_files:
            metrics['result_files'] = job.result_files
        
        return metrics


# Helper functions

def create_training_job(
    client: OpenAIClient,
    training_file: str,
    model: str = "gpt-3.5-turbo",
    validation_file: Optional[str] = None,
    n_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate_multiplier: Optional[float] = None,
    suffix: Optional[str] = None,
    wait_for_completion: bool = False
) -> TrainingJob:
    """
    Create training job with simplified parameters
    
    Args:
        client: OpenAI client
        training_file: Training file ID
        model: Base model
        validation_file: Validation file ID
        n_epochs: Number of epochs
        batch_size: Batch size
        learning_rate_multiplier: Learning rate multiplier
        suffix: Model name suffix
        wait_for_completion: Wait for completion
    
    Returns:
        Training job
    """
    config = TrainingConfig(
        training_file=training_file,
        model=model,
        validation_file=validation_file,
        hyperparameters=HyperParameters(
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate_multiplier=learning_rate_multiplier
        ),
        suffix=suffix
    )
    
    manager = TrainingJobManager(client)
    return manager.create_job(config, wait_for_completion=wait_for_completion)


def get_available_models() -> List[str]:
    """
    Get list of models available for fine-tuning
    
    Returns:
        List of model names
    """
    # As of 2024, these are the main models supporting fine-tuning
    return [
        'gpt-3.5-turbo',
        'gpt-3.5-turbo-0125',
        'gpt-3.5-turbo-1106',
        'gpt-4',
        'gpt-4-0613',
        'babbage-002',
        'davinci-002'
    ]


if __name__ == "__main__":
    # Example usage
    import sys
    
    print("=== Training Job Test ===\n")
    
    # Test 1: Create config
    try:
        print("Test 1: Create training config")
        config = TrainingConfig(
            training_file="file-abc123",
            model="gpt-3.5-turbo",
            hyperparameters=HyperParameters(
                n_epochs=3,
                batch_size=4,
                learning_rate_multiplier=1.5
            ),
            suffix="test-model"
        )
        print(f"✓ Config created: {config.model}\n")
    except Exception as e:
        print(f"✗ Failed: {e}\n")
        sys.exit(1)
    
    # Test 2: Hyperparameters
    try:
        print("Test 2: Hyperparameters")
        hp = HyperParameters(n_epochs=5, batch_size=8)
        hp_dict = hp.to_dict()
        print(f"✓ Hyperparameters: {hp_dict}\n")
    except Exception as e:
        print(f"✗ Failed: {e}\n")
    
    # Test 3: Available models
    try:
        print("Test 3: Available models")
        models = get_available_models()
        print(f"✓ {len(models)} models available\n")
        for model in models[:3]:
            print(f"  - {model}")
        print()
    except Exception as e:
        print(f"✗ Failed: {e}\n")
    
    print("=== Tests Complete ===")
