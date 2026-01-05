#!/usr/bin/env python3
"""
Training Registry Database - Task 7.1

Python interface for training job registry.

Phase A1 Week 5-6: Task 7.1 COMPLETE
"""

import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    logger.error("psycopg2 not installed. Run: pip install psycopg2-binary")
    raise


@dataclass
class TrainingJobRecord:
    """Training job record"""
    job_id: str
    provider: str
    provider_job_id: str
    base_model: str
    training_file_id: str
    status: str
    validation_file_id: Optional[str] = None
    error_message: Optional[str] = None
    fine_tuned_model: Optional[str] = None
    trained_tokens: Optional[int] = None
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class HyperparametersRecord:
    """Hyperparameters record"""
    hyperparameter_id: str
    job_id: str
    n_epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate_multiplier: Optional[float] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None


@dataclass
class MetricsRecord:
    """Training metrics record"""
    metric_id: str
    job_id: str
    step: int
    epoch: Optional[float] = None
    training_loss: Optional[float] = None
    validation_loss: Optional[float] = None
    training_accuracy: Optional[float] = None
    validation_accuracy: Optional[float] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    recorded_at: Optional[datetime] = None


@dataclass
class ModelRecord:
    """Fine-tuned model record"""
    model_id: str
    job_id: str
    model_name: str
    base_model: str
    provider: str
    version: str
    status: str
    parent_model_id: Optional[str] = None
    final_training_loss: Optional[float] = None
    final_validation_loss: Optional[float] = None
    trained_tokens: Optional[int] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    deprecated_at: Optional[datetime] = None


class TrainingRegistryDB:
    """Training registry database interface"""
    
    def __init__(self, connection_string: str):
        """
        Initialize database interface
        
        Args:
            connection_string: Postgres connection string
        """
        self.connection_string = connection_string
        self.conn = None
        logger.info("TrainingRegistryDB initialized")
    
    def connect(self):
        """Connect to database"""
        self.conn = psycopg2.connect(self.connection_string)
        logger.info("✓ Connected to database")
    
    def disconnect(self):
        """Disconnect from database"""
        if self.conn:
            self.conn.close()
            logger.info("✓ Disconnected from database")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
    
    # Training jobs operations
    
    def create_job(self, job: TrainingJobRecord) -> str:
        """
        Create training job
        
        Args:
            job: Job record
        
        Returns:
            Job ID
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO training_jobs (
                    provider, provider_job_id, base_model,
                    training_file_id, validation_file_id, status,
                    created_by
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING job_id
            """, (
                job.provider,
                job.provider_job_id,
                job.base_model,
                job.training_file_id,
                job.validation_file_id,
                job.status,
                job.created_by
            ))
            job_id = cur.fetchone()[0]
            self.conn.commit()
            logger.info(f"✓ Job created: {job_id}")
            return str(job_id)
    
    def get_job(self, job_id: str) -> Optional[TrainingJobRecord]:
        """
        Get training job
        
        Args:
            job_id: Job ID
        
        Returns:
            Job record or None
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM training_jobs WHERE job_id = %s
            """, (job_id,))
            row = cur.fetchone()
            if row:
                return TrainingJobRecord(**dict(row))
            return None
    
    def update_job_status(
        self,
        job_id: str,
        status: str,
        error_message: Optional[str] = None,
        fine_tuned_model: Optional[str] = None,
        trained_tokens: Optional[int] = None
    ):
        """
        Update job status
        
        Args:
            job_id: Job ID
            status: New status
            error_message: Error message (if failed)
            fine_tuned_model: Fine-tuned model name
            trained_tokens: Trained tokens count
        """
        with self.conn.cursor() as cur:
            # Update timestamps based on status
            if status == 'running':
                cur.execute("""
                    UPDATE training_jobs
                    SET status = %s, started_at = NOW()
                    WHERE job_id = %s
                """, (status, job_id))
            elif status in ('succeeded', 'failed', 'cancelled'):
                cur.execute("""
                    UPDATE training_jobs
                    SET status = %s, finished_at = NOW(),
                        error_message = %s, fine_tuned_model = %s,
                        trained_tokens = %s
                    WHERE job_id = %s
                """, (status, error_message, fine_tuned_model, trained_tokens, job_id))
            else:
                cur.execute("""
                    UPDATE training_jobs
                    SET status = %s
                    WHERE job_id = %s
                """, (status, job_id))
            
            self.conn.commit()
            logger.info(f"✓ Job status updated: {job_id} -> {status}")
    
    def list_jobs(
        self,
        provider: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[TrainingJobRecord]:
        """
        List training jobs
        
        Args:
            provider: Filter by provider
            status: Filter by status
            limit: Maximum results
        
        Returns:
            List of job records
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = "SELECT * FROM training_jobs WHERE 1=1"
            params = []
            
            if provider:
                query += " AND provider = %s"
                params.append(provider)
            
            if status:
                query += " AND status = %s"
                params.append(status)
            
            query += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit)
            
            cur.execute(query, params)
            return [TrainingJobRecord(**dict(row)) for row in cur.fetchall()]
    
    # Hyperparameters operations
    
    def create_hyperparameters(self, hyper: HyperparametersRecord) -> str:
        """
        Create hyperparameters
        
        Args:
            hyper: Hyperparameters record
        
        Returns:
            Hyperparameter ID
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO training_hyperparameters (
                    job_id, n_epochs, batch_size,
                    learning_rate_multiplier, additional_params
                )
                VALUES (%s, %s, %s, %s, %s)
                RETURNING hyperparameter_id
            """, (
                hyper.job_id,
                hyper.n_epochs,
                hyper.batch_size,
                hyper.learning_rate_multiplier,
                psycopg2.extras.Json(hyper.additional_params)
            ))
            hyper_id = cur.fetchone()[0]
            self.conn.commit()
            logger.info(f"✓ Hyperparameters created: {hyper_id}")
            return str(hyper_id)
    
    def get_hyperparameters(self, job_id: str) -> Optional[HyperparametersRecord]:
        """
        Get hyperparameters for job
        
        Args:
            job_id: Job ID
        
        Returns:
            Hyperparameters record or None
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM training_hyperparameters WHERE job_id = %s
            """, (job_id,))
            row = cur.fetchone()
            if row:
                return HyperparametersRecord(**dict(row))
            return None
    
    # Metrics operations
    
    def create_metric(self, metric: MetricsRecord) -> str:
        """
        Create training metric
        
        Args:
            metric: Metrics record
        
        Returns:
            Metric ID
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO training_metrics (
                    job_id, step, epoch, training_loss,
                    validation_loss, training_accuracy,
                    validation_accuracy, additional_metrics
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING metric_id
            """, (
                metric.job_id,
                metric.step,
                metric.epoch,
                metric.training_loss,
                metric.validation_loss,
                metric.training_accuracy,
                metric.validation_accuracy,
                psycopg2.extras.Json(metric.additional_metrics)
            ))
            metric_id = cur.fetchone()[0]
            self.conn.commit()
            return str(metric_id)
    
    def get_metrics(self, job_id: str) -> List[MetricsRecord]:
        """
        Get all metrics for job
        
        Args:
            job_id: Job ID
        
        Returns:
            List of metrics records
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM training_metrics
                WHERE job_id = %s
                ORDER BY step
            """, (job_id,))
            return [MetricsRecord(**dict(row)) for row in cur.fetchall()]
    
    # Model operations
    
    def create_model(self, model: ModelRecord) -> str:
        """
        Create model record
        
        Args:
            model: Model record
        
        Returns:
            Model ID
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO training_models (
                    job_id, model_name, base_model, provider,
                    version, parent_model_id, status,
                    final_training_loss, final_validation_loss,
                    trained_tokens, description, tags
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING model_id
            """, (
                model.job_id,
                model.model_name,
                model.base_model,
                model.provider,
                model.version,
                model.parent_model_id,
                model.status,
                model.final_training_loss,
                model.final_validation_loss,
                model.trained_tokens,
                model.description,
                model.tags
            ))
            model_id = cur.fetchone()[0]
            self.conn.commit()
            logger.info(f"✓ Model created: {model_id}")
            return str(model_id)
    
    def get_model(self, model_id: str) -> Optional[ModelRecord]:
        """
        Get model record
        
        Args:
            model_id: Model ID
        
        Returns:
            Model record or None
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM training_models WHERE model_id = %s
            """, (model_id,))
            row = cur.fetchone()
            if row:
                return ModelRecord(**dict(row))
            return None
    
    def list_models(
        self,
        provider: Optional[str] = None,
        status: str = 'active',
        limit: int = 100
    ) -> List[ModelRecord]:
        """
        List models
        
        Args:
            provider: Filter by provider
            status: Filter by status
            limit: Maximum results
        
        Returns:
            List of model records
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = "SELECT * FROM training_models WHERE status = %s"
            params = [status]
            
            if provider:
                query += " AND provider = %s"
                params.append(provider)
            
            query += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit)
            
            cur.execute(query, params)
            return [ModelRecord(**dict(row)) for row in cur.fetchall()]


if __name__ == "__main__":
    print("=== Training Registry DB Test ===\n")
    
    print("Test 1: Create job record")
    job = TrainingJobRecord(
        job_id="test-id",
        provider="openai",
        provider_job_id="ftjob-abc123",
        base_model="gpt-3.5-turbo",
        training_file_id="file-xyz",
        status="queued"
    )
    print(f"✓ Job record created: {job.provider}/{job.status}\n")
    
    print("Test 2: Create hyperparameters record")
    hyper = HyperparametersRecord(
        hyperparameter_id="hyper-id",
        job_id="test-id",
        n_epochs=5,
        batch_size=8,
        learning_rate_multiplier=1.5
    )
    print(f"✓ Hyperparameters: epochs={hyper.n_epochs}, batch={hyper.batch_size}\n")
    
    print("Test 3: Create metrics record")
    metric = MetricsRecord(
        metric_id="metric-id",
        job_id="test-id",
        step=100,
        training_loss=0.25,
        validation_loss=0.30
    )
    print(f"✓ Metrics: step={metric.step}, loss={metric.training_loss}\n")
    
    print("=== Tests Complete ===")
