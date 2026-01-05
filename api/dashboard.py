#!/usr/bin/env python3
"""
Dashboard API - Extended endpoints for TailAdmin Dashboard

Provides comprehensive API endpoints for the admin dashboard:
- Real-time performance metrics
- Training progress tracking
- Model management
- System health monitoring
- Configuration management
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.data_synthesis.experience_library import ExperienceLibrary
from orchestration.coordinator import SystemCoordinator

# Create router for dashboard endpoints
router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


# ========== Request/Response Models ==========

class DashboardStats(BaseModel):
    """Dashboard statistics response"""
    total_trajectories: int
    training_iterations: int
    best_sharpe: float
    best_win_rate: float
    current_sharpe: float
    current_win_rate: float
    goals_achieved: Dict[str, bool]
    progress: Dict[str, float]


class TrainingProgress(BaseModel):
    """Training progress response"""
    current_iteration: int
    total_iterations: int
    trajectories_collected: int
    target_trajectories: int
    status: str
    estimated_completion: Optional[str] = None


class ModelInfo(BaseModel):
    """Model information"""
    name: str
    type: str
    path: str
    size_mb: float
    last_modified: str
    status: str
    performance: Optional[Dict] = None


class SystemHealth(BaseModel):
    """System health response"""
    status: str
    uptime: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    gpu_available: bool
    gpu_memory_usage: Optional[float] = None
    active_processes: int


class PerformanceMetrics(BaseModel):
    """Performance metrics response"""
    timestamp: str
    sharpe_ratio: float
    win_rate: float
    total_return: float
    max_drawdown: float
    profit_factor: float
    total_trades: int


class TrainingJob(BaseModel):
    """Training job information"""
    job_id: str
    type: str  # sft, rl, continuous
    status: str  # pending, running, completed, failed
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: float
    metrics: Optional[Dict] = None


# ========== Dashboard Endpoints ==========

@router.get("/stats", response_model=DashboardStats)
async def get_dashboard_stats():
    """
    Get overall dashboard statistics.

    Returns current state of the system including:
    - Trajectory counts
    - Training iterations
    - Performance metrics
    - Goal progress
    """
    try:
        # Load training state
        state_file = Path("continuous_training/training_state.json")

        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)
        else:
            state = {
                'total_trajectories': 0,
                'training_iterations': 0,
                'best_sharpe': 0.0,
                'best_win_rate': 0.0,
                'benchmarks': []
            }

        # Get latest performance
        latest = state['benchmarks'][-1] if state['benchmarks'] else {}

        # Calculate goal achievement
        targets = {
            'trajectories': 10000,
            'sharpe': 1.5,
            'win_rate': 0.55
        }

        goals_achieved = {
            'trajectories': state['total_trajectories'] >= targets['trajectories'],
            'sharpe': state['best_sharpe'] >= targets['sharpe'],
            'win_rate': state['best_win_rate'] >= targets['win_rate']
        }

        # Calculate progress percentages
        progress = {
            'trajectories': min(state['total_trajectories'] / targets['trajectories'] * 100, 100),
            'sharpe': min(state['best_sharpe'] / targets['sharpe'] * 100, 100),
            'win_rate': min(state['best_win_rate'] / targets['win_rate'] * 100, 100)
        }

        return DashboardStats(
            total_trajectories=state['total_trajectories'],
            training_iterations=state['training_iterations'],
            best_sharpe=state['best_sharpe'],
            best_win_rate=state['best_win_rate'],
            current_sharpe=latest.get('sharpe_ratio', state['best_sharpe']),
            current_win_rate=latest.get('win_rate', state['best_win_rate']),
            goals_achieved=goals_achieved,
            progress=progress
        )

    except Exception as e:
        logger.error(f"Error getting dashboard stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training/progress", response_model=TrainingProgress)
async def get_training_progress():
    """Get current training progress"""
    try:
        state_file = Path("continuous_training/training_state.json")

        if not state_file.exists():
            return TrainingProgress(
                current_iteration=0,
                total_iterations=0,
                trajectories_collected=0,
                target_trajectories=10000,
                status="not_started"
            )

        with open(state_file, 'r') as f:
            state = json.load(f)

        # Estimate completion time based on current rate
        if state['total_trajectories'] > 0:
            start_time = datetime.fromisoformat(state['start_time'])
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = state['total_trajectories'] / elapsed  # trajectories per second

            remaining = 10000 - state['total_trajectories']
            if rate > 0:
                seconds_remaining = remaining / rate
                completion = datetime.now() + timedelta(seconds=seconds_remaining)
                estimated_completion = completion.isoformat()
            else:
                estimated_completion = None
        else:
            estimated_completion = None

        return TrainingProgress(
            current_iteration=state['training_iterations'],
            total_iterations=20,  # Estimated based on 10k trajectories / 500 per iteration
            trajectories_collected=state['total_trajectories'],
            target_trajectories=10000,
            status="running" if state['total_trajectories'] < 10000 else "completed",
            estimated_completion=estimated_completion
        )

    except Exception as e:
        logger.error(f"Error getting training progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=List[ModelInfo])
async def get_models():
    """List all available models"""
    try:
        models = []

        # Check production models
        prod_dir = Path("models/production")
        if prod_dir.exists():
            for model_dir in prod_dir.iterdir():
                if model_dir.is_dir():
                    size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())

                    models.append(ModelInfo(
                        name=model_dir.name,
                        type="production",
                        path=str(model_dir),
                        size_mb=size / (1024 * 1024),
                        last_modified=datetime.fromtimestamp(model_dir.stat().st_mtime).isoformat(),
                        status="active"
                    ))

        # Check training models
        training_dir = Path("training_runs")
        if training_dir.exists():
            for run_dir in training_dir.glob("*/models/*/final"):
                if run_dir.is_dir():
                    size = sum(f.stat().st_size for f in run_dir.rglob('*') if f.is_file())

                    models.append(ModelInfo(
                        name=run_dir.parent.name,
                        type="training",
                        path=str(run_dir),
                        size_mb=size / (1024 * 1024),
                        last_modified=datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(),
                        status="trained"
                    ))

        return models

    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=SystemHealth)
async def get_system_health():
    """Get system health status"""
    try:
        import psutil
        import torch

        # Get system metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # GPU info
        gpu_available = torch.cuda.is_available()
        gpu_memory_usage = None

        if gpu_available:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_used = torch.cuda.memory_allocated(0)
            gpu_memory_usage = (gpu_memory_used / gpu_memory) * 100

        # Active processes
        python_processes = [p for p in psutil.process_iter(['name']) if 'python' in p.info['name'].lower()]

        # Uptime (approximate based on training state)
        state_file = Path("continuous_training/training_state.json")
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)

            start_time = datetime.fromisoformat(state['start_time'])
            uptime_seconds = (datetime.now() - start_time).total_seconds()
            uptime = str(timedelta(seconds=int(uptime_seconds)))
        else:
            uptime = "N/A"

        return SystemHealth(
            status="healthy" if cpu_usage < 90 and memory.percent < 90 else "warning",
            uptime=uptime,
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            disk_usage=disk.percent,
            gpu_available=gpu_available,
            gpu_memory_usage=gpu_memory_usage,
            active_processes=len(python_processes)
        )

    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/metrics", response_model=List[PerformanceMetrics])
async def get_performance_metrics(limit: int = 10):
    """Get recent performance metrics"""
    try:
        state_file = Path("continuous_training/training_state.json")

        if not state_file.exists():
            return []

        with open(state_file, 'r') as f:
            state = json.load(f)

        benchmarks = state.get('benchmarks', [])[-limit:]

        metrics = []
        for bm in benchmarks:
            metrics.append(PerformanceMetrics(
                timestamp=bm['evaluated_at'],
                sharpe_ratio=bm['sharpe_ratio'],
                win_rate=bm['win_rate'],
                total_return=bm['total_return'],
                max_drawdown=bm['max_drawdown'],
                profit_factor=bm.get('profit_factor', 0.0),
                total_trades=bm.get('total_trades', 0)
            ))

        return metrics

    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training/jobs", response_model=List[TrainingJob])
async def get_training_jobs():
    """Get list of training jobs"""
    try:
        jobs = []

        # Check for running continuous training
        state_file = Path("continuous_training/training_state.json")
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)

            jobs.append(TrainingJob(
                job_id=state['run_id'],
                type="continuous",
                status="running" if state['total_trajectories'] < 10000 else "completed",
                started_at=state['start_time'],
                progress=min(state['total_trajectories'] / 10000 * 100, 100),
                metrics={
                    'trajectories': state['total_trajectories'],
                    'iterations': state['training_iterations'],
                    'best_sharpe': state['best_sharpe']
                }
            ))

        # Check for recent SFT/RL runs
        training_dir = Path("training_runs")
        if training_dir.exists():
            for run_dir in sorted(training_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                if run_dir.is_dir():
                    job_type = "sft" if "sft" in run_dir.name else "rl" if "rl" in run_dir.name else "unknown"

                    jobs.append(TrainingJob(
                        job_id=run_dir.name,
                        type=job_type,
                        status="completed",
                        started_at=datetime.fromtimestamp(run_dir.stat().st_ctime).isoformat(),
                        completed_at=datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(),
                        progress=100.0
                    ))

        return jobs

    except Exception as e:
        logger.error(f"Error getting training jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experience-library/stats")
async def get_experience_library_stats():
    """Get experience library statistics"""
    try:
        lib_path = Path("continuous_training/experience_library.db")

        if not lib_path.exists():
            return {
                'total_trajectories': 0,
                'success_rate': 0.0,
                'avg_reward': 0.0
            }

        lib = ExperienceLibrary(str(lib_path))
        stats = lib.get_statistics()

        return stats

    except Exception as e:
        logger.error(f"Error getting experience library stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/training/start")
async def start_training(
    background_tasks: BackgroundTasks,
    training_type: str = "continuous",
    config: Optional[Dict] = None
):
    """Start a new training job"""
    try:
        if training_type == "continuous":
            # Start continuous training in background
            background_tasks.add_task(run_continuous_training, config or {})

            return {
                "status": "started",
                "type": training_type,
                "message": "Continuous training started in background"
            }

        else:
            raise HTTPException(status_code=400, detail=f"Unknown training type: {training_type}")

    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_continuous_training(config: Dict):
    """Background task to run continuous training"""
    from scripts.continuous_training import ContinuousTrainingPipeline

    pipeline = ContinuousTrainingPipeline(
        target_trajectories=config.get('target_trajectories', 10000),
        target_sharpe=config.get('target_sharpe', 1.5),
        target_win_rate=config.get('target_win_rate', 0.55)
    )

    pipeline.run_continuous_loop(
        collection_batch_size=config.get('batch_size', 1000),
        training_interval=config.get('training_interval', 5),
        evaluation_interval=config.get('evaluation_interval', 10)
    )


# Add router to main FastAPI app
def register_dashboard_routes(app):
    """Register dashboard routes with main app"""
    app.include_router(router)
