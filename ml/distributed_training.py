#!/usr/bin/env python3
"""
Distributed Training for Large-Scale ML

Enterprise-grade distributed training with:
- Multi-GPU training (DataParallel, DistributedDataParallel)
- Multi-node training
- Mixed precision training (FP16)
- Gradient accumulation
- Distributed data loading
- Model checkpointing and resuming
- Monitoring across processes
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from loguru import logger
import time
from datetime import datetime


@dataclass
class DistributedConfig:
    """Configuration for distributed training"""
    # Multi-GPU settings
    use_distributed: bool = False
    backend: str = 'nccl'  # 'nccl' for GPU, 'gloo' for CPU
    world_size: int = 1  # Total number of processes
    rank: int = 0  # Rank of current process
    local_rank: int = 0  # Local rank on current node

    # Multi-node settings
    master_addr: str = 'localhost'
    master_port: str = '12355'
    num_nodes: int = 1
    gpus_per_node: int = torch.cuda.device_count()

    # Training optimization
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    find_unused_parameters: bool = False

    # Checkpointing
    checkpoint_dir: Path = Path('./checkpoints')
    save_frequency: int = 1000  # steps
    keep_n_checkpoints: int = 3

    # Logging
    log_frequency: int = 100  # steps
    use_tensorboard: bool = True
    tensorboard_dir: Path = Path('./runs')


class DistributedTrainer:
    """
    Distributed training manager.

    Handles:
    - Multi-GPU training (single node)
    - Multi-node training
    - Mixed precision
    - Gradient accumulation
    - Checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        config: DistributedConfig,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Distributed setup
        self.is_distributed = config.use_distributed
        self.is_main_process = config.rank == 0

        # Mixed precision
        self.scaler = GradScaler() if config.use_mixed_precision else None

        # Checkpointing
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self.writer = None
        if self.is_main_process and config.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(config.tensorboard_dir)

        # Setup model
        self._setup_model()

        logger.info(f"Initialized DistributedTrainer (rank={config.rank}/{config.world_size})")

    def _setup_model(self):
        """Setup model for distributed training"""
        self.model = self.model.to(self.device)

        if self.is_distributed:
            # Use DistributedDataParallel
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.config.local_rank],
                output_device=self.config.local_rank,
                find_unused_parameters=self.config.find_unused_parameters,
            )
            logger.info(f"Wrapped model with DistributedDataParallel (rank={self.config.rank})")
        elif torch.cuda.device_count() > 1:
            # Use DataParallel for single-node multi-GPU
            self.model = DataParallel(self.model)
            logger.info(f"Wrapped model with DataParallel ({torch.cuda.device_count()} GPUs)")

    def train(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        num_epochs: int,
        val_loader: Optional[DataLoader] = None,
        scheduler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Train model with distributed support.

        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            num_epochs: Number of epochs
            val_loader: Validation data loader
            scheduler: Learning rate scheduler

        Returns:
            Training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
        }

        global_step = 0
        start_time = time.time()

        for epoch in range(num_epochs):
            # Set epoch for distributed sampler
            if self.is_distributed and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

            # Train epoch
            train_loss, train_metrics = self._train_epoch(
                train_loader,
                optimizer,
                criterion,
                epoch,
                global_step,
            )

            history['train_loss'].append(train_loss)
            history['train_metrics'].append(train_metrics)

            # Update global step
            global_step += len(train_loader)

            # Validation
            if val_loader is not None:
                val_loss, val_metrics = self._validate(val_loader, criterion, epoch)
                history['val_loss'].append(val_loss)
                history['val_metrics'].append(val_metrics)

            # Learning rate scheduling
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss if val_loader else train_loss)
                else:
                    scheduler.step()

            # Checkpointing
            if self.is_main_process and (epoch + 1) % 5 == 0:
                self.save_checkpoint(
                    epoch=epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metrics={'train_loss': train_loss, 'val_loss': val_loss if val_loader else None},
                )

            # Logging
            if self.is_main_process:
                elapsed = time.time() - start_time
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f if val_loader else 'N/A'} | "
                    f"Time: {elapsed:.2f}s"
                )

        return history

    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epoch: int,
        global_step: int,
    ) -> Tuple[float, Dict[str, float]]:
        """Train single epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        optimizer.zero_grad()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Mixed precision forward pass
            if self.config.use_mixed_precision:
                with autocast():
                    output = self.model(data)
                    loss = criterion(output, target)
                    loss = loss / self.config.gradient_accumulation_steps

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    # Optimizer step
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
            else:
                # Standard training
                output = self.model(data)
                loss = criterion(output, target)
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

            # Logging
            step = global_step + batch_idx
            if self.is_main_process and step % self.config.log_frequency == 0:
                if self.writer:
                    self.writer.add_scalar('Loss/train_step', loss.item(), step)

                logger.debug(
                    f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Synchronize loss across processes
        if self.is_distributed:
            avg_loss = self._reduce_value(avg_loss)

        return avg_loss, {}

    def _validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
        epoch: int,
    ) -> Tuple[float, Dict[str, float]]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                if self.config.use_mixed_precision:
                    with autocast():
                        output = self.model(data)
                        loss = criterion(output, target)
                else:
                    output = self.model(data)
                    loss = criterion(output, target)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Synchronize loss across processes
        if self.is_distributed:
            avg_loss = self._reduce_value(avg_loss)

        if self.is_main_process and self.writer:
            self.writer.add_scalar('Loss/val', avg_loss, epoch)

        return avg_loss, {}

    def _reduce_value(self, value: float) -> float:
        """Reduce value across all processes"""
        tensor = torch.tensor(value, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.item() / self.config.world_size

    def save_checkpoint(
        self,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
    ):
        """Save training checkpoint"""
        if not self.is_main_process:
            return

        # Get model state dict (unwrap DDP/DP if needed)
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics or {},
            'config': self.config,
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Save checkpoint
        checkpoint_path = self.config.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Clean old checkpoints
        self._cleanup_checkpoints()

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> int:
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load scaler state
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        epoch = checkpoint.get('epoch', 0)
        logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch})")

        return epoch

    def _cleanup_checkpoints(self):
        """Keep only N most recent checkpoints"""
        checkpoints = sorted(
            self.config.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )

        # Remove old checkpoints
        for checkpoint in checkpoints[:-self.config.keep_n_checkpoints]:
            checkpoint.unlink()
            logger.debug(f"Removed old checkpoint: {checkpoint}")


def setup_distributed(rank: int, world_size: int, backend: str = 'nccl'):
    """
    Setup distributed training.

    Args:
        rank: Rank of current process
        world_size: Total number of processes
        backend: Communication backend ('nccl' or 'gloo')
    """
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')

    # Initialize process group
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

    # Set device
    if backend == 'nccl':
        torch.cuda.set_device(rank)

    logger.info(f"Initialized distributed training (rank={rank}/{world_size}, backend={backend})")


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Cleaned up distributed training")


def create_distributed_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    is_distributed: bool = False,
    num_workers: int = 4,
    shuffle: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Create data loader with distributed sampler.

    Args:
        dataset: PyTorch dataset
        batch_size: Batch size per process
        is_distributed: Whether to use distributed sampler
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        **kwargs: Additional DataLoader arguments

    Returns:
        DataLoader with distributed sampler if needed
    """
    sampler = None
    if is_distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=shuffle,
        )
        shuffle = False  # Sampler handles shuffling

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs,
    )

    return loader


def train_distributed(
    rank: int,
    world_size: int,
    model_fn: Callable[[], nn.Module],
    train_dataset: torch.utils.data.Dataset,
    val_dataset: Optional[torch.utils.data.Dataset],
    config: DistributedConfig,
    num_epochs: int,
    batch_size: int,
    learning_rate: float = 1e-3,
):
    """
    Main distributed training function.

    This function should be spawned for each process.

    Args:
        rank: Rank of current process
        world_size: Total number of processes
        model_fn: Function that creates model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Distributed config
        num_epochs: Number of epochs
        batch_size: Batch size per process
        learning_rate: Learning rate
    """
    # Setup distributed
    setup_distributed(rank, world_size, backend=config.backend)

    # Update config
    config.rank = rank
    config.world_size = world_size
    config.local_rank = rank % config.gpus_per_node

    # Create model
    model = model_fn()

    # Create trainer
    device = torch.device(f'cuda:{config.local_rank}' if torch.cuda.is_available() else 'cpu')
    trainer = DistributedTrainer(model, config, device)

    # Create data loaders
    train_loader = create_distributed_dataloader(
        train_dataset,
        batch_size=batch_size,
        is_distributed=True,
        num_workers=4,
        shuffle=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = create_distributed_dataloader(
            val_dataset,
            batch_size=batch_size,
            is_distributed=True,
            num_workers=4,
            shuffle=False,
        )

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Create criterion
    criterion = nn.CrossEntropyLoss()

    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
    )

    # Train
    history = trainer.train(
        train_loader,
        optimizer,
        criterion,
        num_epochs,
        val_loader,
        scheduler,
    )

    # Cleanup
    cleanup_distributed()

    return history


def launch_distributed_training(
    model_fn: Callable[[], nn.Module],
    train_dataset: torch.utils.data.Dataset,
    val_dataset: Optional[torch.utils.data.Dataset],
    config: DistributedConfig,
    num_epochs: int,
    batch_size: int,
    learning_rate: float = 1e-3,
):
    """
    Launch distributed training across multiple processes.

    Args:
        model_fn: Function that creates model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Distributed config
        num_epochs: Number of epochs
        batch_size: Batch size per process
        learning_rate: Learning rate
    """
    world_size = config.gpus_per_node * config.num_nodes

    logger.info(f"Launching distributed training with {world_size} processes")

    mp.spawn(
        train_distributed,
        args=(world_size, model_fn, train_dataset, val_dataset, config, num_epochs, batch_size, learning_rate),
        nprocs=world_size,
        join=True,
    )


if __name__ == '__main__':
    # Example usage
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset

    # Create dummy dataset
    X = torch.randn(1000, 10)
    y = torch.randint(0, 3, (1000,))
    train_dataset = TensorDataset(X, y)
    val_dataset = TensorDataset(X[:200], y[:200])

    # Model factory
    def create_model():
        return nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    # Config
    config = DistributedConfig(
        use_distributed=torch.cuda.device_count() > 1,
        use_mixed_precision=True,
        gradient_accumulation_steps=2,
    )

    if config.use_distributed:
        # Launch distributed training
        launch_distributed_training(
            create_model,
            train_dataset,
            val_dataset,
            config,
            num_epochs=10,
            batch_size=32,
            learning_rate=1e-3,
        )
    else:
        # Single GPU/CPU training
        model = create_model()
        trainer = DistributedTrainer(model, config)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        history = trainer.train(
            train_loader,
            optimizer,
            criterion,
            num_epochs=10,
            val_loader=val_loader,
        )

        print(f"\nTraining complete!")
        print(f"Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"Final val loss: {history['val_loss'][-1]:.4f}")
