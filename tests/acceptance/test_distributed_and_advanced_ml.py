#!/usr/bin/env python3
"""
Acceptance tests for Distributed Training and Advanced ML.

Tests:
1. Distributed training setup
2. TimeGAN generation
3. Conditional GAN
4. MAML few-shot adaptation
5. Prototypical networks
"""

try:
    import pytest
except ImportError:
    pytest = None

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


def test_distributed_trainer_single_gpu():
    """Test distributed trainer on single GPU/CPU"""
    from ml.distributed_training import DistributedTrainer, DistributedConfig

    # Config
    config = DistributedConfig(
        use_distributed=False,
        use_mixed_precision=False,  # Disable for CPU testing
        gradient_accumulation_steps=2,
    )

    # Simple model
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 3),
    )

    # Trainer
    trainer = DistributedTrainer(model, config)

    # Data
    X = torch.randn(100, 10)
    y = torch.randint(0, 3, (100,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Train
    history = trainer.train(
        loader,
        optimizer,
        criterion,
        num_epochs=2,
    )

    # Verify
    assert len(history['train_loss']) == 2
    assert history['train_loss'][-1] < history['train_loss'][0], "Loss should decrease"

    print("✅ DistributedTrainer single GPU test passed")


def test_timegan_generation():
    """Test TimeGAN generation"""
    from ml.advanced_models import TimeGANGenerator, TimeGANDiscriminator, GANConfig

    # Config
    config = GANConfig(
        latent_dim=50,
        hidden_dim=64,
        num_layers=2,
        sequence_length=20,
        feature_dim=5,
    )

    # Models
    generator = TimeGANGenerator(config)
    discriminator = TimeGANDiscriminator(config)

    # Generate
    batch_size = 8
    z = torch.randn(batch_size, config.sequence_length, config.latent_dim)
    fake_data = generator(z)

    # Verify shape
    assert fake_data.shape == (batch_size, config.sequence_length, config.feature_dim)

    # Discriminate
    prob = discriminator(fake_data)

    # Verify shape
    assert prob.shape == (batch_size, 1)
    assert (prob >= 0).all() and (prob <= 1).all(), "Probabilities should be in [0, 1]"

    print("✅ TimeGAN generation test passed")


def test_conditional_gan():
    """Test Conditional GAN"""
    from ml.advanced_models import ConditionalGAN, GANConfig

    # Config
    config = GANConfig(
        latent_dim=50,
        sequence_length=20,
        feature_dim=5,
    )

    # Model
    num_conditions = 3  # Bull, bear, sideways
    cgan = ConditionalGAN(config, num_conditions=num_conditions)

    # Generate conditioned samples
    conditions = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # Bull, Bear
    fake_data = cgan.generate(conditions, num_samples=2)

    # Verify shape
    assert fake_data.shape == (2, config.sequence_length, config.feature_dim)

    # Train step
    real_data = torch.randn(8, config.sequence_length, config.feature_dim)
    conditions = torch.rand(8, num_conditions)  # Random conditions

    losses = cgan.train_step(real_data, conditions)

    # Verify losses
    assert 'd_loss' in losses
    assert 'g_loss' in losses
    assert losses['d_loss'] >= 0
    assert losses['g_loss'] >= 0

    print("✅ Conditional GAN test passed")


def test_wgan_gp():
    """Test Wasserstein GAN with Gradient Penalty"""
    from ml.advanced_models import WassersteinGAN, GANConfig

    # Config
    config = GANConfig(
        latent_dim=50,
        hidden_dim=64,
        num_layers=2,
        sequence_length=20,
        feature_dim=5,
        gp_lambda=10.0,
        n_critic=2,  # Reduced for faster testing
    )

    # Model
    wgan = WassersteinGAN(config)

    # Train step
    real_data = torch.randn(8, config.sequence_length, config.feature_dim)
    losses = wgan.train_step(real_data)

    # Verify losses
    assert 'c_loss' in losses
    assert 'g_loss' in losses
    assert 'gp' in losses
    assert losses['gp'] >= 0, "Gradient penalty should be non-negative"

    print("✅ WGAN-GP test passed")


def test_maml_adaptation():
    """Test MAML few-shot adaptation"""
    from ml.advanced_models import MAML, MetaLearningConfig

    # Config
    config = MetaLearningConfig(
        n_way=3,
        k_shot=5,
        num_inner_steps=3,
        inner_lr=1e-2,
        meta_lr=1e-3,
    )

    # Simple model
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 3),
    )

    # MAML
    maml = MAML(model, config)

    # Create task
    support_x = torch.randn(15, 10)  # 3 classes × 5 shots
    support_y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

    # Adapt
    adapted_model = maml.adapt(support_x, support_y, num_steps=3)

    # Verify adaptation
    with torch.no_grad():
        predictions = adapted_model(support_x)
        assert predictions.shape == (15, 3)

    # Meta-training step
    query_x = torch.randn(9, 10)
    query_y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])

    tasks = [(support_x, support_y, query_x, query_y)]
    meta_loss = maml.outer_loop(tasks)

    assert meta_loss >= 0

    print("✅ MAML adaptation test passed")


def test_prototypical_networks():
    """Test Prototypical Networks"""
    from ml.advanced_models import PrototypicalNetwork

    # Model
    input_dim = 10
    hidden_dim = 32
    proto_net = PrototypicalNetwork(input_dim, hidden_dim)

    # Support set
    support_x = torch.randn(15, input_dim)  # 3 classes × 5 shots
    support_y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

    # Encode
    support_embeddings = proto_net(support_x)
    assert support_embeddings.shape == (15, hidden_dim)

    # Compute prototypes
    prototypes = proto_net.compute_prototypes(support_embeddings, support_y, n_way=3)
    assert prototypes.shape == (3, hidden_dim)

    # Query
    query_x = torch.randn(6, input_dim)
    query_embeddings = proto_net(query_x)

    # Classify
    log_probs = proto_net.classify(query_embeddings, prototypes)
    predictions = log_probs.argmax(dim=1)

    assert predictions.shape == (6,)
    assert (predictions >= 0).all() and (predictions < 3).all()

    print("✅ Prototypical Networks test passed")


def test_few_shot_learner():
    """Test Few-Shot Learner"""
    from ml.advanced_models import FewShotLearner, MetaLearningConfig

    # Config
    config = MetaLearningConfig(
        n_way=3,
        k_shot=5,
        hidden_dim=32,
        meta_lr=1e-3,
    )

    # Learner
    input_dim = 10
    learner = FewShotLearner(config, input_dim)

    # Episode
    support_x = torch.randn(15, input_dim)
    support_y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    query_x = torch.randn(9, input_dim)
    query_y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])

    # Train episode
    loss = learner.train_episode(support_x, support_y, query_x, query_y)
    assert loss >= 0

    # Predict
    new_query_x = torch.randn(6, input_dim)
    predictions = learner.predict(support_x, support_y, new_query_x)

    assert predictions.shape == (6,)
    assert (predictions >= 0).all() and (predictions < 3).all()

    print("✅ Few-Shot Learner test passed")


def test_distributed_dataloader():
    """Test distributed dataloader creation"""
    from ml.distributed_training import create_distributed_dataloader

    # Dataset
    X = torch.randn(100, 10)
    y = torch.randint(0, 3, (100,))
    dataset = TensorDataset(X, y)

    # Create loader (non-distributed)
    loader = create_distributed_dataloader(
        dataset,
        batch_size=16,
        is_distributed=False,
        num_workers=0,  # Disable for testing
        shuffle=True,
    )

    # Verify
    assert len(loader) > 0

    # Get batch
    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape[0] <= 16
    assert batch_y.shape[0] <= 16

    print("✅ Distributed DataLoader test passed")


if __name__ == '__main__':
    print("Running Distributed Training and Advanced ML Tests...\n")

    test_distributed_trainer_single_gpu()
    test_timegan_generation()
    test_conditional_gan()
    test_wgan_gp()
    test_maml_adaptation()
    test_prototypical_networks()
    test_few_shot_learner()
    test_distributed_dataloader()

    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)
