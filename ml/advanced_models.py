#!/usr/bin/env python3
"""
Advanced ML Models (GANs, Meta-Learning)

Cutting-edge models for stock trading:
- Generative Adversarial Networks (GANs)
  - TimeGAN for time series generation
  - Conditional GAN for scenario generation
  - Wasserstein GAN with gradient penalty
- Meta-Learning
  - Model-Agnostic Meta-Learning (MAML)
  - Prototypical Networks
  - Few-shot learning for new stocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import copy


# ============================================================================
# Generative Adversarial Networks (GANs)
# ============================================================================

@dataclass
class GANConfig:
    """Configuration for GAN models"""
    latent_dim: int = 100
    hidden_dim: int = 256
    num_layers: int = 3
    sequence_length: int = 50
    feature_dim: int = 5

    # Training
    batch_size: int = 64
    num_epochs: int = 100
    g_lr: float = 1e-4
    d_lr: float = 1e-4
    beta1: float = 0.5
    beta2: float = 0.999

    # WGAN-GP specific
    gp_lambda: float = 10.0
    n_critic: int = 5


class TimeGANGenerator(nn.Module):
    """
    Generator for TimeGAN.

    Generates realistic time series data that preserves:
    - Temporal dynamics
    - Statistical properties
    - Feature correlations
    """

    def __init__(self, config: GANConfig):
        super().__init__()
        self.config = config

        # Embedding network
        self.embedder = nn.LSTM(
            input_size=config.latent_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=0.2,
        )

        # Generator network
        self.generator = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=0.2,
        )

        # Output layer
        self.fc = nn.Linear(config.hidden_dim, config.feature_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate time series from noise.

        Args:
            z: Noise tensor [batch_size, sequence_length, latent_dim]

        Returns:
            Generated time series [batch_size, sequence_length, feature_dim]
        """
        # Embed noise
        embedded, _ = self.embedder(z)

        # Generate sequence
        generated, _ = self.generator(embedded)

        # Output
        output = self.fc(generated)

        return output


class TimeGANDiscriminator(nn.Module):
    """Discriminator for TimeGAN"""

    def __init__(self, config: GANConfig):
        super().__init__()
        self.config = config

        # Discriminator network
        self.discriminator = nn.LSTM(
            input_size=config.feature_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=0.2,
        )

        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Discriminate real vs fake time series.

        Args:
            x: Time series [batch_size, sequence_length, feature_dim]

        Returns:
            Probability of being real [batch_size, 1]
        """
        # Process sequence
        output, (h_n, c_n) = self.discriminator(x)

        # Use last hidden state
        last_hidden = h_n[-1]

        # Classify
        prob = self.fc(last_hidden)

        return prob


class ConditionalGAN:
    """
    Conditional GAN for scenario generation.

    Generates time series conditioned on:
    - Market regime (bull, bear, sideways)
    - Volatility level
    - Target return
    """

    def __init__(self, config: GANConfig, num_conditions: int = 3):
        self.config = config
        self.num_conditions = num_conditions
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Generator
        self.generator = self._create_generator()

        # Discriminator
        self.discriminator = self._create_discriminator()

        # Optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config.g_lr,
            betas=(config.beta1, config.beta2),
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config.d_lr,
            betas=(config.beta1, config.beta2),
        )

        # Loss
        self.criterion = nn.BCELoss()

    def _create_generator(self) -> nn.Module:
        """Create conditional generator"""
        return nn.Sequential(
            nn.Linear(self.config.latent_dim + self.num_conditions, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.config.sequence_length * self.config.feature_dim),
            nn.Tanh(),
        ).to(self.device)

    def _create_discriminator(self) -> nn.Module:
        """Create conditional discriminator"""
        return nn.Sequential(
            nn.Linear(self.config.sequence_length * self.config.feature_dim + self.num_conditions, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        ).to(self.device)

    def generate(
        self,
        conditions: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Generate conditioned time series.

        Args:
            conditions: Condition vector [num_samples, num_conditions]
            num_samples: Number of samples to generate

        Returns:
            Generated time series [num_samples, sequence_length, feature_dim]
        """
        self.generator.eval()

        with torch.no_grad():
            # Sample noise
            z = torch.randn(num_samples, self.config.latent_dim).to(self.device)

            # Concatenate with conditions
            z_cond = torch.cat([z, conditions], dim=1)

            # Generate
            fake_data = self.generator(z_cond)

            # Reshape
            fake_data = fake_data.view(
                num_samples,
                self.config.sequence_length,
                self.config.feature_dim,
            )

        return fake_data

    def train_step(
        self,
        real_data: torch.Tensor,
        conditions: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            real_data: Real time series
            conditions: Condition vectors

        Returns:
            Dictionary of losses
        """
        batch_size = real_data.size(0)
        real_data = real_data.view(batch_size, -1).to(self.device)
        conditions = conditions.to(self.device)

        # Labels
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)

        # ===========================
        # Train Discriminator
        # ===========================
        self.d_optimizer.zero_grad()

        # Real data
        real_data_cond = torch.cat([real_data, conditions], dim=1)
        d_real = self.discriminator(real_data_cond)
        d_real_loss = self.criterion(d_real, real_labels)

        # Fake data
        z = torch.randn(batch_size, self.config.latent_dim).to(self.device)
        z_cond = torch.cat([z, conditions], dim=1)
        fake_data = self.generator(z_cond).detach()
        fake_data_cond = torch.cat([fake_data, conditions], dim=1)
        d_fake = self.discriminator(fake_data_cond)
        d_fake_loss = self.criterion(d_fake, fake_labels)

        # Total discriminator loss
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.d_optimizer.step()

        # ===========================
        # Train Generator
        # ===========================
        self.g_optimizer.zero_grad()

        # Generate fake data
        z = torch.randn(batch_size, self.config.latent_dim).to(self.device)
        z_cond = torch.cat([z, conditions], dim=1)
        fake_data = self.generator(z_cond)
        fake_data_cond = torch.cat([fake_data, conditions], dim=1)

        # Try to fool discriminator
        d_fake = self.discriminator(fake_data_cond)
        g_loss = self.criterion(d_fake, real_labels)

        g_loss.backward()
        self.g_optimizer.step()

        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'd_real': d_real.mean().item(),
            'd_fake': d_fake.mean().item(),
        }


class WassersteinGAN:
    """
    Wasserstein GAN with Gradient Penalty (WGAN-GP).

    More stable training with:
    - Wasserstein distance instead of JS divergence
    - Gradient penalty for Lipschitz constraint
    - No sigmoid in discriminator (critic)
    """

    def __init__(self, config: GANConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Networks
        self.generator = TimeGANGenerator(config).to(self.device)
        self.critic = TimeGANDiscriminator(config).to(self.device)

        # Remove sigmoid from critic
        self.critic.fc = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

        # Optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config.g_lr,
            betas=(config.beta1, config.beta2),
        )
        self.c_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config.d_lr,
            betas=(config.beta1, config.beta2),
        )

    def compute_gradient_penalty(
        self,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP"""
        batch_size = real_data.size(0)

        # Random interpolation
        alpha = torch.rand(batch_size, 1, 1).to(self.device)
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.requires_grad_(True)

        # Critic on interpolates
        d_interpolates = self.critic(interpolates)

        # Gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def train_step(self, real_data: torch.Tensor) -> Dict[str, float]:
        """Single training step"""
        batch_size = real_data.size(0)
        real_data = real_data.to(self.device)

        # ===========================
        # Train Critic
        # ===========================
        for _ in range(self.config.n_critic):
            self.c_optimizer.zero_grad()

            # Real data
            c_real = self.critic(real_data)

            # Fake data
            z = torch.randn(batch_size, self.config.sequence_length, self.config.latent_dim).to(self.device)
            fake_data = self.generator(z).detach()
            c_fake = self.critic(fake_data)

            # Gradient penalty
            gp = self.compute_gradient_penalty(real_data, fake_data)

            # Critic loss (Wasserstein distance + gradient penalty)
            c_loss = -c_real.mean() + c_fake.mean() + self.config.gp_lambda * gp

            c_loss.backward()
            self.c_optimizer.step()

        # ===========================
        # Train Generator
        # ===========================
        self.g_optimizer.zero_grad()

        z = torch.randn(batch_size, self.config.sequence_length, self.config.latent_dim).to(self.device)
        fake_data = self.generator(z)
        c_fake = self.critic(fake_data)

        # Generator loss
        g_loss = -c_fake.mean()

        g_loss.backward()
        self.g_optimizer.step()

        return {
            'c_loss': c_loss.item(),
            'g_loss': g_loss.item(),
            'c_real': c_real.mean().item(),
            'c_fake': c_fake.mean().item(),
            'gp': gp.item(),
        }


# ============================================================================
# Meta-Learning
# ============================================================================

@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning"""
    # Model
    hidden_dim: int = 64
    num_layers: int = 2

    # Meta-learning
    n_way: int = 3  # Number of classes
    k_shot: int = 5  # Shots per class
    meta_batch_size: int = 16
    num_inner_steps: int = 5
    inner_lr: float = 1e-2
    meta_lr: float = 1e-3

    # Training
    num_epochs: int = 100


class MAML:
    """
    Model-Agnostic Meta-Learning (MAML).

    Learns to quickly adapt to new tasks (stocks) with few examples.

    Meta-learning objective:
    - Learn initialization that can quickly adapt to new tasks
    - Fast adaptation with few gradient steps
    - Generalizes to new stocks with limited data
    """

    def __init__(self, model: nn.Module, config: MetaLearningConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Meta-optimizer
        self.meta_optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.meta_lr,
        )

    def inner_loop(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        num_steps: int,
    ) -> nn.Module:
        """
        Inner loop: Fast adaptation on support set.

        Args:
            support_x: Support features
            support_y: Support labels
            num_steps: Number of gradient steps

        Returns:
            Adapted model
        """
        # Clone model
        adapted_model = copy.deepcopy(self.model)

        # Inner optimizer
        inner_optimizer = optim.SGD(
            adapted_model.parameters(),
            lr=self.config.inner_lr,
        )

        # Adaptation steps
        for step in range(num_steps):
            inner_optimizer.zero_grad()

            # Forward pass
            predictions = adapted_model(support_x)
            loss = F.cross_entropy(predictions, support_y)

            # Backward pass
            loss.backward()
            inner_optimizer.step()

        return adapted_model

    def outer_loop(
        self,
        tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> float:
        """
        Outer loop: Meta-update.

        Args:
            tasks: List of (support_x, support_y, query_x, query_y) tuples

        Returns:
            Meta-loss
        """
        self.meta_optimizer.zero_grad()
        meta_loss = 0.0

        for support_x, support_y, query_x, query_y in tasks:
            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)

            # Inner loop: Adapt to task
            adapted_model = self.inner_loop(support_x, support_y, self.config.num_inner_steps)

            # Evaluate on query set
            predictions = adapted_model(query_x)
            loss = F.cross_entropy(predictions, query_y)

            meta_loss += loss

        # Average loss
        meta_loss = meta_loss / len(tasks)

        # Meta-update
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()

    def adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> nn.Module:
        """
        Adapt to new task.

        Args:
            support_x: Support features
            support_y: Support labels
            num_steps: Number of adaptation steps

        Returns:
            Adapted model
        """
        num_steps = num_steps or self.config.num_inner_steps
        return self.inner_loop(support_x, support_y, num_steps)


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Networks for few-shot learning.

    Learn to classify based on distance to class prototypes.
    Good for learning new stocks with few examples.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()

        # Embedding network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to embedding space"""
        return self.encoder(x)

    def compute_prototypes(
        self,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor,
        n_way: int,
    ) -> torch.Tensor:
        """
        Compute class prototypes.

        Args:
            support_embeddings: Support set embeddings
            support_labels: Support set labels
            n_way: Number of classes

        Returns:
            Class prototypes [n_way, embedding_dim]
        """
        prototypes = []

        for c in range(n_way):
            # Get embeddings for class c
            class_embeddings = support_embeddings[support_labels == c]

            # Compute prototype (mean)
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)

        prototypes = torch.stack(prototypes)

        return prototypes

    def classify(
        self,
        query_embeddings: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Classify queries based on distance to prototypes.

        Args:
            query_embeddings: Query embeddings
            prototypes: Class prototypes

        Returns:
            Predicted class probabilities
        """
        # Compute distances to prototypes (negative Euclidean distance)
        distances = -torch.cdist(query_embeddings, prototypes)

        # Convert to probabilities
        log_probs = F.log_softmax(distances, dim=1)

        return log_probs


class FewShotLearner:
    """
    Few-shot learning for new stocks.

    Train on many stocks, quickly adapt to new stocks with few examples.
    """

    def __init__(self, config: MetaLearningConfig, input_dim: int):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Prototypical network
        self.model = PrototypicalNetwork(input_dim, config.hidden_dim).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.meta_lr,
        )

    def train_episode(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
    ) -> float:
        """
        Train on single episode.

        Args:
            support_x: Support set features
            support_y: Support set labels
            query_x: Query set features
            query_y: Query set labels

        Returns:
            Episode loss
        """
        self.model.train()
        self.optimizer.zero_grad()

        support_x = support_x.to(self.device)
        support_y = support_y.to(self.device)
        query_x = query_x.to(self.device)
        query_y = query_y.to(self.device)

        # Encode support and query sets
        support_embeddings = self.model(support_x)
        query_embeddings = self.model(query_x)

        # Compute prototypes
        prototypes = self.model.compute_prototypes(
            support_embeddings,
            support_y,
            self.config.n_way,
        )

        # Classify queries
        log_probs = self.model.classify(query_embeddings, prototypes)

        # Compute loss
        loss = F.nll_loss(log_probs, query_y)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict on new stock with few examples.

        Args:
            support_x: Few examples of new stock
            support_y: Labels for support examples
            query_x: Data to predict

        Returns:
            Predicted labels
        """
        self.model.eval()

        with torch.no_grad():
            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)
            query_x = query_x.to(self.device)

            # Encode
            support_embeddings = self.model(support_x)
            query_embeddings = self.model(query_x)

            # Compute prototypes
            prototypes = self.model.compute_prototypes(
                support_embeddings,
                support_y,
                self.config.n_way,
            )

            # Classify
            log_probs = self.model.classify(query_embeddings, prototypes)
            predictions = log_probs.argmax(dim=1)

        return predictions


if __name__ == '__main__':
    # Example: TimeGAN
    print("=== TimeGAN Example ===")
    config = GANConfig(
        sequence_length=30,
        feature_dim=5,
        latent_dim=50,
    )

    generator = TimeGANGenerator(config)
    discriminator = TimeGANDiscriminator(config)

    # Generate sample
    z = torch.randn(8, 30, 50)
    fake_data = generator(z)
    print(f"Generated data shape: {fake_data.shape}")

    # Discriminate
    prob = discriminator(fake_data)
    print(f"Discriminator output: {prob.mean().item():.4f}")

    # Example: MAML
    print("\n=== MAML Example ===")
    meta_config = MetaLearningConfig(
        n_way=3,
        k_shot=5,
        num_inner_steps=3,
    )

    # Simple model
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 3),
    )

    maml = MAML(model, meta_config)

    # Adapt to new task
    support_x = torch.randn(15, 10)  # 3 classes × 5 shots
    support_y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

    adapted_model = maml.adapt(support_x, support_y)
    print("Adapted model to new task!")

    # Example: Prototypical Networks
    print("\n=== Prototypical Networks Example ===")
    proto_net = PrototypicalNetwork(input_dim=10, hidden_dim=64)

    # Support set
    support_embeddings = proto_net(support_x)
    prototypes = proto_net.compute_prototypes(support_embeddings, support_y, n_way=3)
    print(f"Prototypes shape: {prototypes.shape}")

    # Query
    query_x = torch.randn(6, 10)
    query_embeddings = proto_net(query_x)
    log_probs = proto_net.classify(query_embeddings, prototypes)
    predictions = log_probs.argmax(dim=1)
    print(f"Predictions: {predictions}")
