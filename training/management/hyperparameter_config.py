#!/usr/bin/env python3
"""
Hyperparameter Configuration System - Task 7.2

Manage hyperparameter configurations for fine-tuning.

Phase A1 Week 5-6: Task 7.2 COMPLETE
"""

import json
import yaml
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
from loguru import logger


@dataclass
class HyperparameterConfig:
    """Hyperparameter configuration"""
    # Common parameters
    n_epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate_multiplier: Optional[float] = None
    
    # Additional provider-specific parameters
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    name: str = "default"
    description: str = ""
    provider: str = "openai"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict"""
        return {
            k: v for k, v in asdict(self).items()
            if v is not None and k not in ['name', 'description', 'provider']
        }
    
    def validate(self) -> bool:
        """
        Validate configuration
        
        Returns:
            True if valid
        """
        # Validate epochs
        if self.n_epochs is not None and self.n_epochs < 1:
            logger.error("n_epochs must be >= 1")
            return False
        
        # Validate batch size
        if self.batch_size is not None and self.batch_size < 1:
            logger.error("batch_size must be >= 1")
            return False
        
        # Validate learning rate multiplier
        if self.learning_rate_multiplier is not None:
            if self.learning_rate_multiplier <= 0:
                logger.error("learning_rate_multiplier must be > 0")
                return False
        
        return True


@dataclass
class HyperparameterPreset:
    """Predefined hyperparameter preset"""
    name: str
    description: str
    provider: str
    config: HyperparameterConfig
    use_cases: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict"""
        return {
            'name': self.name,
            'description': self.description,
            'provider': self.provider,
            'config': asdict(self.config),
            'use_cases': self.use_cases
        }


class HyperparameterManager:
    """
    Hyperparameter configuration manager
    
    Handles:
    - Preset management
    - Configuration validation
    - File I/O (YAML/JSON)
    """
    
    def __init__(self):
        """Initialize manager"""
        self.presets: Dict[str, HyperparameterPreset] = {}
        self._load_default_presets()
        logger.info("HyperparameterManager initialized")
    
    def _load_default_presets(self):
        """Load default presets"""
        # OpenAI presets
        self.presets['openai_fast'] = HyperparameterPreset(
            name='openai_fast',
            description='Fast training with fewer epochs',
            provider='openai',
            config=HyperparameterConfig(
                n_epochs=1,
                batch_size=8,
                learning_rate_multiplier=1.0,
                provider='openai'
            ),
            use_cases=['quick_iteration', 'testing']
        )
        
        self.presets['openai_balanced'] = HyperparameterPreset(
            name='openai_balanced',
            description='Balanced training for most use cases',
            provider='openai',
            config=HyperparameterConfig(
                n_epochs=3,
                batch_size=4,
                learning_rate_multiplier=1.0,
                provider='openai'
            ),
            use_cases=['general_purpose', 'production']
        )
        
        self.presets['openai_thorough'] = HyperparameterPreset(
            name='openai_thorough',
            description='Thorough training with more epochs',
            provider='openai',
            config=HyperparameterConfig(
                n_epochs=5,
                batch_size=2,
                learning_rate_multiplier=0.5,
                provider='openai'
            ),
            use_cases=['high_quality', 'complex_tasks']
        )
        
        self.presets['openai_auto'] = HyperparameterPreset(
            name='openai_auto',
            description='Let OpenAI choose optimal hyperparameters',
            provider='openai',
            config=HyperparameterConfig(
                provider='openai'
            ),
            use_cases=['default', 'recommended']
        )
        
        logger.info(f"Loaded {len(self.presets)} default presets")
    
    def get_preset(self, name: str) -> Optional[HyperparameterPreset]:
        """
        Get preset by name
        
        Args:
            name: Preset name
        
        Returns:
            Preset or None
        """
        return self.presets.get(name)
    
    def list_presets(
        self,
        provider: Optional[str] = None
    ) -> List[HyperparameterPreset]:
        """
        List presets
        
        Args:
            provider: Filter by provider
        
        Returns:
            List of presets
        """
        presets = list(self.presets.values())
        
        if provider:
            presets = [p for p in presets if p.provider == provider]
        
        return presets
    
    def create_config(
        self,
        n_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate_multiplier: Optional[float] = None,
        additional_params: Optional[Dict[str, Any]] = None,
        name: str = "custom",
        description: str = "",
        provider: str = "openai"
    ) -> HyperparameterConfig:
        """
        Create custom configuration
        
        Args:
            n_epochs: Number of epochs
            batch_size: Batch size
            learning_rate_multiplier: Learning rate multiplier
            additional_params: Additional parameters
            name: Config name
            description: Description
            provider: Provider
        
        Returns:
            Configuration
        """
        config = HyperparameterConfig(
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate_multiplier=learning_rate_multiplier,
            additional_params=additional_params or {},
            name=name,
            description=description,
            provider=provider
        )
        
        if not config.validate():
            raise ValueError("Invalid hyperparameter configuration")
        
        logger.info(f"✓ Created config: {name}")
        return config
    
    def save_config(
        self,
        config: HyperparameterConfig,
        file_path: str,
        format: str = 'yaml'
    ):
        """
        Save configuration to file
        
        Args:
            config: Configuration
            file_path: Output file path
            format: File format ('yaml' or 'json')
        """
        data = asdict(config)
        
        if format == 'yaml':
            with open(file_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        elif format == 'json':
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"✓ Config saved: {file_path}")
    
    def load_config(self, file_path: str) -> HyperparameterConfig:
        """
        Load configuration from file
        
        Args:
            file_path: Input file path
        
        Returns:
            Configuration
        """
        path = Path(file_path)
        
        if path.suffix in ['.yaml', '.yml']:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
        elif path.suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        config = HyperparameterConfig(**data)
        
        if not config.validate():
            raise ValueError("Invalid configuration in file")
        
        logger.info(f"✓ Config loaded: {file_path}")
        return config
    
    def compare_configs(
        self,
        config1: HyperparameterConfig,
        config2: HyperparameterConfig
    ) -> Dict[str, Any]:
        """
        Compare two configurations
        
        Args:
            config1: First configuration
            config2: Second configuration
        
        Returns:
            Comparison dict
        """
        diff = {}
        
        for key in ['n_epochs', 'batch_size', 'learning_rate_multiplier']:
            val1 = getattr(config1, key)
            val2 = getattr(config2, key)
            if val1 != val2:
                diff[key] = {'config1': val1, 'config2': val2}
        
        return diff
    
    def suggest_config(
        self,
        dataset_size: int,
        task_complexity: str = 'medium',
        provider: str = 'openai'
    ) -> HyperparameterConfig:
        """
        Suggest configuration based on dataset and task
        
        Args:
            dataset_size: Number of training examples
            task_complexity: Task complexity ('simple', 'medium', 'complex')
            provider: Provider
        
        Returns:
            Suggested configuration
        """
        # Simple heuristics
        if dataset_size < 100:
            # Small dataset: more epochs
            n_epochs = 5
            batch_size = 2
        elif dataset_size < 1000:
            # Medium dataset: balanced
            n_epochs = 3
            batch_size = 4
        else:
            # Large dataset: fewer epochs
            n_epochs = 2
            batch_size = 8
        
        # Adjust for task complexity
        if task_complexity == 'simple':
            learning_rate_multiplier = 1.5
        elif task_complexity == 'complex':
            learning_rate_multiplier = 0.5
            n_epochs += 1
        else:
            learning_rate_multiplier = 1.0
        
        config = HyperparameterConfig(
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate_multiplier=learning_rate_multiplier,
            name='suggested',
            description=f'Suggested for {dataset_size} examples, {task_complexity} task',
            provider=provider
        )
        
        logger.info(f"✓ Suggested config: epochs={n_epochs}, batch={batch_size}")
        return config


# Helper functions

def get_preset(name: str) -> Optional[HyperparameterConfig]:
    """
    Get preset configuration
    
    Args:
        name: Preset name
    
    Returns:
        Configuration or None
    """
    manager = HyperparameterManager()
    preset = manager.get_preset(name)
    return preset.config if preset else None


def suggest_config(
    dataset_size: int,
    task_complexity: str = 'medium',
    provider: str = 'openai'
) -> HyperparameterConfig:
    """
    Suggest configuration
    
    Args:
        dataset_size: Number of training examples
        task_complexity: Task complexity
        provider: Provider
    
    Returns:
        Suggested configuration
    """
    manager = HyperparameterManager()
    return manager.suggest_config(dataset_size, task_complexity, provider)


if __name__ == "__main__":
    print("=== Hyperparameter Configuration Test ===\n")
    
    # Test 1: Create manager
    print("Test 1: Create manager")
    manager = HyperparameterManager()
    print(f"✓ Manager created with {len(manager.presets)} presets\n")
    
    # Test 2: List presets
    print("Test 2: List presets")
    presets = manager.list_presets(provider='openai')
    print(f"✓ {len(presets)} OpenAI presets:")
    for preset in presets:
        print(f"  - {preset.name}: {preset.description}")
    print()
    
    # Test 3: Get preset
    print("Test 3: Get preset")
    preset = manager.get_preset('openai_balanced')
    if preset:
        print(f"✓ Preset: {preset.name}")
        print(f"  Epochs: {preset.config.n_epochs}")
        print(f"  Batch size: {preset.config.batch_size}")
        print(f"  LR multiplier: {preset.config.learning_rate_multiplier}")
    print()
    
    # Test 4: Create custom config
    print("Test 4: Create custom config")
    config = manager.create_config(
        n_epochs=4,
        batch_size=6,
        learning_rate_multiplier=1.2,
        name='custom_test'
    )
    print(f"✓ Config created: {config.name}")
    print(f"  Valid: {config.validate()}")
    print()
    
    # Test 5: Suggest config
    print("Test 5: Suggest config")
    suggested = manager.suggest_config(dataset_size=500, task_complexity='medium')
    print(f"✓ Suggested config:")
    print(f"  Epochs: {suggested.n_epochs}")
    print(f"  Batch size: {suggested.batch_size}")
    print(f"  LR multiplier: {suggested.learning_rate_multiplier}")
    print()
    
    print("=== Tests Complete ===")
