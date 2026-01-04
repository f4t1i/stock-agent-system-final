"""
Configuration Loader - YAML Config Management
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import os


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load YAML configuration file
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Environment variable substitution
    config = _substitute_env_vars(config)
    
    return config


def _substitute_env_vars(config: Any) -> Any:
    """
    Rekursive Substitution von Environment-Variablen
    
    Syntax: ${VAR_NAME} oder ${VAR_NAME:default_value}
    """
    if isinstance(config, dict):
        return {k: _substitute_env_vars(v) for k, v in config.items()}
    
    elif isinstance(config, list):
        return [_substitute_env_vars(item) for item in config]
    
    elif isinstance(config, str):
        if config.startswith('${') and config.endswith('}'):
            var_expr = config[2:-1]
            
            # Check for default value
            if ':' in var_expr:
                var_name, default = var_expr.split(':', 1)
                return os.getenv(var_name, default)
            else:
                return os.getenv(var_expr, config)
        
        return config
    
    else:
        return config


def save_config(config: Dict[str, Any], output_path: str | Path):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        output_path: Output path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
