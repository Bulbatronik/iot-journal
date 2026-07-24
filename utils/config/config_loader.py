"""
Configuration module for loading and validating model and experiment settings.
"""
import os
import yaml
import logging
from typing import Dict, Any, Optional


logger = logging.getLogger(__name__)


def load_config(model_name: Optional[str] = None, 
               base_config_name: str = 'base_config') -> Dict[str, Any]:
    # Load base configuration
    base_config_path = _get_config_path(base_config_name)
    config = _load_yaml_config(base_config_path)
    
    # If model name provided, load and merge model configuration
    if model_name:
        model_config_path = _get_config_path(model_name)
        try:
            model_config = _load_yaml_config(model_config_path)
            
            # Add model config under its own namespace
            config[model_name] = model_config
            
            logger.debug(f"Loaded configuration for model {model_name}")
        except FileNotFoundError:
            logger.warning(f"Model configuration not found for {model_name}")
    
    return config


def _get_config_path(config_name: str, config_dir: str = 'configs') -> str:
    # If config_name already contains .yaml extension, use it directly
    if config_name.endswith('.yaml'):
        config_path = os.path.join(config_dir, config_name)
    else:
        # Check if it's a model name
        model_path = os.path.join(config_dir, 'models', f"{config_name}.yaml")
        if os.path.exists(model_path):
            config_path = model_path
        else:
            # Assume it's a base config
            config_path = os.path.join(config_dir, f"{config_name}.yaml")
    
    # Convert to absolute path if needed
    if not os.path.isabs(config_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(base_dir, config_path)
        
    return config_path


def _load_yaml_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        return config


def validate_config(config: Dict[str, Any], model_name: Optional[str] = None) -> bool:
    # Validate base configuration
    required_base_fields = ['dataset']
    for field in required_base_fields:
        if field not in config:
            raise ValueError(f"Missing required field in base configuration: {field}")
    
    # Validate model configuration if model name provided
    if model_name and model_name in config:
        model_config = config[model_name]
        required_model_fields = ['architecture', 'optimizer']
        for field in required_model_fields:
            if field not in model_config:
                raise ValueError(f"Missing required field in {model_name} configuration: {field}")
    
    return True