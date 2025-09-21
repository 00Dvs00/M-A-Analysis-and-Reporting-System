"""Configuration management module for the M&A Analysis System."""
import os
import re
import yaml
import json
from typing import Any, Dict
from dotenv import load_dotenv

def resolve_env_vars(value: str) -> str:
    """Resolve environment variables in a string value.
    Format: ${ENV_VAR:default_value}
    """
    pattern = r'\${([^:}]+)(?::([^}]+))?}'
    
    def _replace(match):
        env_var = match.group(1)
        default = match.group(2)
        return os.getenv(env_var, default) if default else os.getenv(env_var, '')
    
    return re.sub(pattern, _replace, str(value))

def resolve_config_values(config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively resolve environment variables in configuration values."""
    resolved = {}
    for key, value in config.items():
        if isinstance(value, dict):
            resolved[key] = resolve_config_values(value)
        elif isinstance(value, list):
            resolved[key] = [resolve_config_values(item) if isinstance(item, dict) else item for item in value]
        elif isinstance(value, str):
            resolved[key] = resolve_env_vars(value)
        else:
            resolved[key] = value
    return resolved

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and process configuration from a YAML or JSON file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        Dict containing the configuration with resolved values.
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        ValueError: If the file format is not supported.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load environment variables first
    load_dotenv()
    
    # Load the configuration file
    with open(config_path, 'r') as f:
        if config_path.endswith(('.yaml', '.yml')):
            config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            config = json.load(f)
        else:
            raise ValueError("Unsupported config file format. Use YAML or JSON.")
    
    # Resolve environment variables and return
    return resolve_config_values(config)

def get_config_value(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Get a configuration value using a dot-notation path.
    
    Args:
        config: The configuration dictionary.
        path: Dot-notation path (e.g., "database.host").
        default: Default value if path doesn't exist.
        
    Returns:
        The configuration value or default.
    """
    parts = path.split('.')
    current = config
    
    try:
        for part in parts:
            current = current[part]
        return current
    except (KeyError, TypeError):
        return default