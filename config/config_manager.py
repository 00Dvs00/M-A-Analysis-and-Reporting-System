"""Configuration management for the M&A Analysis System."""
import os
import yaml
import json
import logging
from pathlib import Path

def load_config(config_path):
    """Load configuration from a YAML or JSON file.
    
    Args:
        config_path (str): Path to the configuration file.
    
    Returns:
        dict: Configuration data.
    
    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        ValueError: If the file format is not supported (must be YAML or JSON).
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as file:
        if config_path.endswith(('.yaml', '.yml')):
            return yaml.safe_load(file)
        elif config_path.endswith('.json'):
            return json.load(file)
        else:
            raise ValueError("Unsupported config file format. Use YAML or JSON.")