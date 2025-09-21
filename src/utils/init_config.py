"""Initialization script to generate default configuration files."""
import os
import yaml

DEFAULT_CONFIG = {
    'database': {
        'host': '${DB_HOST:localhost}',  # Environment variable with default
        'port': '${DB_PORT:5432}',
        'name': '${DB_NAME:sec_data}',
        'user': '${DB_USER:sec_user}'
    },
    'analysis': {
        'screens': {
            'market_cap_min': 1000000000,  # $1B minimum
            'revenue_min': 100000000,      # $100M minimum
            'liquidity_min': 0.8           # Current ratio minimum
        },
        'metrics': [
            'ev_to_revenue',
            'ev_to_ebitda',
            'price_to_earnings',
            'price_to_book'
        ],
        'outlier_rules': {
            'zscore_threshold': 2.5,
            'iqr_multiplier': 1.5
        }
    },
    'batch_processing': {
        'max_workers': 4,
        'rate_limit': {
            'requests_per_minute': 60,
            'burst_size': 10
        },
        'retry': {
            'max_attempts': 3,
            'base_delay': 1.0,
            'max_delay': 30.0,
            'jitter': True
        }
    },
    'logging': {
        'level': '${LOG_LEVEL:INFO}',
        'format': 'json',
        'output': ['file', 'console'],
        'file_path': 'batch_processing.log'
    }
}

def initialize_config():
    """Create default configuration files if they don't exist."""
    # Get project root directory (2 levels up from utils)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # Ensure config directory exists
    config_dir = os.path.join(project_root, 'config')
    os.makedirs(config_dir, exist_ok=True)
    
    # Write default YAML config
    yaml_path = os.path.join(config_dir, 'default_config.yaml')
    if not os.path.exists(yaml_path):
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(DEFAULT_CONFIG, f, default_flow_style=False)
            print(f"Created default configuration at: {yaml_path}")

    # Create example .env file
    env_path = os.path.join(project_root, '.env.example')
    if not os.path.exists(env_path):
        with open(env_path, 'w') as f:
            f.write("""# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=sec_data
DB_USER=sec_user
DB_PASSWORD=your_password_here

# Logging Configuration
LOG_LEVEL=INFO

# API Keys and Secrets
FINNHUB_API_KEY=your_api_key_here
SEC_API_KEY=your_api_key_here
""")
            print(f"Created example environment file at: {env_path}")

if __name__ == '__main__':
    initialize_config()