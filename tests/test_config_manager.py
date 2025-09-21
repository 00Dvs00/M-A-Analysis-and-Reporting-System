"""Test suite for configuration management system."""
import os
import unittest
import tempfile
import yaml
import json
from src.utils.config_manager import load_config, get_config_value, resolve_env_vars

class TestConfigManager(unittest.TestCase):
    def setUp(self):
        # Sample configuration for testing
        self.config_data = {
            'database': {
                'host': '${DB_HOST:localhost}',
                'port': '${DB_PORT:5432}',
                'name': 'test_db'
            },
            'batch_processing': {
                'max_workers': 4,
                'rate_limit': {
                    'requests_per_minute': 60
                }
            }
        }
    
    def test_load_yaml_config(self):
        """Test loading YAML configuration with environment variables."""
        with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False) as f:
            yaml.dump(self.config_data, f)
            config_path = f.name
        
        try:
            # Set test environment variable
            os.environ['DB_HOST'] = 'testhost.com'
            
            # Load configuration
            config = load_config(config_path)
            
            # Verify values
            self.assertEqual(config['database']['host'], 'testhost.com')
            self.assertEqual(config['database']['port'], '5432')  # Default value
            self.assertEqual(config['database']['name'], 'test_db')
        finally:
            os.unlink(config_path)
            os.environ.pop('DB_HOST', None)
    
    def test_get_config_value(self):
        """Test retrieving values using dot notation."""
        config = self.config_data
        
        # Test existing paths
        self.assertEqual(
            get_config_value(config, 'database.host'),
            '${DB_HOST:localhost}'
        )
        self.assertEqual(
            get_config_value(config, 'batch_processing.max_workers'),
            4
        )
        
        # Test non-existent paths
        self.assertIsNone(get_config_value(config, 'invalid.path'))
        self.assertEqual(
            get_config_value(config, 'invalid.path', default='default'),
            'default'
        )
    
    def test_resolve_env_vars(self):
        """Test environment variable resolution."""
        # Test with environment variable set
        os.environ['TEST_VAR'] = 'test_value'
        self.assertEqual(
            resolve_env_vars('${TEST_VAR}'),
            'test_value'
        )
        
        # Test with default value
        self.assertEqual(
            resolve_env_vars('${NONEXISTENT_VAR:default}'),
            'default'
        )
        
        # Test with no default and no environment variable
        self.assertEqual(
            resolve_env_vars('${NONEXISTENT_VAR}'),
            ''
        )
        
        os.environ.pop('TEST_VAR', None)

if __name__ == '__main__':
    unittest.main()