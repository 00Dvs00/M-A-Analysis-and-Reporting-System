import unittest
import os
import yaml
import json
import tempfile

class TestConfig(unittest.TestCase):
    def setUp(self):
        # Create a sample configuration
        self.config_data = {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'name': 'test_db',
                'user': 'test_user'
            },
            'analysis': {
                'screens': {
                    'market_cap_min': 1000000000,
                    'revenue_min': 100000000
                }
            }
        }

    def test_yaml_config(self):
        # Test YAML configuration loading
        with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False) as f:
            yaml.dump(self.config_data, f)
            config_path = f.name

        try:
            from src.runners.main import load_config
            loaded_config = load_config(config_path)
            self.assertEqual(loaded_config, self.config_data)
        finally:
            os.unlink(config_path)

    def test_json_config(self):
        # Test JSON configuration loading
        with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as f:
            json.dump(self.config_data, f)
            config_path = f.name

        try:
            from src.runners.main import load_config
            loaded_config = load_config(config_path)
            self.assertEqual(loaded_config, self.config_data)
        finally:
            os.unlink(config_path)

    def test_invalid_config_path(self):
        # Test loading non-existent configuration file
        with self.assertRaises(FileNotFoundError):
            from src.runners.main import load_config
            load_config('nonexistent.yaml')

    def test_invalid_config_format(self):
        # Test loading configuration with unsupported format
        with tempfile.NamedTemporaryFile(suffix='.txt', mode='w', delete=False) as f:
            f.write('invalid config')
            config_path = f.name

        try:
            from src.runners.main import load_config
            with self.assertRaises(ValueError) as context:
                load_config(config_path)
            self.assertIn('Unsupported config file format', str(context.exception))
        finally:
            os.unlink(config_path)

if __name__ == '__main__':
    unittest.main()