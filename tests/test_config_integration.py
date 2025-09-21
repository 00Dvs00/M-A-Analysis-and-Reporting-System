import os
import tempfile
import yaml
import json
import sys
from unittest.mock import patch
from src.runners.main import load_config

# Mock sys.argv to prevent argparse from running in imported scripts
with patch.object(sys, 'argv', ['test']):
    from src.runners.run_batch import load_config

def test_load_yaml_config():
    """Test loading a YAML configuration file."""
    config_data = {"key": "value"}
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w", encoding="utf-8") as temp_file:
        yaml.dump(config_data, temp_file)
        temp_file.close()
        loaded_config = load_config(temp_file.name)
        assert loaded_config == config_data
        os.unlink(temp_file.name)

def test_load_json_config():
    """Test loading a JSON configuration file."""
    config_data = {"key": "value"}
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8") as temp_file:
        json.dump(config_data, temp_file)
        temp_file.close()
        loaded_config = load_config(temp_file.name)
        assert loaded_config == config_data
        os.unlink(temp_file.name)

def test_invalid_config_format():
    """Test loading an unsupported configuration file format."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
        temp_file.write(b"unsupported format")
        temp_file.close()
        try:
            load_config(temp_file.name)
        except ValueError as e:
            assert str(e) == "Unsupported config file format. Use YAML or JSON."
        finally:
            os.unlink(temp_file.name)