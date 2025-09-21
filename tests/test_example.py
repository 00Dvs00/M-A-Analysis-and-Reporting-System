import pytest

def test_sample_data_fixture(sample_data):
    """Test to demonstrate usage of the sample_data fixture."""
    assert "tickers" in sample_data
    assert len(sample_data["tickers"]) == 3
    assert sample_data["prices"] == [150.0, 300.0, 2800.0]

def test_mock_config_fixture(mock_config):
    """Test to demonstrate usage of the mock_config fixture."""
    assert mock_config["api_key"] == "test_api_key"
    assert mock_config["base_url"] == "https://api.example.com"
    assert mock_config["timeout"] == 30