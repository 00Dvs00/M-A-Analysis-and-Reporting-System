import pytest

@pytest.fixture(scope="module")
def sample_data():
    """Fixture to provide sample data for testing."""
    return {
        "tickers": ["AAPL", "MSFT", "GOOGL"],
        "prices": [150.0, 300.0, 2800.0],
        "volumes": [1000000, 2000000, 1500000]
    }

@pytest.fixture(scope="module")
def mock_config():
    """Fixture to provide a mock configuration for testing."""
    return {
        "api_key": "test_api_key",
        "base_url": "https://api.example.com",
        "timeout": 30
    }