import cProfile
import pstats
from etl_pipeline import run_etl_pipeline
import pandas as pd

def profile_etl():
    """Profile the ETL pipeline with sample data."""
    # Create a small sample dataset for profiling
    sample_data = pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "GOOGL"],
        "cik": [320193, 789019, 1652044]
    })

    # Profile the ETL pipeline
    profiler = cProfile.Profile()
    profiler.enable()
    run_etl_pipeline(tickers_to_process=sample_data, num_filings_per_form=2)
    profiler.disable()

    # Print profiling stats
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats()

if __name__ == "__main__":
    profile_etl()