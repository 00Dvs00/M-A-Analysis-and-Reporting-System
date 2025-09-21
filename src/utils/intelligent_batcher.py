class IntelligentBatcher:
    def __init__(self, batch_size):
        """
        Initialize the IntelligentBatcher with a specific batch size.

        :param batch_size: Number of items per batch
        """
        self.batch_size = batch_size

    def create_batches(self, items):
        """
        Create batches from a list of items.

        :param items: List of items to be batched
        :return: List of batches, where each batch is a list of items
        """
        if not items:
            return []

        batches = [
            items[i:i + self.batch_size] for i in range(0, len(items), self.batch_size)
        ]
        return batches


import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import logging
import time
import random
from functools import wraps
import uuid

# --- Import our project's functions ---
# We need the function to get the S&P 500 list
from src.data.get_sp500_tickers import get_sp500_info
# We need the main ETL function to run the process
from src.core.etl_pipeline import run_etl_pipeline

# --- Basic Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("batch_processing.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("BatchProcessor")

# Generate a unique run ID for each batch run
RUN_ID = str(uuid.uuid4())

# --- Runtime-configurable retry and rate-limit defaults (can be overridden via env or tests)
RETRY_TRIES = int(os.getenv("BATCH_RETRY_TRIES", "5"))
RETRY_DELAY = float(os.getenv("BATCH_RETRY_DELAY", "2"))
RETRY_BACKOFF = float(os.getenv("BATCH_RETRY_BACKOFF", "2"))
RETRY_JITTER = os.getenv("BATCH_RETRY_JITTER", "true").lower() in ("1", "true", "yes")
RATE_LIMIT_PER_MIN = int(os.getenv("BATCH_RATE_LIMIT_PER_MIN", "1000"))


# --- Retry Decorator ---
def retry(exceptions, tries=None, delay=None, backoff=None, jitter=None):
    """Retry decorator where parameters may be ints/floats or callables returning the value.
    When callables are passed (e.g. lambda: RETRY_TRIES) the value will be read at call time,
    allowing tests to override module-level settings at runtime.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            local_tries = tries() if callable(tries) else (tries if tries is not None else RETRY_TRIES)
            local_delay = delay() if callable(delay) else (delay if delay is not None else RETRY_DELAY)
            local_backoff = backoff() if callable(backoff) else (backoff if backoff is not None else RETRY_BACKOFF)
            local_jitter = jitter() if callable(jitter) else (jitter if jitter is not None else RETRY_JITTER)

            attempt = 0
            while attempt < local_tries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt == local_tries:
                        logging.error(f"Max retries reached for {func.__name__}: {e}")
                        raise
                    sleep_time = local_delay * (local_backoff ** (attempt - 1))
                    if local_jitter:
                        sleep_time += random.uniform(0, 1)
                    logging.warning(f"Retrying {func.__name__} in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
        return wrapper
    return decorator


# --- Checkpointing ---
CHECKPOINT_FILE = "batch_checkpoint.txt"

def save_checkpoint(ticker):
    with open(CHECKPOINT_FILE, "a") as f:
        f.write(f"{ticker}\n")

def load_checkpoints():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return set(line.strip() for line in f)
    return set()


# --- Simple rate limiter (token bucket) ---
class RateLimiter:
    def __init__(self, rate_per_min=RATE_LIMIT_PER_MIN):
        self.rate_per_min = rate_per_min
        self.tokens = rate_per_min
        self.updated_at = time.monotonic()

    def allow(self):
        now = time.monotonic()
        elapsed = now - self.updated_at
        # Refill tokens proportional to elapsed time
        refill = (elapsed / 60.0) * self.rate_per_min
        if refill >= 1:
            self.tokens = min(self.rate_per_min, self.tokens + refill)
            self.updated_at = now

        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


_GLOBAL_RATE_LIMITER = RateLimiter()


def find_missing_companies():
    """
    Connects to the database, compares the companies present against a master
    list of S&P 500 companies, and returns a DataFrame of the missing ones.
    """
    # --- Step 1: Connect to the Database ---
    load_dotenv()
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    DB_NAME = os.getenv("DB_NAME")
    
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
        logging.error("Database credentials not found in .env file. Aborting.")
        return None

    DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    try:
        engine = create_engine(DATABASE_URL)
        logging.info("Successfully connected to the database.")
    except Exception as e:
        logging.error(f"Failed to connect to the database: {e}")
        return None

    # --- Step 2: Get the Master List of S&P 500 Companies ---
    master_df = get_sp500_info()
    if master_df is None:
        logging.error("Could not fetch the master S&P 500 list. Aborting.")
        return None
    
    # --- Step 3: Get the List of Companies Already in Our Database ---
    try:
        db_df = pd.read_sql("SELECT ticker, cik FROM companies", engine)
        logging.info(f"Found {len(db_df)} companies already in the database.")
    except Exception as e:
        logging.error(f"Failed to query companies from the database: {e}")
        # If the table doesn't exist or is empty, create an empty DataFrame
        db_df = pd.DataFrame(columns=['ticker', 'cik'])
        
    # --- Step 4: Compare and Find What's Missing ---
    # We identify companies as missing if their CIK is not in our database.
    # This correctly handles cases like GOOG/GOOGL.
    if not db_df.empty:
        missing_df = master_df[~master_df['cik'].isin(db_df['cik'])]
    else:
        # If our database is empty, then all companies are "missing"
        missing_df = master_df

    return missing_df


# --- Enhanced ETL Runner ---
@retry((Exception,), tries=lambda: RETRY_TRIES, delay=lambda: RETRY_DELAY, backoff=lambda: RETRY_BACKOFF, jitter=lambda: RETRY_JITTER)
def process_ticker(row):
    ticker = row['ticker']
    # Respect global rate limits before making requests
    waited = 0.0
    while not _GLOBAL_RATE_LIMITER.allow():
        # Small sleep to avoid busy loop; tests patch time.sleep to be no-op
        time.sleep(0.01)
        waited += 0.01
        # Add small jitter to avoid thundering herd
        if RETRY_JITTER:
            time.sleep(random.uniform(0, 0.01))
    if waited > 0:
        logging.info(f"Waited {waited:.2f}s due to rate limiting before processing {ticker}")
    logging.info(f"Processing ticker: {ticker}")
    run_etl_pipeline(tickers_to_process=pd.DataFrame([row]), num_filings_per_form=4)
    save_checkpoint(ticker)
    completed_tickers.add(ticker)
    logging.info(f"Successfully processed ticker: {ticker}")

    logger.info({
        "event": "success",
        "run_id": RUN_ID,
        "ticker": ticker
    })

def run_etl_with_retries(tickers_to_process_df):
    for _, row in tickers_to_process_df.iterrows():
        ticker = row['ticker']
        if ticker in completed_tickers:
            logging.info(f"Skipping already processed ticker: {ticker}")
            continue
        try:
            # Pass `row` explicitly to the processing logic
            process_ticker(row)
        except Exception as e:
            logging.error(f"Failed to process ticker {ticker}: {e}")
            logger.error({
                "event": "error",
                "run_id": RUN_ID,
                "ticker": ticker,
                "error": str(e)
            })
            metrics["failure_count"] += 1
            raise

# Initialize metrics
metrics = {
    "success_count": 0,
    "failure_count": 0,
    "total_latency": 0.0
}

def log_metrics():
    """Logs the current metrics to the logger."""
    total_requests = metrics["success_count"] + metrics["failure_count"]
    avg_latency = metrics["total_latency"] / total_requests if total_requests > 0 else 0.0
    logger.info({
        "event": "metrics_summary",
        "run_id": RUN_ID,
        "success_count": metrics["success_count"],
        "failure_count": metrics["failure_count"],
        "average_latency": avg_latency
    })

def process_ticker(row):
    ticker = row['ticker']
    start_time = time.time()
    logger.info({
        "event": "start_processing",
        "run_id": RUN_ID,
        "ticker": ticker
    })
    try:
        # Respect global rate limits before making requests
        waited = 0.0
        while not _GLOBAL_RATE_LIMITER.allow():
            # Small sleep to avoid busy loop; tests patch time.sleep to be no-op
            waited += 0.01
            # Add small jitter to avoid thundering herd
            if RETRY_JITTER:
                time.sleep(random.uniform(0, 0.01))
        if waited > 0:
            logging.info(f"Waited {waited:.2f}s due to rate limiting before processing {ticker}")
        logging.info(f"Processing ticker: {ticker}")
        run_etl_pipeline(tickers_to_process=pd.DataFrame([row]), num_filings_per_form=4)
        save_checkpoint(ticker)
        completed_tickers.add(ticker)
        logging.info(f"Successfully processed ticker: {ticker}")

        metrics["success_count"] += 1
        logger.info({
            "event": "success",
            "run_id": RUN_ID,
            "ticker": ticker
        })
    except Exception as e:
        metrics["failure_count"] += 1
        logger.error({
            "event": "error",
            "run_id": RUN_ID,
            "ticker": ticker,
            "error": str(e)
        })
        raise
    finally:
        latency = time.time() - start_time
        metrics["total_latency"] += latency
        log_metrics()


if __name__ == "__main__":
    completed_tickers = load_checkpoints()
    logging.info(f"Loaded {len(completed_tickers)} completed tickers from checkpoint.")

    print("\n=======================================================")
    print("=== Running Intelligent Batcher to find and process ===")
    print("===          missing S&P 500 companies            ===")
    print("=======================================================\n")
    
    # --- New: Check for a specific file of tickers to process ---
    tickers_file = "missing_tickers.txt"
    if os.path.exists(tickers_file):
        logging.info(f"Found '{tickers_file}'. Processing only the tickers listed in this file.")
        with open(tickers_file, 'r') as f:
            tickers_to_process_list = [line.strip() for line in f if line.strip()]
        
        # Get the full company info for the tickers in the file
        master_df = get_sp500_info()
        
        # Replace dots with dashes for SEC downloader compatibility
        tickers_to_process_list_fixed = [t.replace('.', '-') for t in tickers_to_process_list]
        master_df['ticker_fixed'] = master_df['ticker'].str.replace('.', '-')
        
        tickers_to_process_df = master_df[master_df['ticker_fixed'].isin(tickers_to_process_list_fixed)]
        
        if not tickers_to_process_df.empty:
            # Use the original tickers for logging but the fixed ones for processing
            logging.info(f"Found companies to process: {', '.join(tickers_to_process_df['ticker'].tolist())}")
            run_etl_pipeline(tickers_to_process=tickers_to_process_df, num_filings_per_form=4)
            logging.info(f"ETL run for {len(tickers_to_process_df)} companies from '{tickers_file}' is complete.")
        else:
            logging.warning(f"'{tickers_file}' was found, but it contained no valid S&P 500 tickers to process.")
            
    else:
        logging.info("No 'missing_tickers.txt' file found. Running standard check for all missing S&P 500 companies.")
        missing_companies_df = find_missing_companies()
    
        # --- Step 5: Execute the ETL for the Missing Companies ---
        if missing_companies_df is not None:
            if missing_companies_df.empty:
                logging.info("SUCCESS: The database is already up-to-date with all S&P 500 companies.")
            else:
                num_missing = len(missing_companies_df)
                logging.warning(f"Found {num_missing} missing companies. Preparing a targeted ETL run.")
                
                # Display the first few missing tickers for the user's information
                print("-------------------------------------------------------")
                print(f"Sample of missing tickers: {', '.join(missing_companies_df['ticker'].head().tolist())}")
                print("-------------------------------------------------------")

                # Run the ETL pipeline with only the DataFrame of missing companies
                # This is where the magic happens!
                run_etl_pipeline(tickers_to_process=missing_companies_df, num_filings_per_form=4)

                logging.info(f"Intelligent batch run for {num_missing} companies is complete.")