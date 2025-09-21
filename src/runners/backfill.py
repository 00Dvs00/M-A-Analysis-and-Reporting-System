import os
import logging
from dotenv import load_dotenv
from etl_pipeline import run_etl_pipeline
from get_sp500_tickers import get_sp500_info

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

def backfill_missing_data():
    """
    A targeted script to download and process data for tickers listed in 'missing_tickers.txt'.
    """
    tickers_file = "missing_tickers.txt"
    if not os.path.exists(tickers_file):
        logging.info(f"'{tickers_file}' not found. No backfill needed.")
        return

    logging.info(f"Found '{tickers_file}'. Starting backfill process.")
    with open(tickers_file, 'r') as f:
        tickers_to_process_list = [line.strip() for line in f if line.strip()]

    if not tickers_to_process_list:
        logging.info("Ticker file is empty. No backfill needed.")
        return

    # Get the full company info for the tickers in the file
    master_df = get_sp500_info()
    if master_df is None:
        logging.error("Could not fetch master S&P 500 list. Aborting.")
        return
        
    # Filter the master list to only the tickers we need to process
    tickers_to_process_df = master_df[master_df['ticker'].isin(tickers_to_process_list)]

    if tickers_to_process_df.empty:
        logging.warning("No valid S&P 500 tickers from the file to process.")
        return

    logging.info(f"Preparing to run ETL for: {', '.join(tickers_to_process_df['ticker'].tolist())}")

    # Run the main ETL pipeline with our small, targeted DataFrame
    # We'll fetch a decent number of historical filings
    run_etl_pipeline(tickers_to_process=tickers_to_process_df, num_filings_per_form=10)

    logging.info("Backfill process complete.")
    # Clean up the file so we don't run it again accidentally
    os.remove(tickers_file)
    logging.info(f"Removed '{tickers_file}'.")


if __name__ == "__main__":
    print("\n=======================================================")
    print("===         Running Targeted Backfill Script        ===")
    print("=======================================================\n")
    backfill_missing_data()
