"""
M&A Analysis and Reporting System - Batch Runner
=================================================

This script automates the execution of the COMPS analysis for a predefined
list of companies, covering a wide range of industries.

Usage:
    python run_batch.py --config <path_to_config>
"""

import logging
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
import yaml
import argparse
from dotenv import load_dotenv
import sys

# --- Local Imports ---
from src.core.comps_analysis import generate_comps_report, get_finnhub_client
from src.core.database_schema import DATABASE_URL

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("batch_analysis.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables from .env file
load_dotenv()

# Adjust sys.path to include project root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def load_config(config_path):
    """Load configuration from a YAML or JSON file."""
    with open(config_path, 'r') as file:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(file)
        elif config_path.endswith('.json'):
            import json
            return json.load(file)
        else:
            raise ValueError("Unsupported config file format. Use YAML or JSON.")

def load_industry_tickers(config):
    """Load industry tickers dynamically from the configuration file."""
    if 'industry_tickers' in config:
        return config['industry_tickers']
    else:
        logging.warning("No industry tickers found in the configuration. Using default tickers.")
        return {
            "Information Technology": "MSFT",
            "Health Care": "JNJ",
            "Consumer Discretionary": "AMZN",
            "Communication Services": "GOOGL",
            "Financials": "JPM",
            "Industrials": "BA",
            "Consumer Staples": "PG",
            "Energy": "XOM",
            "Utilities": "NEE",
            "Real Estate": "AMT",
            "Materials": "LIN"
        }

def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Run batch processing for M&A Analysis.")
    parser.add_argument('--config', type=str, 
                       default='config/default_config.yaml',
                       help="Path to the configuration file (YAML/JSON).")
    args = parser.parse_args()

    # Load configuration
    try:
        from config.config_manager import load_config
        config = load_config(args.config)
        industry_tickers = load_industry_tickers(config)
        logging.info(f"Loaded industry tickers: {industry_tickers}")
        print("Configuration loaded successfully.")
        print(f"Processing with settings:")
        print(f"- Database: {config['database']['name']}")
        print(f"- Market Cap Min: ${config['analysis']['screens']['market_cap_min']:,}")
        print(f"- Revenue Min: ${config['analysis']['screens']['revenue_min']:,}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1

    # --- Batch Analysis ---
    try:
        run_batch_analysis(industry_tickers, config)
    except Exception as e:
        logging.error(f"Error during batch analysis: {e}", exc_info=True)
        return 1

    return 0

def run_batch_analysis(industry_tickers, config):
    """
    Executes the COMPS analysis for all tickers in the predefined list.
    """
    start_time = datetime.now()
    logging.info("=== STARTING BATCH ANALYSIS RUN ===")
    
    # --- Database and Finnhub Setup ---
    try:
        engine = create_engine(DATABASE_URL)
        Session = sessionmaker(bind=engine)
        db_session = Session()
        finnhub_client = get_finnhub_client()
        
        if not finnhub_client:
            logging.error("Finnhub client could not be initialized. Aborting batch run.")
            return
            
    except Exception as e:
        logging.error(f"Failed to set up database or services: {e}")
        return

    successful_runs = []
    failed_runs = []

    for industry, ticker in industry_tickers.items():
        try:
            logging.info(f"--- Processing {industry}: {ticker} ---")
            report_filename, warnings = generate_comps_report(ticker, db_session, finnhub_client)
            
            if report_filename:
                logging.info(f"Successfully generated report for {ticker}: {report_filename}")
                if warnings:
                    logging.warning(f"Analysis for {ticker} completed with warnings: {warnings}")
                successful_runs.append(ticker)
            else:
                logging.error(f"Analysis failed for {ticker}, no report generated.")
                failed_runs.append(ticker)
                
        except Exception as e:
            logging.error(f"An unexpected error occurred while processing {ticker}: {e}", exc_info=True)
            failed_runs.append(ticker)
        
        logging.info("-" * 50)

    # --- Clean Up ---
    db_session.close()
    
    # --- Summary ---
    end_time = datetime.now()
    duration = end_time - start_time
    
    logging.info("=== BATCH ANALYSIS COMPLETE ===")
    logging.info(f"Total Duration: {duration}")
    logging.info(f"Successful runs ({len(successful_runs)}): {', '.join(successful_runs)}")
    logging.info(f"Failed runs ({len(failed_runs)}): {', '.join(failed_runs) if failed_runs else 'None'}")

if __name__ == "__main__":
    exit(main())
