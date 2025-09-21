#!/usr/bin/env python3
"""
M&A Analysis and Reporting System - Single Ticker Runner
========================================================

This script runs the COMPS analysis for a single ticker, specified via command line argument.

Usage:
    python run_single_ticker.py --ticker <ticker_symbol>
"""

import logging
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# --- Local Imports ---
from src.core.comps_analysis import generate_comps_report, get_finnhub_client
from src.core.database_schema import DATABASE_URL

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("single_ticker_analysis.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables from .env file
load_dotenv()

def run_single_ticker_analysis(ticker):
    """Run the COMPS analysis for a single ticker."""
    
    start_time = datetime.now()
    logging.info(f"=== STARTING ANALYSIS FOR {ticker} ===")
    
    try:
        # Create database session
        engine = create_engine(DATABASE_URL)
        Session = sessionmaker(bind=engine)
        db_session = Session()
        
        # Initialize Finnhub client
        finnhub_client = get_finnhub_client()
        
        # Run the analysis
        logging.info(f"--- Processing {ticker} ---")
        report_filename, warnings = generate_comps_report(ticker, db_session, finnhub_client)
        
        if report_filename:
            logging.info(f"Successfully generated report for {ticker}: {report_filename}")
            if warnings:
                logging.warning(f"Analysis for {ticker} completed with warnings: {warnings}")
                
            # Print the full path to the report for easy access
            report_path = os.path.abspath(report_filename)
            logging.info(f"Report saved at: {report_path}")
            
            return True, report_path
        else:
            logging.error(f"Analysis failed for {ticker}, no report generated.")
            if warnings:
                logging.warning(f"Warnings: {warnings}")
            return False, None
                
    except Exception as e:
        logging.error(f"An unexpected error occurred while processing {ticker}: {e}", exc_info=True)
        return False, None
    finally:
        # Close database session if it was created
        if 'db_session' in locals():
            db_session.close()
        
        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"=== ANALYSIS COMPLETE FOR {ticker} ===")
        logging.info(f"Total Duration: {duration}")

def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Run COMPS analysis for a single ticker.")
    parser.add_argument('--ticker', type=str, required=True,
                      help="The ticker symbol to analyze (e.g., AAPL, MSFT)")
    args = parser.parse_args()
    
    # Validate the ticker
    ticker = args.ticker.strip().upper()
    if not ticker:
        logging.error("Please provide a valid ticker symbol.")
        return 1
    
    # Run the analysis
    success, report_path = run_single_ticker_analysis(ticker)
    
    if success:
        print(f"\nAnalysis completed successfully! Report saved at:\n{report_path}")
        return 0
    else:
        print(f"\nAnalysis failed for {ticker}. Check the logs for more details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())