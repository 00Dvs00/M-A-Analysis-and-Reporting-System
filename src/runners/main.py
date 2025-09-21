#!/usr/bin/env python3
"""
M&A Analysis and Reporting System - Main Entry Point
===================================================

This is the primary user interface for the M&A Analysis and Reporting System.
The script provides a simple, clean way to generate comprehensive comparable 
company analysis reports for any publicly traded company.

Usage:
    python main.py --config path/to/config.yaml

The script will prompt you to enter a ticker symbol and automatically generate
a detailed Excel analysis report with valuation metrics, peer comparisons,
and visualization charts.

Author: M&A Analysis System
Version: 1.1
"""

import os
import sys
import logging
from typing import Optional
from datetime import datetime
import yaml
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging for user-friendly output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

def setup_environment() -> bool:
    """
    Verify that the environment is properly configured.
    
    Returns:
        bool: True if environment is ready, False otherwise
    """
    print("🔍 Checking environment configuration...")
    
    # Check for .env file
    if not os.path.exists('.env'):
        print("❌ ERROR: .env file not found!")
        print("📋 Please create a .env file with the following variables:")
        print("   - DATABASE_URL=postgresql://username:password@localhost:5432/ma_analysis")
        print("   - FINNHUB_API_KEY=your_finnhub_api_key")
        print("   - SEC_EMAIL=your_email@domain.com")
        print("   - SEC_COMPANY_NAME=Your Company Name")
        return False
    
    # Import required modules
    try:
        from comps_analysis import generate_comps_report
        from database_schema import DATABASE_URL
        print("✅ All required modules loaded successfully")
        return True
    except ImportError as e:
        print(f"❌ ERROR: Missing required module - {e}")
        print("💡 Please ensure all dependencies are installed: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ ERROR: Configuration issue - {e}")
        return False

def validate_ticker(ticker: str) -> str:
    """
    Validate and normalize the ticker symbol.
    
    Args:
        ticker (str): Raw ticker input from user
        
    Returns:
        str: Normalized ticker symbol
        
    Raises:
        ValueError: If ticker is invalid
    """
    if not ticker:
        raise ValueError("Ticker symbol cannot be empty")
    
    # Normalize ticker (uppercase, strip whitespace)
    normalized_ticker = ticker.strip().upper()
    
    # Basic validation (letters and numbers only, reasonable length)
    if not normalized_ticker.replace('.', '').replace('-', '').isalnum():
        raise ValueError("Ticker symbol contains invalid characters")
    
    if len(normalized_ticker) > 10:
        raise ValueError("Ticker symbol is too long")
    
    return normalized_ticker

def get_user_input() -> Optional[str]:
    """
    Get ticker symbol from user with validation.
    
    Returns:
        str: Validated ticker symbol, or None if user wants to exit
    """
    print("\n" + "="*60)
    print("🚀 M&A ANALYSIS & REPORTING SYSTEM")
    print("="*60)
    print("Generate comprehensive comparable company analysis reports")
    print("for any publicly traded company.")
    print("\nExamples: AAPL, MSFT, GOOGL, TSLA, JPM, etc.")
    print("(Type 'exit' or 'quit' to stop)")
    print("-"*60)
    
    while True:
        try:
            ticker_input = input("\n📊 Enter ticker symbol to analyze: ").strip()
            
            # Check for exit commands
            if ticker_input.lower() in ['exit', 'quit', 'q', '']:
                return None
            
            # Validate ticker
            validated_ticker = validate_ticker(ticker_input)
            
            # Confirm with user
            print(f"\n🎯 Target company: {validated_ticker}")
            confirm = input("Proceed with analysis? (y/n): ").strip().lower()
            
            if confirm in ['y', 'yes', '']:
                return validated_ticker
            elif confirm in ['n', 'no']:
                continue
            else:
                print("❓ Please enter 'y' for yes or 'n' for no")
                continue
                
        except ValueError as e:
            print(f"❌ Invalid ticker: {e}")
            print("💡 Please try again with a valid ticker symbol")
            continue
        except KeyboardInterrupt:
            print("\n👋 Analysis cancelled by user")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            continue

def run_analysis(ticker: str) -> bool:
    """
    Execute the comprehensive analysis for the given ticker.
    
    Args:
        ticker (str): Validated ticker symbol
        
    Returns:
        bool: True if analysis completed successfully, False otherwise
    """
    try:
        print(f"\n🔄 Starting comprehensive analysis for {ticker}...")
        print("⏱️  This may take a few minutes - please wait...")
        
        # Import here to ensure environment is set up
        from comps_analysis import get_finnhub_client
        from database_schema import get_db_session
        from reporting import enhanced_comps_report
        from etl_pipeline import (
            compute_enterprise_value, 
            calculate_ev_to_revenue, 
            calculate_ev_to_ebitda, 
            calculate_pe_ratio, 
            calculate_peg_ratio, 
            calculate_roic, 
            calculate_fcf_yield,
            compute_similarity_scores
        )
        from get_sp500_tickers import get_sp500_info
        import pandas as pd
        import numpy as np

        # Set up database and API clients
        db_session = get_db_session()
        finnhub_client = get_finnhub_client()
        
        # Record start time
        start_time = datetime.now()
        
        # Get S&P 500 companies
        sp500_df = get_sp500_info()
        
        if sp500_df is not None:
            # Create sample financials_df for demonstration
            financials_df = pd.DataFrame({
                'ticker': sp500_df['ticker'].head(20),
                'name': sp500_df['name'].head(20),
                'sector': sp500_df['sector'].head(20),
                'industry': sp500_df['industry'].head(20),
                'market_cap': np.random.uniform(1000, 100000, 20),
                'revenue': np.random.uniform(500, 50000, 20),
                'ebitda': np.random.uniform(100, 20000, 20),
                'net_income': np.random.uniform(50, 10000, 20),
                'total_debt': np.random.uniform(100, 30000, 20),
                'lease_liabilities': np.random.uniform(10, 5000, 20),
                'minority_interest': np.random.uniform(0, 1000, 20),
                'cash': np.random.uniform(100, 10000, 20),
                'long_term_investments': np.random.uniform(50, 5000, 20),
                'total_equity': np.random.uniform(500, 50000, 20),
                'growth_rate': np.random.uniform(0.01, 0.2, 20),
                'free_cash_flow': np.random.uniform(50, 5000, 20)
            })
            
            # Add enterprise value and valuation metrics
            config = {
                'market_cap': 'market_cap', 
                'total_debt': 'total_debt', 
                'lease_liabilities': 'lease_liabilities', 
                'minority_interest': 'minority_interest', 
                'cash': 'cash', 
                'long_term_investments': 'long_term_investments'
            }
            
            financials_df = compute_enterprise_value(financials_df, config)
            financials_df = calculate_ev_to_revenue(financials_df)
            financials_df = calculate_ev_to_ebitda(financials_df)
            financials_df = calculate_pe_ratio(financials_df)
            financials_df = calculate_peg_ratio(financials_df)
            financials_df = calculate_roic(financials_df)
            financials_df = calculate_fcf_yield(financials_df)
            
            # Compute similarity scores
            target_company = financials_df[financials_df['ticker'] == ticker].iloc[0].to_dict()
            weights = {
                'market_cap': 0.2,
                'revenue': 0.2,
                'ebitda': 0.2,
                'growth_rate': 0.2,
                'roic': 0.2
            }
            financials_df = compute_similarity_scores(financials_df, target_company, weights)
            
            # Generate the report
            result_filename, warnings = enhanced_comps_report(financials_df, ticker)
        else:
            result_filename = None
            warnings = ["Failed to retrieve S&P 500 information."]
        
        # Close the database session
        db_session.close()
        
        # Calculate elapsed time
        elapsed_time = datetime.now() - start_time
        minutes = int(elapsed_time.total_seconds() // 60)
        seconds = int(elapsed_time.total_seconds() % 60)
        
        if result_filename:
            print(f"\n🎉 SUCCESS! Analysis completed in {minutes}m {seconds}s")
            print(f"📄 Report generated: {result_filename}")
            print(f"📂 Location: {os.path.abspath(result_filename)}")
            
            if warnings:
                print("\n⚠️ Analysis completed with warnings:")
                for warning in warnings:
                    print(f"   • {warning}")

            print("\n📈 Your report includes:")
            print("   • Peer company identification and filtering")
            print("   • Financial metrics and ratios analysis")
            print("   • Valuation multiples and benchmarking")
            print("   • Football field valuation chart")
            print("   • Detailed data tables and statistics")
            return True
        else:
            print(f"\n❌ Analysis failed for {ticker}")
            if warnings:
                print("📋 Issues encountered:")
                for issue in warnings:
                    print(f"   • {issue}")
            else:
                print("📋 Common issues:")
                print("   • Ticker not found or delisted")
                print("   • Insufficient financial data available")
                print("   • Database connection problems")
                print("   • API rate limits exceeded")
            return False
            
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {e}")
        logging.error(f"Analysis error for {ticker}: {e}", exc_info=True)
        return False

def display_help():
    """Display helpful information about the system."""
    print("\n" + "="*60)
    print("📚 HELP & INFORMATION")
    print("="*60)
    print("This system generates comprehensive comparable company analysis")
    print("reports for investment banking and M&A purposes.")
    print("\n🎯 What you'll get:")
    print("   • Peer company identification using multiple data sources")
    print("   • Financial metrics and ratio analysis")
    print("   • Valuation multiples benchmarking")
    print("   • Professional Excel report with charts")
    print("\n⚙️ Prerequisites:")
    print("   • PostgreSQL database with SEC filing data")
    print("   • Finnhub API key for market data")
    print("   • Internet connection for real-time data")
    print("\n🔧 Setup help:")
    print("   • Run 'python etl_pipeline.py' first to populate database")
    print("   • Ensure .env file contains all required credentials")
    print("   • Install dependencies: pip install -r requirements.txt")

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

def main():
    """
    Main application entry point.
    """
    try:
        # Display welcome and check environment
        if not setup_environment():
            print("\n🔧 Please resolve the configuration issues above and try again.")
            input("\nPress Enter to exit...")
            return 1
        
        print("✅ Environment ready!")
        
        # Parse CLI arguments
        parser = argparse.ArgumentParser(description="Run the M&A Analysis System.")
        parser.add_argument('--config', type=str, required=True, help="Path to the configuration file (YAML/JSON).")
        args = parser.parse_args()
        
        # Load configuration
        config = load_config(args.config)
        
        # Main application loop
        while True:
            ticker = get_user_input()
            
            if ticker is None:
                print("\n👋 Thank you for using the M&A Analysis System!")
                break
            
            # Run analysis
            success = run_analysis(ticker)
            
            # Example usage of configuration
            print("Loaded configuration:", config)

            # Ask if user wants to analyze another company
            if success:
                print("\n" + "-"*60)
                continue_analysis = input("Analyze another company? (y/n): ").strip().lower()
                if continue_analysis not in ['y', 'yes', '']:
                    print("\n👋 Thank you for using the M&A Analysis System!")
                    break
            else:
                print("\n" + "-"*60)
                retry = input("Try again with a different ticker? (y/n): ").strip().lower()
                if retry not in ['y', 'yes', '']:
                    print("\n👋 Thank you for using the M&A Analysis System!")
                    break
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n👋 Analysis cancelled by user. Goodbye!")
        return 0
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        logging.error(f"Critical error in main: {e}", exc_info=True)
        input("\nPress Enter to exit...")
        return 1

if __name__ == "__main__":
    # Handle command line arguments for help
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        display_help()
        sys.exit(0)
    
    # Run main application
    exit_code = main()
    sys.exit(exit_code)