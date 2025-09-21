#!/usr/bin/env python3
"""
Test script for the comps_analysis module with a specific focus on AAPL
"""

import pandas as pd
import numpy as np
from src.core.database_schema import get_db_session
from src.core.comps_analysis import get_database_peers, get_finnhub_client
from src.data.get_sp500_tickers import get_sp500_info

def test_comps_analysis_for_aapl():
    """Test the comps analysis functionality for AAPL"""
    print("Testing comps analysis for AAPL...")
    
    # Get S&P 500 data
    sp500_df = get_sp500_info()
    
    # Ensure AAPL is in our test data
    aapl_row = sp500_df[sp500_df['ticker'] == 'AAPL']
    if aapl_row.empty:
        print("Error: AAPL not found in S&P 500 data")
        return False
    
    # Get database peers
    session = get_db_session()
    peers = get_database_peers('AAPL', session)
    print(f"Database peers for AAPL: {peers}")
    
    # Create a focused sample dataframe for testing
    # Include AAPL and its peers
    test_tickers = ['AAPL'] + peers[:5]  # AAPL plus up to 5 peers
    
    # Filter S&P 500 data to our test tickers
    test_companies = sp500_df[sp500_df['ticker'].isin(test_tickers)]
    print(f"Test companies found in S&P 500: {test_companies['ticker'].tolist()}")
    
    # Create financials_df with our test tickers
    financials_df = pd.DataFrame({
        'ticker': test_companies['ticker'],
        'name': test_companies['name'],
        'sector': test_companies['sector'],
        'industry': test_companies['industry'],
        'market_cap': np.random.uniform(1000, 100000, len(test_companies)),
        'revenue': np.random.uniform(500, 50000, len(test_companies)),
        'ebitda': np.random.uniform(100, 20000, len(test_companies)),
        'net_income': np.random.uniform(50, 10000, len(test_companies)),
        'total_debt': np.random.uniform(100, 30000, len(test_companies)),
        'lease_liabilities': np.random.uniform(10, 5000, len(test_companies)),
        'minority_interest': np.random.uniform(0, 1000, len(test_companies)),
        'cash': np.random.uniform(100, 10000, len(test_companies)),
        'long_term_investments': np.random.uniform(50, 5000, len(test_companies)),
        'total_equity': np.random.uniform(500, 50000, len(test_companies)),
        'growth_rate': np.random.uniform(0.01, 0.2, len(test_companies)),
        'free_cash_flow': np.random.uniform(50, 5000, len(test_companies))
    })
    
    # Verify AAPL is in our financials dataframe
    if 'AAPL' not in financials_df['ticker'].values:
        print("Error: AAPL missing from test financials dataframe")
        return False
    
    print("Test data prepared successfully!")
    print(f"Financials dataframe contains {len(financials_df)} companies")
    print(financials_df[['ticker', 'name', 'market_cap', 'revenue', 'ebitda']].to_string(index=False))
    
    # Add enterprise value and valuation metrics
    # This simulates what would normally happen in the full analysis
    financials_df['enterprise_value'] = financials_df['market_cap'] + financials_df['total_debt'] - financials_df['cash']
    financials_df['ev_to_revenue'] = financials_df['enterprise_value'] / financials_df['revenue']
    financials_df['ev_to_ebitda'] = financials_df['enterprise_value'] / financials_df['ebitda']
    
    # Compute basic similarity score
    aapl_row = financials_df[financials_df['ticker'] == 'AAPL'].iloc[0]
    financials_df['similarity_score'] = 1.0 - abs(financials_df['market_cap'] - aapl_row['market_cap']) / aapl_row['market_cap']
    
    print("\nSimilarity scores:")
    print(financials_df[['ticker', 'similarity_score']].sort_values('similarity_score', ascending=False).to_string(index=False))
    
    # Close the database session
    session.close()
    
    return True

if __name__ == "__main__":
    success = test_comps_analysis_for_aapl()
    print(f"\nTest completed {'successfully' if success else 'with errors'}")