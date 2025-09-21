#!/usr/bin/env python3
"""
Fix for main.py to ensure target ticker is in sample data
"""

import os
import sys
from src.core.database_schema import get_db_session
from src.core.comps_analysis import get_finnhub_client, get_database_peers
from src.data.get_sp500_tickers import get_sp500_info
import pandas as pd
import numpy as np

import pytest

@pytest.fixture(params=['AAPL', 'MSFT'])
def ticker(request):
    return request.param

def test_run_with_ticker(ticker):
    """Run a test of the analysis pipeline with a specific ticker"""
    print(f"\n🔄 Starting analysis for {ticker}...")
    
    # Get the database session and API client
    db_session = get_db_session()
    finnhub_client = get_finnhub_client()
    
    # Get S&P 500 companies
    sp500_df = get_sp500_info()
    
    if sp500_df is not None:
        # Make sure the target ticker is in the S&P 500 data
        target_row = sp500_df[sp500_df['ticker'] == ticker]
        if target_row.empty:
            print(f"⚠️ Warning: {ticker} not found in S&P 500 data. Using fallback.")
            # Create a fallback row for the target company
            target_row = pd.DataFrame({
                'ticker': [ticker],
                'name': [f"{ticker} Inc."],
                'sector': ["Information Technology"],
                'industry': ["Technology Hardware, Storage & Peripherals"]
            })
        
        # Get peers from database
        peers = get_database_peers(ticker, db_session)
        print(f"📊 Found {len(peers)} database peers for {ticker}")
        
        # Create a sample of companies including the target and some peers
        sample_tickers = [ticker] + peers[:4]  # Target plus up to 4 peers
        
        # Get those companies from the S&P 500 data
        sample_companies = sp500_df[sp500_df['ticker'].isin(sample_tickers)]
        
        # Add any missing companies (might happen if peers aren't in S&P 500)
        missing_tickers = [t for t in sample_tickers if t not in sample_companies['ticker'].values]
        for t in missing_tickers:
            # Create a placeholder row
            sample_companies = pd.concat([
                sample_companies,
                pd.DataFrame({
                    'ticker': [t],
                    'name': [f"{t} Inc."],
                    'sector': ["Unknown"],
                    'industry': ["Unknown"]
                })
            ])
        
        # Add more random companies to fill out the sample
        remaining_count = 20 - len(sample_companies)
        if remaining_count > 0:
            # Filter out companies we already included
            other_companies = sp500_df[~sp500_df['ticker'].isin(sample_companies['ticker'])]
            if len(other_companies) >= remaining_count:
                random_companies = other_companies.sample(remaining_count)
                sample_companies = pd.concat([sample_companies, random_companies])
        
        # Reset index after concatenation
        sample_companies = sample_companies.reset_index(drop=True)
        
        # Create sample financials_df
        financials_df = pd.DataFrame({
            'ticker': sample_companies['ticker'],
            'name': sample_companies['name'],
            'sector': sample_companies['sector'],
            'industry': sample_companies['industry'],
            'market_cap': np.random.uniform(1000, 100000, len(sample_companies)),
            'revenue': np.random.uniform(500, 50000, len(sample_companies)),
            'ebitda': np.random.uniform(100, 20000, len(sample_companies)),
            'net_income': np.random.uniform(50, 10000, len(sample_companies)),
            'total_debt': np.random.uniform(100, 30000, len(sample_companies)),
            'lease_liabilities': np.random.uniform(10, 5000, len(sample_companies)),
            'minority_interest': np.random.uniform(0, 1000, len(sample_companies)),
            'cash': np.random.uniform(100, 10000, len(sample_companies)),
            'long_term_investments': np.random.uniform(50, 5000, len(sample_companies)),
            'total_equity': np.random.uniform(500, 50000, len(sample_companies)),
            'growth_rate': np.random.uniform(0.01, 0.2, len(sample_companies)),
            'free_cash_flow': np.random.uniform(50, 5000, len(sample_companies))
        })
        
        # Verify target is in the financials dataframe
        if ticker in financials_df['ticker'].values:
            print(f"✅ Successfully included {ticker} in sample data")
        else:
            print(f"❌ Error: {ticker} missing from sample data")
            return False
        
        # Add enterprise value and valuation metrics
        financials_df['enterprise_value'] = financials_df['market_cap'] + financials_df['total_debt'] - financials_df['cash']
        financials_df['ev_to_revenue'] = financials_df['enterprise_value'] / financials_df['revenue']
        financials_df['ev_to_ebitda'] = financials_df['enterprise_value'] / financials_df['ebitda']
        financials_df['pe_ratio'] = financials_df['market_cap'] / financials_df['net_income']
        financials_df['peg_ratio'] = financials_df['pe_ratio'] / (financials_df['growth_rate'] * 100)
        financials_df['roic'] = financials_df['net_income'] / financials_df['total_equity']
        financials_df['fcf_yield'] = financials_df['free_cash_flow'] / financials_df['market_cap']
        
        # Compute similarity scores
        target_company = financials_df[financials_df['ticker'] == ticker].iloc[0].to_dict()
        
        weights = {
            'market_cap': 0.2,
            'revenue': 0.2,
            'ebitda': 0.2,
            'growth_rate': 0.2,
            'roic': 0.2
        }
        
        # Simple similarity score calculation
        financials_df['similarity_score'] = 0.0
        
        for metric, weight in weights.items():
            if target_company[metric] != 0:
                # Normalized difference for each metric
                financials_df[f'{metric}_similarity'] = 1.0 - abs(financials_df[metric] - target_company[metric]) / abs(target_company[metric])
                # Apply weight to this metric's similarity
                financials_df['similarity_score'] += weight * financials_df[f'{metric}_similarity']
            
        # Show top 5 most similar companies
        print("\n🔍 Most similar companies:")
        top_similar = financials_df.sort_values('similarity_score', ascending=False).head(5)
        for _, row in top_similar.iterrows():
            print(f"  • {row['ticker']} ({row['name']}): {row['similarity_score']:.2f}")
        
        # Close the database session
        db_session.close()
        
        return True
    else:
        print("❌ Failed to retrieve S&P 500 information.")
        return False

if __name__ == "__main__":
    # Use command line argument as ticker or default to AAPL
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    success = test_run_with_ticker(ticker)
    print(f"\nTest completed {'successfully' if success else 'with errors'}")