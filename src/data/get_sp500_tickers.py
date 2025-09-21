import pandas as pd
import requests
import logging
from datetime import datetime
from sqlalchemy.orm import sessionmaker
from src.core.database_schema import Company, get_db_session, engine  # Correct import for database setup

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_sp500_info():
    """
    Scrapes Wikipedia for the list of S&P 500 companies, their tickers, CIKs,
    sectors, and industries.

    Returns:
        A pandas DataFrame with the company information, or None if it fails.
    """
    logging.info("--- Fetching S&P 500 component list from Wikipedia ---")
    WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    # Use a user-agent to be respectful to Wikipedia's servers
    headers = {'User-Agent': 'FinancialDataAnalysis/1.0 (Contact: your.email@example.com)'}
    
    try:
        response = requests.get(WIKI_URL, headers=headers, timeout=10)
        # Raise an exception for bad status codes (like 404 or 500)
        response.raise_for_status()
        
        # pd.read_html() automatically finds and parses tables from HTML
        tables = pd.read_html(response.text)
        sp500_df = tables[0]

        # Clean up column names to match what our database expects
        sp500_df.rename(columns={
            'Symbol': 'ticker',
            'Security': 'name',
            'GICS Sector': 'sector',
            'GICS Sub-Industry': 'industry',
            'CIK': 'cik'
        }, inplace=True)

        # Ensure the CIK is treated as a number, handling any potential errors
        sp500_df['cik'] = pd.to_numeric(sp500_df['cik'], errors='coerce')
        sp500_df.dropna(subset=['cik'], inplace=True) # Remove rows where CIK is invalid
        sp500_df['cik'] = sp500_df['cik'].astype(int)

        logging.info(f"Successfully fetched and parsed {len(sp500_df)} companies from the S&P 500 list.")
        
        # Return only the columns we need
        return sp500_df[['ticker', 'name', 'cik', 'sector', 'industry']]
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Could not fetch data from Wikipedia: {e}")
        return None
    except Exception as e:
        logging.error(f"An error occurred during scraping or data processing: {e}")
        return None

# --- Update Database with Active/Inactive Flags ---
def update_ticker_universe(session, sp500_df):
    """
    Updates the database with active/inactive flags based on the latest S&P 500 data.
    
    Args:
        session: Database session object.
        sp500_df: DataFrame containing the latest S&P 500 data.
    """
    # Fetch all companies from the database
    db_companies = session.query(Company).all()
    db_tickers = {company.ticker: company for company in db_companies}

    # Mark all companies as inactive initially
    for company in db_companies:
        company.active = 0
        company.inactive_date = datetime.now().date()

    # Update active companies based on the latest S&P 500 data
    for _, row in sp500_df.iterrows():
        ticker = row['ticker']
        if ticker in db_tickers:
            company = db_tickers[ticker]
            company.active = 1
            company.inactive_date = None
        else:
            # Add new company to the database
            new_company = Company(
                ticker=row['ticker'],
                cik=row['cik'],
                name=row['name'],
                sector=row['sector'],
                industry=row['industry'],
                active=1,
                membership_date=datetime.now().date()
            )
            session.add(new_company)

    session.commit()
    logging.info("Ticker universe updated successfully.")

if __name__ == '__main__':
    # This block allows you to test this script by itself
    print("Running a standalone test to fetch S&P 500 company list...")
    sp500_companies = get_sp500_info()
    if sp500_companies is not None:
        # Save to CSV for inspection or backup
        sp500_companies.to_csv("sp500_companies.csv", index=False)
        print("S&P 500 company list saved to 'sp500_companies.csv'")
        print("\nFirst 5 companies:")
        print(sp500_companies.head())
    
    # Database update part - this requires a valid database session
    Session = sessionmaker(bind=engine)
    session = Session()
    
    update_ticker_universe(session, sp500_companies)