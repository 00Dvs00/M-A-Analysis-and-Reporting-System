import os
import re
import yfinance as yf
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader
import finnhub 
from dotenv import load_dotenv

# --- Part 0a: Finnhub Peer Retrieval (MODIFIED) ---

def get_peer_tickers(ticker, finnhub_client):
    """
    Fetches a list of peer tickers from the Finnhub API.
    """
    print(f"--- Fetching Peer Tickers for {ticker} using Finnhub ---")
    try:
        peers = finnhub_client.company_peers(ticker)
        # The first element is often the company itself, so we can filter it out if needed,
        # but for this purpose, we'll rely on Finnhub's list as-is.
        if peers:
            print(f"Successfully found {len(peers)} peers: {', '.join(peers)}")
            return peers
        else:
            print(f"No peers found for {ticker} via Finnhub API.")
            return []
    except Exception as e:
        print(f"Could not fetch peer data for {ticker} from Finnhub: {e}")
        return []

# --- Part 0b: Risk-Free Rate Retrieval (MODIFIED) ---

def get_risk_free_rate():
    """
    Fetches the latest 10-Year Treasury yield (^TNX) using yfinance.
    """
    print("--- Fetching Risk-Free Rate (10-Year Treasury Yield via ^TNX) ---")
    try:
        tnx = yf.Ticker("^TNX")
        # Get the most recent data point
        hist = tnx.history(period="1d")
        if not hist.empty:
            # The 'Close' price for ^TNX is the yield percentage
            latest_yield = hist['Close'].iloc[-1]
            risk_free_rate = latest_yield / 100  # Convert percentage to decimal
            print(f"Successfully fetched 10-Year Treasury Yield: {risk_free_rate:.4%}")
            return risk_free_rate
        else:
            print("Could not retrieve ^TNX data from yfinance.")
            return None
    except Exception as e:
        print(f"Could not fetch risk-free rate from yfinance: {e}")
        return None

# --- Part 1: yfinance Data Fetcher (Unchanged) ---

def get_financial_data(ticker_symbol):
    """
    Fetches a comprehensive set of financial data for a given stock ticker.
    This version includes a debug helper to print available metrics if a lookup fails.
    """
    try:
        company = yf.Ticker(ticker_symbol)
        data = {}

        # 1. Stock and Market Data from .info
        info = company.info
        data['current_price'] = info.get('currentPrice')
        data['beta'] = info.get('beta')
        data['shares_outstanding'] = info.get('sharesOutstanding')
        data['_debug_metrics_printed'] = False # Flag to prevent repeated printing

        if not data['current_price']:
            history = company.history(period='1d')
            if not history.empty:
                data['current_price'] = history['Close'].iloc[-1]

        # 2. Historical Prices (DataFrame)
        data['historical_prices'] = company.history(period="5y")

        # 3. Financial Statements Data
        financials = company.financials
        balance_sheet = company.balance_sheet
        cash_flow = company.cashflow

        # --- UPDATED METRIC MAP from your last request ---
        metric_map = {
            # Our Standard Name: ([List of Possible yfinance Names], Statement DataFrame)
            
            # Income Statement Metrics
            'Total Revenue': (['Total Revenue'], financials),
            'EBITDA': (['EBITDA'], financials),
            'EBIT': (['EBIT'], financials),
            'Net Income': (['Net Income'], financials),
            'Interest Expense': (['Interest Expense', 'Net Non Operating Interest Income Expense'], financials),
            # NEW: Added Tax Provision as a fallback for taxes
            'Tax Provision': (['Tax Provision'], financials),

            # Balance Sheet Metrics
            'Total Current Assets': (['Total Current Assets', 'Current Assets'], balance_sheet),
            'Total Current Liabilities': (['Total Current Liabilities', 'Current Liabilities'], balance_sheet),
            'Cash And Cash Equivalents': (['Cash And Cash Equivalents', 'Cash'], balance_sheet),
            'Total Debt': (['Total Debt'], balance_sheet),
            
            # Cash Flow Metrics
            'Depreciation And Amortization': (['Depreciation And Amortization', 'Depreciation'], cash_flow),
            'Capital Expenditure': (['Capital Expenditure'], cash_flow),
            # This remains to catch actual cash taxes paid when available
            'Income Taxes Paid': (['Income Tax Paid Supplemental Data', 'Income Taxes Paid'], cash_flow),
            'Operating Cash Flow': ([
                'Operating Cash Flow',
                'Cash Flow From Continuing Operating Activities',
                'Net Cash Provided by Operating Activities',
                'Total Cash From Operating Activities',
                'Cash Flow From Operating Activities',
                'Change In Cash and Cash Equivalents'
            ], cash_flow),
        }

        # Loop through our standard metrics and find the data
        for standard_name, (possible_names, statement_df) in metric_map.items():
            data[standard_name] = None
            found_metric = False
            if not statement_df.empty:
                for name in possible_names:
                    if name in statement_df.index:
                        data[standard_name] = statement_df.loc[name]
                        found_metric = True
                        break  # Found it, move on
            
            # --- THIS IS THE NEW DEBUGGING BLOCK ---
            if not found_metric:
                print(f"\nWarning: Metric '{standard_name}' (tried: {', '.join(possible_names)}) not found for {ticker_symbol}.")
                
                # Check if we've already printed the debug info for this ticker
                if not data.get('_debug_metrics_printed'):
                    print(f"--- DEBUG: Printing all available yfinance metrics for {ticker_symbol} ---")
                    
                    if not financials.empty:
                        print("\n--- Available Income Statement Metrics ---")
                        print(sorted(list(financials.index)))
                    else:
                        print("\n--- Income Statement data not available ---")

                    if not balance_sheet.empty:
                        print("\n--- Available Balance Sheet Metrics ---")
                        print(sorted(list(balance_sheet.index)))
                    else:
                        print("\n--- Balance Sheet data not available ---")

                    if not cash_flow.empty:
                        print("\n--- Available Cash Flow Metrics ---")
                        print(sorted(list(cash_flow.index)))
                    else:
                        print("\n--- Cash Flow data not available ---")
                    
                    print(f"---------------------------------------------------\n")
                    data['_debug_metrics_printed'] = True # Set flag to avoid re-printing

        # Clean up the helper key from the final data object
        del data['_debug_metrics_printed']
        return data

    except Exception as e:
        print(f"An error occurred while fetching data for ticker {ticker_symbol}: {e}")
        return None

# --- MODIFIED to include risk_free_rate ---
def save_yfinance_data(ticker, data, risk_free_rate):
    """Saves the data fetched from yfinance into CSV files."""
    if not data:
        print("No yfinance data to save.")
        return

    if not os.path.exists(ticker):
        os.makedirs(ticker)
        
    summary_data = {
        'Metric': ['Current Price', 'Beta', 'Shares Outstanding', 'Risk-Free Rate (10Y)'],
        'Value': [data['current_price'], data['beta'], data['shares_outstanding'], risk_free_rate]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_filename = os.path.join(ticker, f"{ticker}_yfinance_summary.csv")
    summary_df.to_csv(summary_filename, index=False)
    print(f"Saved yfinance summary to '{summary_filename}'")

    if data['historical_prices'] is not None and not data['historical_prices'].empty:
        hist_filename = os.path.join(ticker, f"{ticker}_historical_prices.csv")
        data['historical_prices'].to_csv(hist_filename)
        print(f"Saved 5-year historical prices to '{hist_filename}'")

# --- Part 2: SEC Filing Downloader (Unchanged) ---

def fetch_sec_filings(ticker, form_type, limit, company_name, email_address):
    """Fetches and downloads SEC filings for a given company and form type."""
    try:
        print(f"Downloading the {limit} most recent {form_type} filing(s) for {ticker}...")
        dl = Downloader(company_name, email_address)
        dl.get(form_type, ticker, limit=limit, download_details=True)
        print(f"Successfully downloaded {form_type} filing(s) for {ticker}.")
        return True
    except Exception as e:
        print(f"An error occurred while fetching {form_type} filings for {ticker}: {e}")
        return False

# --- Part 3: Financial Statement Parser (MODIFIED to fix XML warning) ---

def _clean_value(text):
    if text is None: return np.nan
    text = text.strip().replace('$', '').replace(',', '').replace('\n', ' ').replace('(', '-').replace(')', '')
    if text in ('', '—', 'N/A'): return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan

def _extract_financial_table(soup, table_keywords):
    table = None
    for keyword in table_keywords:
        header_tag = soup.find(string=re.compile(keyword, re.IGNORECASE | re.DOTALL))
        if header_tag:
            for parent in header_tag.find_parents('div', limit=5): 
                table = parent.find('table')
                if table: break
            if not table: table = header_tag.find_next('table')
            if table: break
    if not table: return None
    headers = []
    for row in table.find_all('tr'):
        cols = [col.get_text(strip=True) for col in row.find_all(['th', 'td'])]
        if len(cols) > 2 and any(re.search(r'\d{4}', c) for c in cols):
            headers = cols
            break
    if not headers: return None
    num_data_columns = len(headers) - 1
    if num_data_columns <= 0: return None
    data = {}
    for row in table.find_all('tr'):
        cols = row.find_all(['th', 'td'])
        if len(cols) > 1:
            metric_name = cols[0].get_text(strip=True)
            if metric_name and len(metric_name) > 3 and any(c.isalpha() for c in metric_name):
                values = [_clean_value(col.get_text()) for col in cols[1:]]
                padded_values = (values + [np.nan] * num_data_columns)[:num_data_columns]
                data[metric_name] = padded_values
    if not data: return None
    df = pd.DataFrame.from_dict(data, orient='index', columns=headers[1:num_data_columns + 1])
    df.index.name = "Metric"
    df.dropna(axis=1, how='all', inplace=True)
    df.columns = [re.sub(r'<.*?>', '', col) for col in df.columns]
    return df

def parse_financial_statements(filing_path):
    html_file = None
    for file in os.listdir(filing_path):
        if file.endswith(('.htm', '.html')):
            html_file = os.path.join(filing_path, file)
            break
    if not html_file and os.path.exists(os.path.join(filing_path, 'full-submission.txt')):
        html_file = os.path.join(filing_path, 'full-submission.txt')
    elif not html_file:
        print(f"No HTML filing document found in '{filing_path}'")
        return None
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'full-submission' in html_file:
                doc_start = content.find('<DOCUMENT>')
                if doc_start != -1: content = content[doc_start:]
            # Use 'lxml-xml' parser for SEC filings
            soup = BeautifulSoup(content, 'lxml-xml')
            statements = {}
            income_stmt_keys = ['Consolidated Statements of Operations', 'STATEMENTS OF INCOME', 'INCOME STATEMENTS', 'STATEMENTS OF OPERATIONS']
            balance_sheet_keys = ['Consolidated Balance Sheets', 'BALANCE SHEETS', 'STATEMENTS OF FINANCIAL POSITION', 'CONSOLIDATED FINANCIAL CONDITION']
            cash_flow_keys = ['Consolidated Statements of Cash Flows', 'STATEMENTS OF CASH FLOWS', 'CASH FLOW STATEMENTS']
            statements['income_statement'] = _extract_financial_table(soup, income_stmt_keys)
            statements['balance_sheet'] = _extract_financial_table(soup, balance_sheet_keys)
            statements['cash_flow'] = _extract_financial_table(soup, cash_flow_keys)
            return statements
    except Exception as e:
        print(f"An error occurred during parsing of {filing_path}: {e}")
        return None

# --- Part 4 & 5 (Unchanged, but with XML parser fix) ---
def extract_mda_section(filing_path):
    # ... (function remains the same, but ensure BeautifulSoup uses 'lxml-xml')
    html_file = None
    for file in os.listdir(filing_path):
        if file.endswith(('.htm', '.html')):
            html_file = os.path.join(filing_path, file)
            break
    if not html_file: return "HTML document for MD&A not found."
    with open(html_file, 'r', encoding='utf-8') as f:
        # Use 'lxml-xml' parser here as well
        soup = BeautifulSoup(f.read(), 'lxml-xml')
    mda_start_patterns = [
        re.compile(r"item\s+7\s*\.\s*management['’]s\s+discussion", re.IGNORECASE),
        re.compile(r"management['’]s\s+discussion\s+and\s+analysis\s+of\s+financial\s+condition", re.IGNORECASE)
    ]
    mda_end_patterns = [re.compile(r"item\s+7a\s*\.", re.IGNORECASE), re.compile(r"item\s+8\s*\.", re.IGNORECASE)]
    start_tag = None
    for pattern in mda_start_patterns:
        start_tag = soup.find(string=pattern)
        if start_tag: break
    if not start_tag: return "MD&A section start anchor not found."
    mda_content = []
    for elem in start_tag.find_all_next():
        if elem.find(string=mda_end_patterns[0]) or elem.find(string=mda_end_patterns[1]): break
        if elem.name and elem.name.startswith('h') and ("item 8" in elem.get_text(strip=True).lower()): break
        if elem.name == 'p': mda_content.append(elem.get_text(strip=True))
    return "\n\n".join(mda_content) if mda_content else "MD&A content could not be extracted."

def identify_non_recurring_items(financial_statements, mda_text):
    # This function is unchanged
    keywords = ['restructuring', 'impairment', 'one-time', 'gain on sale', 'litigation settlement', 'discontinued operations']
    # ... (rest of the function is identical)
    return {} # Placeholder for brevity

# --- MODIFIED Main Execution Block ---

if __name__ == "__main__":
    load_dotenv()
    
    primary_ticker = input("Enter the primary company ticker: ").strip().upper()
    
    # --- Credentials and Configuration ---
    COMPANY_NAME = "Personal Research Project"
    EMAIL_ADDRESS = "divyeshnarshi@gmail.com"
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

    if not FINNHUB_API_KEY:
        print("Error: FINNHUB_API_KEY not found. Please create a .env file with FINNHUB_API_KEY='your_key_here'")
        exit()
        
    # Initialize Finnhub client
    finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
    
    # --- Step 1: Get Risk-Free Rate ---
    risk_free_rate = get_risk_free_rate()
    if risk_free_rate is None:
        print("\nWarning: Could not retrieve risk-free rate. Using a hardcoded default of 4.25%.")
        risk_free_rate = 0.0425 # Fallback value

    # --- Step 2: Get Peer Tickers ---
    peer_tickers = get_peer_tickers(primary_ticker, finnhub_client)
    if not peer_tickers:
        print(f"Warning: Could not retrieve peer tickers. Proceeding with analysis for {primary_ticker} only.")
    
    tickers_to_analyze = list(dict.fromkeys([primary_ticker] + peer_tickers))
    
    print(f"\n>>> Starting analysis for the following tickers: {', '.join(tickers_to_analyze)} <<<\n")

    # --- Loop Through All Tickers for Analysis ---
    for ticker in tickers_to_analyze:
        print(f"\n{'='*20} Analyzing {ticker} {'='*20}")

        if '.' in ticker:
          print(f"Skipping SEC filings for {ticker} (non-U.S. or invalid ticker).")

        # Create a directory for the ticker's files
        if not os.path.exists(ticker):
            os.makedirs(ticker)
            
        # --- Step 3: Fetch Comprehensive Financial Data ---
        financial_data = get_financial_data(ticker) # Assumes you are using the robust version
        
        if financial_data:
            print(f"Successfully fetched yfinance data for {ticker}")

            # --- COMBINED PROCESSING AND SAVING LOGIC ---

            # Part 3a: Process Financial Statements and Summary Data
            statement_keys = [
                'Total Revenue', 'EBITDA', 'EBIT', 'Net Income', 'Interest Expense',
                'Tax Provision',  
                'Income Taxes Paid', 
                'Total Current Assets',
                'Total Current Liabilities', 
                'Cash And Cash Equivalents', 
                'Total Debt', 
                'Depreciation And Amortization',
                'Capital Expenditure', 
                'Operating Cash Flow'
            ]

            statement_data = {key: financial_data[key] for key in statement_keys if financial_data[key] is not None}
            
            # Filter out any metrics that were not found or were returned as empty
            statement_data = {k: v for k, v in statement_data.items() if v is not None and not v.empty}

            if statement_data:
                financials_df = pd.DataFrame(statement_data).transpose()
                financials_df.index.name = "Metric"
                
                # --- IMPROVEMENT 2: Ensure columns are sorted by date ---
                # Sort columns to make sure the most recent year is first
                financials_df = financials_df.sort_index(axis=1, ascending=False)
                
                # --- IMPROVEMENT 3: Add summary data cleanly using a loop ---
                summary_metrics_to_add = {
                    'Current Stock Price': financial_data.get('current_price'),
                    'Beta': financial_data.get('beta'),
                    'Shares Outstanding': financial_data.get('shares_outstanding')
                }

                if not financials_df.empty:
                    most_recent_column = financials_df.columns[0]
                    
                    for name, value in summary_metrics_to_add.items():
                        if value is not None:
                            # Create a new row with NaNs, then place the value in the most recent column
                            financials_df.loc[name] = np.nan
                            financials_df.loc[name, most_recent_column] = value

            financials_filename = os.path.join(ticker, f"{ticker}_financials.csv")
            financials_df.to_csv(financials_filename)
            print(f"Financial statements and summary data saved to '{financials_filename}'")
        else:
            print("No financial statement data was available to save.")

        # Part 3b: Save Historical Stock Prices
        if 'historical_prices' in financial_data and not financial_data['historical_prices'].empty:
            prices_filename = os.path.join(ticker, f"{ticker}_historical_prices.csv")
            financial_data['historical_prices'].to_csv(prices_filename)
            print(f"Historical prices saved to '{prices_filename}'")
        else:
            print("No historical price data was available to save.")

        # --- Step 4: Download SEC Filings ---
        print(f"--- Proceeding to download SEC filings for {ticker} ---")
        forms_to_download = {"10-K": 10, "10-Q": 4} # Reduced for brevity, add "8-K" if needed
        for form_type, limit in forms_to_download.items():
            fetch_sec_filings(ticker, form_type, limit, COMPANY_NAME, EMAIL_ADDRESS)
            
        # --- Step 5: Analyze Latest Filings (Placeholder) ---
        print(f"--- Placeholder for SEC filing analysis for {ticker} ---")
        pass

    print("\n--- Full Analysis Complete ---")