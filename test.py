import yfinance as yf
import pandas as pd
import numpy as np

def get_financial_data(ticker_symbol):
    """
    Fetches a comprehensive set of financial data for a given stock ticker.
    This version is more robust against missing data.
    """
    try:
        company = yf.Ticker(ticker_symbol)
        data = {}

        # 1. Stock and Market Data from .info
        # Use .get() to avoid errors if a key is missing
        info = company.info
        data['current_price'] = info.get('currentPrice')
        data['beta'] = info.get('beta')
        data['shares_outstanding'] = info.get('sharesOutstanding')
        
        # Fallback for price if not available in .info
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

        # --- IMPROVEMENT 1: Safely extract financial metrics ---
        # Define the metrics we want and which statement they belong to
        metric_map = {
            'Total Revenue': financials,
            'EBITDA': financials,
            'EBIT': financials,
            'Net Income': financials,
            'Interest Expense': financials,
            'Cash And Cash Equivalents': balance_sheet,
            'Total Debt': balance_sheet,
            'Depreciation And Amortization': cash_flow,
            'Capital Expenditure': cash_flow,
            'Cash Flow From Operating Activities': cash_flow,
        }

        for metric, statement_df in metric_map.items():
            # Check if the metric exists in the statement's index (rows)
            if not statement_df.empty and metric in statement_df.index:
                data[metric] = statement_df.loc[metric]
            else:
                # If metric is not found, record it as None to handle later
                data[metric] = None
                print(f"Warning: Metric '{metric}' not found for {ticker_symbol}.")

        return data

    except Exception as e:
        print(f"An error occurred while fetching data for ticker {ticker_symbol}: {e}")
        return None

# --- Main execution block ---
if __name__ == "__main__":
    ticker = "AAPL"  # Example: Apple Inc.
    financial_data = get_financial_data(ticker)

    if financial_data:
        print(f"Successfully fetched data for {ticker}")

        # --- Part 1: Process and Save Financial Statements ---
        statement_keys = [
            'Total Revenue', 'EBITDA', 'EBIT', 'Net Income', 'Interest Expense',
            'Cash And Cash Equivalents', 'Total Debt', 'Depreciation And Amortization',
            'Capital Expenditure', 'Cash Flow From Operating Activities'
        ]
        
        # Create a dictionary of the financial statement Series that were found
        statement_data = {key: financial_data[key] for key in statement_keys if financial_data[key] is not None}

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

            financials_filename = f"{ticker}_financials.csv"
            financials_df.to_csv(financials_filename)
            print(f"Financial statements and summary data saved to '{financials_filename}'")
        else:
            print("No financial statement data was available to save.")

        # --- Part 2: Process and Save Historical Stock Prices ---
        historical_df = financial_data.get('historical_prices')
        if historical_df is not None and not historical_df.empty:
            prices_filename = f"{ticker}_historical_prices.csv"
            historical_df.to_csv(prices_filename)
            print(f"Historical prices saved to '{prices_filename}'")
        else:
            print("No historical price data was available to save.")
