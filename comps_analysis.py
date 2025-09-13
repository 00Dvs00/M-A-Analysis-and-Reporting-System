import pandas as pd
import yfinance as yf
import logging
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import finnhub
from scipy.stats import trim_mean

# --- Local Imports ---
from database_schema import DATABASE_URL, Company

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
pd.set_option('display.float_format', lambda x: '%.2f' % x)
load_dotenv()

# --- Helper Functions ---

def get_finnhub_client():
    """Initializes and returns a Finnhub client if the API key is available."""
    finnhub_key = os.getenv("FINNHUB_API_KEY")
    if not finnhub_key:
        logging.warning("FINNHUB_API_KEY not found in .env. Some features will be limited.")
        return None
    return finnhub.Client(api_key=finnhub_key)

def get_finnhub_peers(target_ticker, client):
    """Gets a list of peer tickers from the Finnhub API."""
    if not client: return []
    try:
        peers = client.company_peers(target_ticker)
        if peers and len(peers) > 1:
            logging.info(f"Successfully fetched {len(peers)-1} initial peers from Finnhub.")
            return peers[1:]
    except Exception as e:
        logging.error(f"Error fetching peers from Finnhub: {e}")
    return []

def get_revenue_segments(ticker, client):
    """Gets revenue segments for a company to enable business model comparison."""
    if not client: return set()
    try:
        segments = client.company_revenue_breakdown(symbol=ticker) # Corrected method name
        if segments and 'breakdown' in segments and segments['breakdown']:
            return {item['segment'] for item in segments['breakdown']}
    except Exception as e:
        logging.warning(f"Could not fetch revenue segments for {ticker}: {e}")
    return set()

def get_comps_data(tickers):
    """Fetches and calculates all necessary financial data for a list of tickers."""
    if not tickers: return pd.DataFrame()
    logging.info(f"Fetching financial data for {len(tickers)} tickers...")
    
    data_list = []
    chunk_size = 50
    ticker_chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]

    for chunk in ticker_chunks:
        ticker_objects = yf.Tickers(' '.join(chunk))
        for ticker_symbol, ticker_obj in ticker_objects.tickers.items():
            try:
                info = ticker_obj.info
                if not info.get('marketCap'):
                    logging.warning(f"Skipping {ticker_symbol} due to missing market cap data.")
                    continue
                
                data_list.append({
                    'Ticker': ticker_symbol, 'Company Name': info.get('shortName'), 'Price': info.get('currentPrice', 0),
                    'Shares Out': info.get('sharesOutstanding', 0), 'Market Cap': info.get('marketCap'),
                    'Debt': info.get('totalDebt', 0), 'Cash': info.get('totalCash', 0),
                    'EV': info.get('marketCap') + info.get('totalDebt', 0) - info.get('totalCash', 0),
                    'Revenue LTM': info.get('totalRevenue', 0), 'EBITDA LTM': info.get('ebitda', 0),
                    'Net Income LTM': info.get('netIncomeToCommon', 0), 'EPS LTM': info.get('trailingEps'),
                    'EPS Forward': info.get('forwardEps')
                })
                logging.info(f"Successfully processed data for {ticker_symbol}")
            except Exception:
                logging.error(f"Could not process data for {ticker_symbol}.")
    
    df = pd.DataFrame(data_list)
    if df.empty: return df

    # Calculate Multiples
    df['EV/Revenue LTM'] = df['EV'] / df['Revenue LTM'].replace(0, pd.NA)
    df['EV/EBITDA LTM'] = df['EV'] / df['EBITDA LTM'].replace(0, pd.NA)
    df['P/E LTM'] = df['Market Cap'] / df['Net Income LTM'].replace(0, pd.NA)
    df['P/E Forward'] = df['Price'] / df['EPS Forward'].replace(0, pd.NA)
    
    return df

# --- Main Analysis Function ---

def generate_comps_report(target_ticker):
    """
    Generates an enhanced comparable company analysis with robust filtering and statistics.
    """
    finnhub_client = get_finnhub_client()
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    target_company_db = session.query(Company).filter(Company.ticker == target_ticker).first()
    if not target_company_db:
        logging.warning(f"Target ticker {target_ticker} not found in database. Proceeding with yfinance data only.")
        # Continue with the analysis using only yfinance data
    else:
        logging.info(f"Found {target_ticker} in database: {target_company_db.name}")

    # 1. EXPAND PEER GROUP (with Fallback)
    peer_tickers = get_finnhub_peers(target_ticker, finnhub_client)
    if not peer_tickers:
        logging.warning("Finnhub did not return peers. Falling back to industry peers from the database.")
        if target_company_db:
            peers_db = session.query(Company).filter(Company.industry == target_company_db.industry, Company.ticker != target_ticker).all()
            peer_tickers = [p.ticker for p in peers_db]
        else:
            # If no database entry and no Finnhub peers, use a default set of tech peers for MSFT
            if target_ticker == "MSFT":
                peer_tickers = ["AAPL", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "ORCL", "CRM", "ADBE", "INTC"]
                logging.info(f"Using default tech peers for {target_ticker}")
    
    if not peer_tickers:
        logging.error("No peers found in Finnhub or the local database. Cannot proceed.")
        session.close()
        return
    session.close()
    
    # 2. SEGMENT-BASED PEER FILTERING
    target_segments = get_revenue_segments(target_ticker, finnhub_client)
    if target_segments:
        relevant_peers = [p for p in peer_tickers if target_segments.intersection(get_revenue_segments(p, finnhub_client))]
        logging.info(f"Filtered peers by business segment: {len(peer_tickers)} -> {len(relevant_peers)} companies.")
        if relevant_peers: peer_tickers = relevant_peers

    # 3. GATHER DATA & CLEAN
    full_ticker_list = list(set(peer_tickers + [target_ticker]))
    full_comps_df = get_comps_data(full_ticker_list)
    
    if full_comps_df.empty or target_ticker not in full_comps_df['Ticker'].values:
        logging.error(f"Failed to retrieve necessary financial data for {target_ticker} or its peers.")
        return

    target_data = full_comps_df[full_comps_df['Ticker'] == target_ticker]
    peer_data = full_comps_df[full_comps_df['Ticker'] != target_ticker].copy()

    # Filter out loss-making companies
    peer_data = peer_data[(peer_data['EBITDA LTM'] > 0) & (peer_data['Net Income LTM'] > 0)]
    
    # 4. ADAPTIVE SIZE FILTERING
    initial_peer_count = len(peer_data)
    size_warning = ""
    MIN_PEERS_FOR_SIZE_FILTER = 8 
    if initial_peer_count >= MIN_PEERS_FOR_SIZE_FILTER:
        target_market_cap = target_data['Market Cap'].iloc[0]
        lower_bound, upper_bound = target_market_cap * 0.50, target_market_cap * 2.0
        peer_data = peer_data[(peer_data['Market Cap'] >= lower_bound) & (peer_data['Market Cap'] <= upper_bound)]
        logging.info(f"Filtered large peer group by market cap (50%-200% of target): {initial_peer_count} -> {len(peer_data)} companies.")
    else:
        size_warning = f"WARNING: Peer group has only {initial_peer_count} companies. Size filter was skipped to preserve sample size. Results may be affected by size disparities."
        logging.warning(size_warning)

    if len(peer_data) < 3:
        logging.error("Insufficient profitable peers remain after filtering. Cannot generate reliable report.")
        return

    # 5. CALCULATE ROBUST STATISTICS
    multiples = ['EV/Revenue LTM', 'EV/EBITDA LTM', 'P/E LTM', 'P/E Forward']
    summary_stats = peer_data[multiples].describe(percentiles=[.25, .5, .75]).loc[['25%', '50%', '75%']].rename(index={'50%': 'Median'})
    
    for m in multiples:
        summary_stats.loc['Trimmed Mean (10%)', m] = trim_mean(peer_data[m].dropna(), 0.1)
        summary_stats.loc['Weighted Avg', m] = (peer_data[m] * peer_data['Market Cap']).sum() / peer_data['Market Cap'].sum()

    # 6. APPLY MULTIPLES & CREATE VALUATION SUMMARY
    target_financials = target_data.iloc[0]
    valuation = [{'Method': 'Current Market Price', 'Implied Share Price': target_financials['Price']}]
    
    for stat in ['Median', 'Trimmed Mean (10%)']:
        ev_ebitda_mult = summary_stats.loc[stat, 'EV/EBITDA LTM']
        implied_ev = ev_ebitda_mult * target_financials['EBITDA LTM']
        implied_equity = implied_ev - target_financials['Debt'] + target_financials['Cash']
        valuation.append({'Method': f'{stat} EV/EBITDA', 'Implied Share Price': implied_equity / target_financials['Shares Out']})

        pe_ltm_mult = summary_stats.loc[stat, 'P/E LTM']
        valuation.append({'Method': f'{stat} P/E LTM', 'Implied Share Price': pe_ltm_mult * target_financials['EPS LTM']})

    valuation_df = pd.DataFrame(valuation)

    # 7. GENERATE PRESENTATION-READY EXCEL OUTPUT
    filename = f"{target_ticker}_Advanced_Comps.xlsx"
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        target_data.to_excel(writer, sheet_name='Target Company', index=False)
        peer_data.to_excel(writer, sheet_name='Peer Analysis', index=False)
        summary_stats.transpose().to_excel(writer, sheet_name='Summary Statistics')
        valuation_df.to_excel(writer, sheet_name='Valuation Summary', index=False)
        
        notes_sheet = writer.book.add_worksheet('Notes and Warnings')
        notes_sheet.write('A1', 'Analysis Notes')
        notes_sheet.write('A2', size_warning if size_warning else "Peer group size was large enough for market cap filtering.")
        
        val_sheet = writer.sheets['Valuation Summary']
        chart = writer.book.add_chart({'type': 'bar'})
        chart_df = valuation_df[valuation_df['Implied Share Price'] > 0]
        if not chart_df.empty:
            min_val, max_val = chart_df['Implied Share Price'].min() * 0.9, chart_df['Implied Share Price'].max() * 1.1
            chart.add_series({
                'name': 'Valuation Range', 'categories': f"='Valuation Summary'!$A$2:$A${len(valuation_df)+1}",
                'values': f"='Valuation Summary'!$B$2:$B${len(valuation_df)+1}"})
            chart.set_title({'name': f'{target_ticker} Valuation Football Field'})
            chart.set_x_axis({'name': 'Implied Share Price ($)', 'min': min_val, 'max': max_val})
            chart.set_y_axis({'reverse': True})
            chart.set_legend({'position': 'none'})
            val_sheet.insert_chart('D2', chart)

    logging.info(f"Analysis complete. Output saved to '{filename}'")
    print(f"\n✅ Successfully generated advanced COMPS analysis: '{filename}'")

if __name__ == "__main__":
    TARGET_TICKER = "MSFT" 
    generate_comps_report(TARGET_TICKER)