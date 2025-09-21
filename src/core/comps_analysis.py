import pandas as pd
import yfinance as yf
import logging
import os
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import finnhub
from scipy.stats import trim_mean
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import copy
from datetime import datetime

# --- Local Imports ---
from src.core.database_schema import DATABASE_URL, Company
from src.core.etl_pipeline import (
    generate_distribution_plot, 
    generate_box_plot, 
    generate_pdf_report, 
    generate_comps_report as etl_generate_comps_report
)
# Import the validation engine
from src.validation.validation_engine import ValidationEngine, ValidationResult, ValidationSeverity
from src.validation.validation_reporting import add_validation_tab, create_validation_log_file

# --- Configuration Class ---
@dataclass
class CompsConfig:
    """Configuration parameters for COMPS analysis"""
    
    # Peer Selection
    max_initial_peers: int = 50
    min_peers_required: int = 3
    
    # Business Model Filtering
    enable_segment_filtering: bool = True
    min_segment_overlap_threshold: float = 0.1  # Minimum overlap ratio
    
    # Financial Filtering
    filter_loss_making: bool = True
    require_positive_ebitda: bool = True
    require_positive_revenue: bool = True
    
    # Size Filtering
    enable_size_filtering: bool = True
    min_peers_for_size_filter: int = 8
    size_filter_lower_bound: float = 0.3   # 30% of target market cap
    size_filter_upper_bound: float = 3.0   # 300% of target market cap
    
    # Data Quality
    min_data_completeness: float = 0.7  # 70% of required fields must be present
    max_outlier_threshold: float = 3.0  # Standard deviations for outlier detection
    
    # Statistics
    trimmed_mean_percentage: float = 0.1  # 10% trimmed mean
    
    # Output
    excel_filename_template: str = "{ticker}_Advanced_Comps.xlsx"
    include_debug_sheet: bool = False

# Initialize default configuration
DEFAULT_CONFIG = CompsConfig()

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

def get_finnhub_peers(target_ticker, client, max_peers=50):
    """
    Gets a list of peer tickers from the Finnhub API with enhanced error handling.
    
    Args:
        target_ticker (str): The target company ticker
        client: Finnhub client instance
        max_peers (int): Maximum number of peers to return
        
    Returns:
        list: List of peer ticker symbols (excluding target ticker)
    """
    if not client: 
        logging.warning("Finnhub client not available")
        return []
    
    try:
        logging.info(f"Fetching peers for {target_ticker} from Finnhub API...")
        peers = client.company_peers(target_ticker)
        
        if not peers:
            logging.warning(f"No peers returned from Finnhub for {target_ticker}")
            return []
            
        # Remove target ticker from peers list if present
        clean_peers = [p for p in peers if p != target_ticker]
        
        if not clean_peers:
            logging.warning(f"No valid peers found for {target_ticker} (only target ticker returned)")
            return []
            
        # Limit the number of peers to avoid excessive API calls
        if len(clean_peers) > max_peers:
            clean_peers = clean_peers[:max_peers]
            logging.info(f"Limited peer list to {max_peers} companies")
            
        logging.info(f"Successfully fetched {len(clean_peers)} peers from Finnhub for {target_ticker}")
        return clean_peers
        
    except Exception as e:
        logging.error(f"Error fetching peers from Finnhub for {target_ticker}: {str(e)}")
        return []

def get_database_peers(target_ticker, session, max_peers=50):
    """
    Gets peer companies from the local database based on industry/sector matching.
    
    Args:
        target_ticker (str): The target company ticker
        session: SQLAlchemy session
        max_peers (int): Maximum number of peers to return
        
    Returns:
        list: List of peer ticker symbols from database
    """
    try:
        # First, get the target company's industry/sector info
        target_company = session.query(Company).filter(Company.ticker == target_ticker).first()
        
        if not target_company:
            logging.warning(f"Target ticker {target_ticker} not found in database")
            return []
            
        # Try to find peers by industry first, then by sector
        peers_query = session.query(Company).filter(
            Company.ticker != target_ticker,
            Company.industry == target_company.industry
        )
        
        peers = peers_query.limit(max_peers).all()
        
        # If not enough industry peers, expand to sector
        if len(peers) < 5 and target_company.sector:
            logging.info(f"Found only {len(peers)} industry peers, expanding to sector-level search")
            peers_query = session.query(Company).filter(
                Company.ticker != target_ticker,
                Company.sector == target_company.sector
            )
            peers = peers_query.limit(max_peers).all()
            
        peer_tickers = [p.ticker for p in peers]
        logging.info(f"Found {len(peer_tickers)} database peers for {target_ticker} "
                    f"(Industry: {target_company.industry}, Sector: {target_company.sector})")
        
        return peer_tickers
        
    except Exception as e:
        logging.error(f"Error fetching peers from database for {target_ticker}: {str(e)}")
        return []


def get_fallback_peers(target_ticker):
    """
    Provides hardcoded fallback peer lists for major companies when other methods fail.
    
    Args:
        target_ticker (str): The target company ticker
        
    Returns:
        list: List of fallback peer tickers
    """
    fallback_maps = {
        'MSFT': ['AAPL', 'GOOGL', 'AMZN', 'META', 'ORCL', 'CRM', 'ADBE', 'INTC', 'IBM', 'NVDA'],
        'AAPL': ['MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'ADBE', 'CRM', 'ORCL', 'IBM'],
        'GOOGL': ['MSFT', 'AAPL', 'AMZN', 'META', 'ORCL', 'CRM', 'ADBE', 'IBM', 'NVDA', 'INTC'],
        'AMZN': ['MSFT', 'AAPL', 'GOOGL', 'META', 'WMT', 'TGT', 'COST', 'HD', 'LOW', 'NVDA'],
        'TSLA': ['F', 'GM', 'TM', 'HMC', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'AAPL'],
        'META': ['GOOGL', 'MSFT', 'AAPL', 'AMZN', 'SNAP', 'PINS', 'TWTR', 'NFLX', 'DIS', 'CRM'],
        # Add more as needed
    }
    
    peers = fallback_maps.get(target_ticker, [])
    if peers:
        logging.info(f"Using fallback peer list for {target_ticker}: {len(peers)} companies")
    
    return peers


def get_revenue_segments(ticker, client):
    """
    Gets revenue segments for a company to enable business model comparison.
    
    Args:
        ticker (str): Company ticker symbol
        client: Finnhub client instance
        
    Returns:
        set: Set of revenue segment names
    """
    if not client: 
        logging.debug(f"No Finnhub client available for revenue segments lookup for {ticker}")
        return set()
        
    try:
        logging.debug(f"Fetching revenue segments for {ticker}")
        segments = client.company_revenue_breakdown(symbol=ticker)
        
        if segments and 'breakdown' in segments and segments['breakdown']:
            segment_names = {item['segment'] for item in segments['breakdown'] if item.get('segment')}
            logging.debug(f"Found {len(segment_names)} revenue segments for {ticker}")
            return segment_names
        else:
            logging.debug(f"No revenue segments found for {ticker}")
            return set()
            
    except Exception as e:
        logging.warning(f"Could not fetch revenue segments for {ticker}: {str(e)}")
        return set()


def filter_peers_by_segments(target_ticker: str, peer_tickers: List[str], 
                           client, config: CompsConfig = DEFAULT_CONFIG) -> List[str]:
    """
    Filter peers based on business model similarity using revenue segments.
    
    Args:
        target_ticker: Target company ticker
        peer_tickers: List of potential peer tickers
        client: Finnhub client
        config: Configuration object
        
    Returns:
        Filtered list of peer tickers
    """
    if not config.enable_segment_filtering or not client:
        logging.info("Segment filtering disabled or no Finnhub client available")
        return peer_tickers
        
    logging.info(f"Applying business segment filtering for {target_ticker}")
    
    target_segments = get_revenue_segments(target_ticker, client)
    if not target_segments:
        logging.warning(f"No revenue segments found for {target_ticker}, skipping segment filtering")
        return peer_tickers
        
    logging.info(f"Target company segments: {target_segments}")
    
    filtered_peers = []
    segment_analysis = []
    
    for peer in peer_tickers:
        peer_segments = get_revenue_segments(peer, client)
        
        if not peer_segments:
            # If no segment data available, include peer but note it
            filtered_peers.append(peer)
            segment_analysis.append({
                'ticker': peer,
                'overlap_ratio': 0.0,
                'shared_segments': set(),
                'status': 'no_segment_data'
            })
            continue
            
        # Calculate overlap
        shared_segments = target_segments.intersection(peer_segments)
        overlap_ratio = len(shared_segments) / len(target_segments.union(peer_segments))
        
        segment_analysis.append({
            'ticker': peer,
            'overlap_ratio': overlap_ratio,
            'shared_segments': shared_segments,
            'status': 'included' if overlap_ratio >= config.min_segment_overlap_threshold else 'filtered'
        })
        
        if overlap_ratio >= config.min_segment_overlap_threshold:
            filtered_peers.append(peer)
            
    # Log filtering results
    included_count = len(filtered_peers)
    excluded_count = len(peer_tickers) - included_count
    
    logging.info(f"Segment filtering results: {included_count} included, {excluded_count} excluded")
    
    if excluded_count > 0:
        excluded_peers = [analysis for analysis in segment_analysis if analysis['status'] == 'filtered']
        logging.debug(f"Excluded peers: {[p['ticker'] for p in excluded_peers]}")
        
    return filtered_peers


def filter_by_data_completeness(peer_data: pd.DataFrame, config: CompsConfig) -> pd.DataFrame:
    """
    Filters peers based on data completeness.

    Args:
        peer_data: DataFrame with peer financial data.
        config: Configuration object.

    Returns:
        Filtered DataFrame.
    """
    if peer_data.empty:
        return peer_data

    # Define the columns that are essential for the analysis
    required_cols = [
        'Market Cap', 'Revenue LTM', 'EBITDA LTM', 'Net Income LTM',
        'EV/Sales LTM', 'EV/EBITDA LTM', 'P/E LTM'
    ]
    
    # Keep only columns that exist in the dataframe to avoid errors
    existing_required_cols = [col for col in required_cols if col in peer_data.columns]
    
    if not existing_required_cols:
        logging.warning("None of the required columns for completeness check are present.")
        return peer_data

    # Calculate the completeness for each row
    completeness = peer_data[existing_required_cols].notna().mean(axis=1)
    
    # Filter rows that meet the minimum completeness threshold
    filtered_peers = peer_data[completeness >= config.min_data_completeness]
    
    initial_count = len(peer_data)
    final_count = len(filtered_peers)
    
    if final_count < initial_count:
        logging.info(f"Filtered out {initial_count - final_count} peers due to data incompleteness "
                     f"(threshold: {config.min_data_completeness:.0%}).")
        
    return filtered_peers


def filter_peers_by_financials(peer_data: pd.DataFrame, target_sector: str, 
                                config: CompsConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """
    Filter peer companies based on financial health and data quality, with industry-specific logic.
    
    Args:
        peer_data: DataFrame with peer financial data
        target_sector: Sector of the target company
        config: Configuration object
        
    Returns:
        Filtered DataFrame
    """
    initial_count = len(peer_data)
    
    # --- Determine if EBITDA filtering should be applied ---
    apply_ebitda_filter = config.require_positive_ebitda
    # Use .startswith() for a more robust check (e.g., 'Financials', 'Financial Services')
    if target_sector and target_sector.lower().startswith('financial'):
        logging.info(f"Target sector '{target_sector}' is financial. Skipping EBITDA-based filtering.")
        apply_ebitda_filter = False
    
    # --- Financial Health Filtering ---
    if apply_ebitda_filter:
        # Make a copy to avoid SettingWithCopyWarning
        peer_data = peer_data.copy()
        peer_data.loc[:, 'EBITDA LTM'] = pd.to_numeric(peer_data['EBITDA LTM'], errors='coerce')
        peer_data = peer_data[peer_data['EBITDA LTM'] > 0]
        logging.info(f"Filtered out loss-making companies (EBITDA): {initial_count} -> {len(peer_data)}")
    
    if config.filter_loss_making:
        # Make a copy to avoid SettingWithCopyWarning
        peer_data = peer_data.copy()
        peer_data.loc[:, 'Net Income LTM'] = pd.to_numeric(peer_data['Net Income LTM'], errors='coerce')
        peer_data = peer_data[peer_data['Net Income LTM'] > 0]
        logging.info(f"Filtered out companies with negative net income: -> {len(peer_data)}")
        
    if config.require_positive_revenue:
        peer_data = peer_data.copy()
        peer_data.loc[:, 'Revenue LTM'] = pd.to_numeric(peer_data['Revenue LTM'], errors='coerce')
        peer_data = peer_data[peer_data['Revenue LTM'] > 0]
        logging.info(f"Filtered out companies with no revenue: -> {len(peer_data)}")
        
    # --- Data Completeness Filtering ---
    filtered_data = filter_by_data_completeness(peer_data, config)
    
    if len(filtered_data) < config.min_peers_required and len(peer_data) >= config.min_peers_required:
        warning_msg = ("Data completeness filter would reduce peers below minimum (3). "
                       "Relaxing completeness requirement.")
        logging.warning(warning_msg)
        return peer_data
    
    logging.info(f"Financial filtering complete: {initial_count} -> {len(filtered_data)} peers remaining")
    return filtered_data


def filter_peers_by_size(target_data: pd.Series, peer_data: pd.DataFrame, 
                        config: CompsConfig = DEFAULT_CONFIG) -> tuple[pd.DataFrame, str]:
    """
    Apply adaptive size filtering based on market capitalization.
    
    Args:
        target_data: Series with target company data
        peer_data: DataFrame with peer data
        config: Configuration object
        
    Returns:
        A tuple containing the filtered DataFrame and a potential warning message.
    """
    if peer_data.empty:
        return peer_data, ""
        
    initial_count = len(peer_data)
    warning_msg = ""
    
    # Ensure 'Market Cap' column is numeric and handle missing values
    if 'Market Cap' not in peer_data.columns:
        warning_msg = "Size filtering skipped: 'Market Cap' column not found in peer data."
        logging.warning(warning_msg)
        return peer_data, warning_msg
        
    peer_data['Market Cap'] = pd.to_numeric(peer_data['Market Cap'], errors='coerce')
    valid_market_cap_peers = peer_data.dropna(subset=['Market Cap'])
    
    if valid_market_cap_peers.empty:
        warning_msg = "Size filtering skipped: no peers with valid Market Cap data."
        logging.warning(warning_msg)
        return peer_data, warning_msg

    # Only apply size filtering if we have enough peers
    if not config.enable_size_filtering or len(valid_market_cap_peers) < config.min_peers_for_size_filter:
        warning_msg = (f"Size filtering skipped: only {len(valid_market_cap_peers)} peers with valid market caps available "
                      f"(minimum {config.min_peers_for_size_filter} required). "
                      f"Results may be affected by size disparities.")
        logging.warning(warning_msg)
        return peer_data, warning_msg
    
    target_market_cap = target_data.get('Market Cap')
    if pd.isna(target_market_cap) or target_market_cap <= 0:
        warning_msg = "Cannot apply size filtering: target company market cap unavailable."
        logging.warning(warning_msg)
        return peer_data, warning_msg
    
    # --- Initial Filtering ---
    lower_bound = target_market_cap * config.size_filter_lower_bound
    upper_bound = target_market_cap * config.size_filter_upper_bound
    
    size_filtered = valid_market_cap_peers[
        (valid_market_cap_peers['Market Cap'] >= lower_bound) & 
        (valid_market_cap_peers['Market Cap'] <= upper_bound)
    ]
    
    logging.info(f"Initial size filtering ({config.size_filter_lower_bound:.0%}-{config.size_filter_upper_bound:.0%}) "
                 f"yielded {len(size_filtered)} peers from {len(valid_market_cap_peers)}.")
    
    # --- Adaptive Expansion if Needed ---
    if len(size_filtered) < config.min_peers_required:
        logging.warning(f"Size filtering reduced peers to {len(size_filtered)}, below the minimum of "
                        f"{config.min_peers_required}. Expanding size criteria.")
        
        expansion_factor = 1.5
        while len(size_filtered) < config.min_peers_required and expansion_factor <= 5.0:
            expanded_lower = target_market_cap * (config.size_filter_lower_bound / expansion_factor)
            expanded_upper = target_market_cap * (config.size_filter_upper_bound * expansion_factor)
            
            size_filtered = valid_market_cap_peers[
                (valid_market_cap_peers['Market Cap'] >= expanded_lower) & 
                (valid_market_cap_peers['Market Cap'] <= expanded_upper)
            ]
            logging.info(f"Expanded size criteria (factor: {expansion_factor:.1f}x). "
                         f"New peer count: {len(size_filtered)}")
            expansion_factor += 0.5
            
        if len(size_filtered) < config.min_peers_required:
            warning_msg = (f"Could not find enough peers ({len(size_filtered)}) even after expanding "
                           f"size criteria up to {expansion_factor-0.5:.1f}x. Using the expanded set.")
            logging.warning(warning_msg)

    final_count = len(size_filtered)
    logging.info(f"Size filtering complete: {initial_count} -> {final_count} peers remaining.")
    
    # Return the filtered dataframe, keeping original indices
    return size_filtered, warning_msg

def get_comps_data(tickers: List[str]) -> pd.DataFrame:
    """
    Fetches financial data for a list of tickers from yfinance.
    Performs data validation immediately after ingestion to ensure data integrity.
    """
    if not tickers:
        return pd.DataFrame()

    logging.info(f"Fetching data for {len(tickers)} companies from yfinance...")
    
    data = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Basic data
            market_cap = info.get('marketCap', 0)
            revenue = info.get('totalRevenue', 0)
            ebitda = info.get('ebitda', 0)
            net_income = info.get('netIncomeToCommon', 0)
            
            # Enterprise Value components
            ev = info.get('enterpriseValue', 0)
            debt = info.get('totalDebt', 0)
            cash = info.get('totalCash', 0)
            
            # Price and shares
            price = info.get('currentPrice', info.get('previousClose'))
            shares_out = info.get('sharesOutstanding', 0)
            
            # Forward estimates
            eps_forward = info.get('forwardEps', 0)
            pe_forward = info.get('forwardPE', 0)

            # Sector and Industry
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')

            data.append({
                'Ticker': ticker,
                'Company Name': info.get('shortName', ticker),
                'Sector': sector,
                'Industry': industry,
                'Price': price,
                'Market Cap': market_cap,
                'EV': ev,
                'Revenue LTM': revenue,
                'EBITDA LTM': ebitda,
                'Net Income LTM': net_income,
                'Debt': debt,
                'Cash': cash,
                'Shares Out': shares_out,
                'EPS Forward': eps_forward,
                'P/E Forward': pe_forward,
            })
        except Exception as e:
            logging.warning(f"Could not fetch data for {ticker}: {e}")
            continue
    
    # Create the DataFrame from collected data
    result_df = pd.DataFrame(data)
    
    if not result_df.empty:
        # Add recalculated columns for more robust validation
        result_df['Market Cap (Recalculated)'] = result_df.apply(
            lambda row: row['Price'] * row['Shares Out'] if not pd.isna(row['Price']) and not pd.isna(row['Shares Out']) else np.nan, 
            axis=1
        )
        
        result_df['EV (Recalculated)'] = result_df.apply(
            lambda row: row['Market Cap'] + row['Debt'] - row['Cash'] 
            if not pd.isna(row['Market Cap']) and not pd.isna(row['Debt']) and not pd.isna(row['Cash']) else np.nan, 
            axis=1
        )
        
        # Calculate multiples directly to ensure data consistency
        result_df = calculate_multiples(result_df)
        
        # Initialize validation engine
        validator = ValidationEngine()
        
        # Validate each company's data
        validation_results = {}
        failed_validations = []
        
        for _, row in result_df.iterrows():
            ticker = row['Ticker']
            validation_result = validator.validate_company_data(row)
            validation_results[ticker] = validation_result
            
            # If validation failed with blocking issues, log and track
            if not validation_result.passed:
                logging.error(f"Validation failed for {ticker} with {len(validation_result.issues)} issues")
                for issue in validation_result.issues:
                    if issue.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL):
                        logging.error(f"  - {issue.metric}: expected={issue.expected:.2f}, actual={issue.actual:.2f}, "
                                    f"diff={issue.pct_difference:.2%}, threshold={issue.threshold:.2%}")
                failed_validations.append(ticker)
                
                # Save validation results to file for further investigation
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                validation_result.save_to_file(f"validation_results/{ticker}_validation_{timestamp}.json")
        
        # If there are blocking validation failures, warn but don't stop processing
        # This allows users to review the data but ensures analysis flags the issues
        if failed_validations:
            logging.warning(f"Data validation failed for {len(failed_validations)} companies: {', '.join(failed_validations)}")
            logging.warning("Analysis will proceed but results may be unreliable. See validation_results/ directory for details.")
            
            # Add a validation flag column to the DataFrame
            result_df['validation_passed'] = result_df['Ticker'].apply(
                lambda t: validation_results.get(t, ValidationResult(company_ticker=t, passed=False)).passed
            )
        else:
            logging.info("Data validation passed for all companies")
    
    return result_df

def calculate_multiples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates valuation multiples for the given DataFrame.
    """
    df['EV/Sales LTM'] = df['EV'] / df['Revenue LTM']
    df['EV/EBITDA LTM'] = df['EV'] / df['EBITDA LTM']
    df['P/E LTM'] = df['Market Cap'] / df['Net Income LTM']
    
    # Replace inf/-inf with NaN
    df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
    
    return df

def generate_comps_report(target_ticker: str, db_session, finnhub_client, config: CompsConfig = DEFAULT_CONFIG):
    """
    Main function to generate the comprehensive COMPS analysis report.
    """
    warnings = []
    
    # Phase 1: Peer Identification
    logging.info(f"=== PHASE 1: PEER IDENTIFICATION FOR {target_ticker} ===")
    finnhub_peers = get_finnhub_peers(target_ticker, finnhub_client, config.max_initial_peers)
    db_peers = get_database_peers(target_ticker, db_session, config.max_initial_peers)
    fallback_peers = get_fallback_peers(target_ticker)
    
    # Combine and de-duplicate peer lists
    initial_peers = list(dict.fromkeys(finnhub_peers + db_peers + fallback_peers))
    if not initial_peers:
        logging.error(f"Could not find any peers for {target_ticker}. Aborting analysis.")
        return

    logging.info(f"Found {len(initial_peers)} unique initial peers.")

    # Phase 2: Data Gathering
    logging.info("=== PHASE 2: DATA GATHERING ===")
    all_tickers = [target_ticker] + initial_peers
    all_data = get_comps_data(all_tickers)

    if all_data.empty or target_ticker not in all_data['Ticker'].values:
        logging.error(f"Failed to fetch data for target {target_ticker}. Aborting.")
        return

    # Phase 2b: Data Validation
    logging.info("=== PHASE 2B: DATA VALIDATION ===")
    # Check if validation results are present in the DataFrame
    if 'validation_passed' in all_data.columns:
        target_validation_passed = all_data.loc[all_data['Ticker'] == target_ticker, 'validation_passed'].iloc[0]
        if not target_validation_passed:
            validation_warning = f"VALIDATION WARNING: Target company {target_ticker} data validation failed. Results may be unreliable."
            logging.warning(validation_warning)
            warnings.append(validation_warning)
            
        # Count failed validations in peer group
        failed_validations_count = all_data[all_data['Ticker'] != target_ticker]['validation_passed'].value_counts().get(False, 0)
        if failed_validations_count > 0:
            validation_warning = f"VALIDATION WARNING: {failed_validations_count} peer companies failed data validation. Results may be affected."
            logging.warning(validation_warning)
            warnings.append(validation_warning)
            
            # If the target validation failed and we have significant peer failures, consider halting
            if not target_validation_passed and failed_validations_count > len(initial_peers) * 0.25:  # More than 25% failed
                halt_warning = ("CRITICAL VALIDATION ISSUE: Target company and significant portion of peer group "
                               "failed validation. Processing halted to prevent unreliable results.")
                logging.error(halt_warning)
                warnings.append(halt_warning)
                return None, warnings
    
    target_data = all_data[all_data['Ticker'] == target_ticker].iloc[0]
    peer_data = all_data[all_data['Ticker'] != target_ticker].copy()

    # Phase 3: Peer Filtering
    logging.info("=== PHASE 3: PEER FILTERING ===")
    
    # 3a: Business Model Filtering
    filtered_peers_business = filter_peers_by_segments(target_ticker, peer_data['Ticker'].tolist(), finnhub_client, config)
    peer_data = peer_data[peer_data['Ticker'].isin(filtered_peers_business)]

    # 3b: Financial Health Filtering
    peer_data = filter_peers_by_financials(peer_data, target_data['Sector'], config)

    # 3c: Size Filtering
    peer_data, size_warning = filter_peers_by_size(target_data, peer_data, config)
    if size_warning:
        warnings.append(size_warning)

    if len(peer_data) < config.min_peers_required:
        logging.error(f"Insufficient peers ({len(peer_data)}) after filtering. Minimum required is {config.min_peers_required}. Aborting.")
        return None, warnings

    logging.info(f"Final peer group contains {len(peer_data)} companies.")

    # Phase 4: Calculate Multiples for the final peer group
    logging.info("=== PHASE 4: CALCULATING MULTIPLES ===")
    peer_data = calculate_multiples(peer_data)
    
    # Phase 5: Statistical Analysis
    summary_stats = calculate_robust_statistics(peer_data, config)
    if summary_stats.empty:
        logging.error("Statistics calculation failed. Aborting.")
        return None, warnings

    # Phase 6: Valuation
    valuation_df = calculate_comprehensive_valuation(target_data, summary_stats, config)

    # Phase 7: Report Generation
    logging.info("=== PHASE 7: GENERATING REPORT ===")
    try:
        report_filename = generate_excel_report(
            target_ticker, target_data, peer_data, summary_stats, valuation_df, warnings, config
        )
        logging.info(f"Successfully generated report: {report_filename}")
        return report_filename, warnings
    except Exception as e:
        logging.error(f"Failed to generate Excel report: {e}")
        return None, warnings


def calculate_robust_statistics(peer_data: pd.DataFrame, config: CompsConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """
    Calculate comprehensive statistics for valuation multiples with outlier detection.
    
    Args:
        peer_data: DataFrame with peer financial data
        config: Configuration object
        
    Returns:
        DataFrame with statistics for each multiple
    """
    import numpy as np
    from scipy import stats
    
    multiples = ['EV/Sales LTM', 'EV/EBITDA LTM', 'P/E LTM', 'P/E Forward']
    
    if peer_data.empty:
        logging.error("No peer data available for statistics calculation")
        return pd.DataFrame()
    
    logging.info("=== PHASE 4: STATISTICAL ANALYSIS ===")
    logging.info(f"Calculating robust statistics for {len(peer_data)} peers across {len(multiples)} multiples")
    
    # Initialize results dictionary
    stats_results = {}
    outlier_summary = {}
    
    for multiple in multiples:
        logging.info(f"Processing {multiple}...")
        
        # Get clean data (remove NaN and infinite values)
        clean_data = peer_data[multiple].dropna()
        
        # Convert to numeric and handle any non-numeric values
        clean_data = pd.to_numeric(clean_data, errors='coerce').dropna()
        
        # Remove infinite and negative values
        clean_data = clean_data[np.isfinite(clean_data.astype(float))]
        clean_data = clean_data[clean_data > 0]  # Remove negative multiples
        
        if len(clean_data) == 0:
            logging.warning(f"No valid data for {multiple}")
            stats_results[multiple] = {
                'Count': 0, 'Mean': np.nan, 'Median': np.nan, 'Std Dev': np.nan,
                'Min': np.nan, 'Max': np.nan, '25th Percentile': np.nan, '75th Percentile': np.nan,
                'Trimmed Mean (10%)': np.nan, 'Weighted Average': np.nan,
                'Outliers Removed': 0, 'Data Quality Score': 0.0
            }
            continue
            
        # Outlier detection using IQR method
        Q1 = clean_data.quantile(0.25)
        Q3 = clean_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Additional outlier detection using Z-score
        z_scores = np.abs(stats.zscore(clean_data))
        z_outliers = clean_data[z_scores > config.max_outlier_threshold]
        
        # Combine outlier detection methods
        iqr_outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
        all_outliers = list(set(iqr_outliers.index).union(set(z_outliers.index)))
        
        # Remove outliers for robust statistics
        robust_data = clean_data.drop(index=all_outliers)
        
        if len(robust_data) < 2:
            logging.warning(f"Too few data points after outlier removal for {multiple}")
            robust_data = clean_data  # Use original data if too few points remain
            outliers_removed = 0
        else:
            outliers_removed = len(all_outliers)
            
        logging.info(f"{multiple}: {len(clean_data)} valid points, {outliers_removed} outliers removed")
        
        # Calculate comprehensive statistics
        try:
            # Basic statistics
            mean_val = robust_data.mean()
            median_val = robust_data.median()
            std_val = robust_data.std()
            min_val = robust_data.min()
            max_val = robust_data.max()
            
            # Percentiles
            pct_25 = robust_data.quantile(0.25)
            pct_75 = robust_data.quantile(0.75)
            
            # Trimmed mean (removes top and bottom X%)
            trimmed_mean_val = trim_mean(robust_data, config.trimmed_mean_percentage)
            
            # Weighted average by market cap
            if 'Market Cap' in peer_data.columns:
                # Get market caps for the robust data points
                robust_indices = robust_data.index
                peer_subset = peer_data.loc[robust_indices]
                weights = peer_subset['Market Cap']
                weighted_avg = (robust_data * weights).sum() / weights.sum()
            else:
                weighted_avg = mean_val  # Fallback to simple mean
                
            # Data quality score
            data_quality = len(robust_data) / len(peer_data)
            
            stats_results[multiple] = {
                'Count': len(robust_data),
                'Mean': mean_val,
                'Median': median_val,
                'Std Dev': std_val,
                'Min': min_val,
                'Max': max_val,
                '25th Percentile': pct_25,
                '75th Percentile': pct_75,
                f'Trimmed Mean ({config.trimmed_mean_percentage*100:.0f}%)': trimmed_mean_val,
                'Weighted Average': weighted_avg,
                'Outliers Removed': outliers_removed,
                'Data Quality Score': data_quality
            }
            
            # Store outlier information
            if outliers_removed > 0:
                outlier_tickers = peer_data.loc[all_outliers, 'Ticker'].tolist() if 'Ticker' in peer_data.columns else list(all_outliers)
                outlier_values = clean_data.loc[all_outliers].tolist()
                outlier_summary[multiple] = {
                    'outlier_tickers': outlier_tickers,
                    'outlier_values': outlier_values,
                    'bounds': {'lower': lower_bound, 'upper': upper_bound}
                }
                
        except Exception as e:
            logging.error(f"Error calculating statistics for {multiple}: {str(e)}")
            stats_results[multiple] = {col: np.nan for col in ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', '25th Percentile', '75th Percentile', f'Trimmed Mean ({config.trimmed_mean_percentage*100:.0f}%)', 'Weighted Average', 'Outliers Removed', 'Data Quality Score']}
    
    # Convert to DataFrame
    summary_stats_df = pd.DataFrame(stats_results).T
    
    # Log summary
    logging.info("Statistical analysis complete:")
    for multiple in multiples:
        if multiple in summary_stats_df.index:
            row = summary_stats_df.loc[multiple]
            logging.info(f"  {multiple}: Median={row['Median']:.2f}, Mean={row['Mean']:.2f}, "
                        f"Count={row['Count']}, Quality={row['Data Quality Score']:.1%}")
    
    # Store outlier information for reporting (using a different approach to avoid pandas warning)
    summary_stats_df.attrs['outlier_summary'] = outlier_summary
    
    return summary_stats_df
    
def calculate_comprehensive_valuation(target_data: pd.Series, summary_stats: pd.DataFrame, 
                                     config: CompsConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """
    Calculate comprehensive valuation scenarios using multiple statistical methods.
    
    Args:
        target_data: Series with target company financial data
        summary_stats: DataFrame with statistical analysis results
        config: Configuration object
        
    Returns:
        DataFrame with valuation scenarios and implied share prices
    """
    import numpy as np
    
    logging.info("=== PHASE 5: VALUATION CALCULATION ===")
    
    valuation_results = []
    
    # Current market valuation as benchmark
    current_price = target_data.get('Price', 0)
    if current_price and current_price > 0:
        valuation_results.append({
            'Method': 'Current Market Price',
            'Multiple Used': 'N/A',
            'Multiple Value': 'N/A',
            'Implied Enterprise Value': target_data.get('EV', 0),
            'Implied Equity Value': target_data.get('Market Cap', 0),
            'Implied Share Price': current_price,
            'Premium/Discount to Current': '0.0%',
            'Confidence Level': 'Market',
            'Data Quality': 'Current'
        })
    
    # Statistical methods to apply
    stat_methods = ['Median', 'Mean', f'Trimmed Mean ({config.trimmed_mean_percentage*100:.0f}%)', 'Weighted Average']
    
    # Valuation multiples and their corresponding target metrics
    valuation_approaches = {
        'EV/Sales LTM': {
            'target_metric': 'Revenue LTM',
            'calculation_type': 'ev_based',
            'description': 'Revenue Multiple'
        },
        'EV/EBITDA LTM': {
            'target_metric': 'EBITDA LTM',
            'calculation_type': 'ev_based',
            'description': 'EBITDA Multiple'
        },
        'P/E LTM': {
            'target_metric': 'Net Income LTM',
            'calculation_type': 'equity_based',
            'description': 'Trailing P/E'
        },
        'P/E Forward': {
            'target_metric': 'EPS Forward',
            'calculation_type': 'per_share',
            'description': 'Forward P/E'
        }
    }
    
    for multiple_name, approach_info in valuation_approaches.items():
        if multiple_name not in summary_stats.index:
            logging.warning(f"Statistics not available for {multiple_name}")
            continue
            
        target_metric_value = target_data.get(approach_info['target_metric'], 0)
        
        if not target_metric_value or target_metric_value <= 0:
            logging.warning(f"Target company {approach_info['target_metric']} not available or non-positive")
            continue
            
        multiple_stats = summary_stats.loc[multiple_name]
        data_quality_score = multiple_stats.get('Data Quality Score', 0)
        
        for stat_method in stat_methods:
            if stat_method not in multiple_stats or pd.isna(multiple_stats[stat_method]):
                continue
                
            multiple_value = multiple_stats[stat_method]
            
            try:
                # Calculate valuation based on approach type
                if approach_info['calculation_type'] == 'ev_based':
                    # Enterprise Value based calculation
                    implied_ev = multiple_value * target_metric_value
                    implied_equity_value = implied_ev - target_data.get('Debt', 0) + target_data.get('Cash', 0)
                    implied_share_price = implied_equity_value / target_data.get('Shares Out', 1)
                    
                elif approach_info['calculation_type'] == 'equity_based':
                    # Direct equity value calculation
                    implied_equity_value = multiple_value * target_metric_value
                    implied_ev = implied_equity_value + target_data.get('Debt', 0) - target_data.get('Cash', 0)
                    implied_share_price = implied_equity_value / target_data.get('Shares Out', 1)
                    
                elif approach_info['calculation_type'] == 'per_share':
                    # Per-share calculation (like P/E with EPS)
                    implied_share_price = multiple_value * target_metric_value
                    implied_equity_value = implied_share_price * target_data.get('Shares Out', 1)
                    implied_ev = implied_equity_value + target_data.get('Debt', 0) - target_data.get('Cash', 0)
                    
                # Calculate premium/discount to current price
                if current_price > 0:
                    premium_discount = (implied_share_price - current_price) / current_price
                    premium_discount_str = f"{premium_discount:+.1%}"
                else:
                    premium_discount_str = "N/A"
                    
                # Determine confidence level based on data quality and statistical method
                if data_quality_score >= 0.8 and stat_method in ['Median', 'Trimmed Mean (10%)']:
                    confidence = 'High'
                elif data_quality_score >= 0.6:
                    confidence = 'Medium'
                else:
                    confidence = 'Low'
                    
                valuation_results.append({
                    'Method': f"{stat_method} {approach_info['description']}",
                    'Multiple Used': multiple_name,
                    'Multiple Value': f"{multiple_value:.2f}x",
                    'Implied Enterprise Value': implied_ev,
                    'Implied Equity Value': implied_equity_value,
                    'Implied Share Price': implied_share_price,
                    'Premium/Discount to Current': premium_discount_str,
                    'Confidence Level': confidence,
                    'Data Quality': f"{data_quality_score:.1%}"
                })
                
            except Exception as e:
                logging.error(f"Error calculating valuation for {multiple_name} {stat_method}: {str(e)}")
                continue
    
    valuation_df = pd.DataFrame(valuation_results)
    
    if not valuation_df.empty:
        # Calculate summary statistics for the valuation range
        share_prices = valuation_df[valuation_df['Method'] != 'Current Market Price']['Implied Share Price'].dropna()
        
        if len(share_prices) > 0:
            valuation_summary = {
                'Min Implied Price': share_prices.min(),
                'Max Implied Price': share_prices.max(),
                'Mean Implied Price': share_prices.mean(),
                'Median Implied Price': share_prices.median(),
                'Current Price': current_price,
                'Valuation Methods': len(share_prices)
            }
            
            logging.info("Valuation Summary:")
            for key, value in valuation_summary.items():
                if isinstance(value, (int, float)) and key != 'Valuation Methods':
                    logging.info(f"  {key}: ${value:.2f}")
                else:
                    logging.info(f"  {key}: {value}")
    
    return valuation_df
    
def generate_excel_report(target_ticker: str, target_data: pd.Series, peer_data: pd.DataFrame, 
                         summary_stats: pd.DataFrame, valuation_df: pd.DataFrame, 
                         warnings: List[str], config: CompsConfig = DEFAULT_CONFIG) -> str:
    """
    Generate a comprehensive, professional Excel workbook with multiple sheets.
    
    Args:
        target_ticker: Target company ticker
        target_data: Target company financial data
        peer_data: Peer group data
        summary_stats: Statistical analysis results
        valuation_df: Valuation scenarios
        warnings: List of analysis warnings
        config: Configuration object
        
    Returns:
        Filename of the generated Excel report
    """
    logging.info("=== PHASE 6: EXCEL REPORT GENERATION ===")

    # Ensure the comps_analysis folder exists
    output_dir = "comps_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename and include the folder path
    filename = os.path.join(output_dir, config.excel_filename_template.format(ticker=target_ticker))

    try:
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            workbook = writer.book

            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'font_size': 12,
                'bg_color': '#4F81BD',
                'font_color': 'white',
                'border': 1,
                'align': 'center',
                'valign': 'vcenter'
            })

            currency_format = workbook.add_format({
                'num_format': '$#,##0',
                'border': 1,
                'align': 'right'
            })

            number_format = workbook.add_format({
                'num_format': '#,##0',
                'border': 1,
                'align': 'right'
            })

            percentage_format = workbook.add_format({
                'num_format': '0.0%',
                'border': 1,
                'align': 'right'
            })

            multiple_format = workbook.add_format({
                'num_format': '0.0x',
                'border': 1,
                'align': 'right'
            })

            title_format = workbook.add_format({
                'bold': True,
                'font_size': 16,
                'font_color': '#4F81BD'
            })

            # Sheet 1: Target Company Overview
            logging.info("Creating Target Company sheet...")
            target_df = pd.DataFrame([target_data]).T
            target_df.columns = ['Value']
            target_df.to_excel(writer, sheet_name='Target Company', startrow=2)
            
            target_sheet = writer.sheets['Target Company']
            target_sheet.write('A1', f'{target_ticker} - Target Company Analysis', title_format)
            
            # Format the target company data
            target_sheet.set_column('A:A', 25)
            target_sheet.set_column('B:B', 15)
            
            # Apply formatting based on data type
            financial_fields = ['Market Cap', 'EV', 'Revenue LTM', 'EBITDA LTM', 'Net Income LTM', 'Debt', 'Cash']
            
            for i, (field, value) in enumerate(target_df.iterrows(), start=4):
                if field in financial_fields:
                    target_sheet.write(f'B{i}', value['Value'], currency_format)
                elif 'Price' in field or 'EPS' in field:
                    target_sheet.write(f'B{i}', value['Value'], currency_format)
                elif 'Shares' in field:
                    target_sheet.write(f'B{i}', value['Value'], number_format)
            
            # Sheet 2: Peer Analysis
            logging.info("Creating Peer Analysis sheet...")
            
            peer_data_formatted = peer_data.copy()
            peer_data_formatted.to_excel(writer, sheet_name='Peer Analysis', index=False, startrow=2)
            
            peer_sheet = writer.sheets['Peer Analysis']
            peer_sheet.write('A1', f'{target_ticker} - Peer Group Analysis', title_format)
            
            # Format peer analysis headers
            for col_num, column in enumerate(peer_data_formatted.columns):
                peer_sheet.write(2, col_num, column, header_format)
            
            # Auto-fit columns and apply formatting
            for col_num, column in enumerate(peer_data_formatted.columns):
                if 'Market Cap' in column or 'EV' in column or 'Revenue' in column or 'EBITDA' in column or 'Income' in column or 'Debt' in column or 'Cash' in column:
                    peer_sheet.set_column(col_num, col_num, 12, currency_format)
                elif 'Price' in column or 'EPS' in column:
                    peer_sheet.set_column(col_num, col_num, 10, currency_format)
                elif 'Shares' in column:
                    peer_sheet.set_column(col_num, col_num, 12, number_format)
                elif '/' in column:  # Multiples
                    peer_sheet.set_column(col_num, col_num, 10, multiple_format)
                else:
                    peer_sheet.set_column(col_num, col_num, 15)
            
            # Sheet 3: Summary Statistics
            logging.info("Creating Summary Statistics sheet...")
            summary_stats.to_excel(writer, sheet_name='Summary Statistics', startrow=2)
            
            stats_sheet = writer.sheets['Summary Statistics']
            stats_sheet.write('A1', f'{target_ticker} - Peer Group Statistics', title_format)
            
            # Format statistics headers
            for col_num, column in enumerate(['Metric'] + list(summary_stats.columns)):
                stats_sheet.write(2, col_num, column, header_format)
            
            # Format statistics data
            for col_num in range(1, len(summary_stats.columns) + 1):
                stats_sheet.set_column(col_num, col_num, 12, multiple_format)
            
            stats_sheet.set_column('A:A', 20)
            
            # Sheet 4: Valuation Summary
            logging.info("Creating Valuation Summary sheet...")
            valuation_df.to_excel(writer, sheet_name='Valuation Summary', index=False, startrow=2)
            
            val_sheet = writer.sheets['Valuation Summary']
            val_sheet.write('A1', f'{target_ticker} - Valuation Analysis', title_format)
            
            # Format valuation headers
            for col_num, column in enumerate(valuation_df.columns):
                val_sheet.write(2, col_num, column, header_format)
            
            # Format valuation data
            for col_num, column in enumerate(valuation_df.columns):
                if 'Price' in column or 'Value' in column:
                    val_sheet.set_column(col_num, col_num, 15, currency_format)
                elif 'Premium' in column:
                    val_sheet.set_column(col_num, col_num, 12, percentage_format)
                else:
                    val_sheet.set_column(col_num, col_num, 20)
            
            # Sheet 5: Notes and Warnings
            logging.info("Creating Notes and Warnings sheet...")
            notes_sheet = writer.book.add_worksheet('Notes and Warnings')
            notes_sheet.write('A1', f'{target_ticker} - Analysis Notes and Warnings', title_format)
            
            row = 3
            notes_sheet.write(row, 0, 'Analysis Methodology:', workbook.add_format({'bold': True}))
            row += 1
            
            methodology_notes = [
                "1. Peer Selection: Multi-tier approach using Finnhub API, database lookup, and fallback peer lists",
                "2. Business Model Filtering: Revenue segment analysis for business model similarity",
                "3. Financial Filtering: Removal of loss-making companies and data quality checks",
                "4. Size Filtering: Market capitalization-based filtering with adaptive thresholds",
                "5. Statistical Analysis: Multiple robust statistical methods with outlier detection",
                "6. Valuation: Multiple methodologies including EV/Revenue, EV/EBITDA, P/E (LTM and Forward)",
                "7. Data Quality: Comprehensive validation and quality scoring throughout the process"
            ]
            
            for note in methodology_notes:
                notes_sheet.write(row, 0, note)
                row += 1
            
            row += 2
            notes_sheet.write(row, 0, 'Analysis Warnings:', workbook.add_format({'bold': True, 'font_color': 'red'}))
            row += 1
            
            if warnings:
                for warning in warnings:
                    notes_sheet.write(row, 0, f"• {warning}", workbook.add_format({'font_color': 'red'}))
                    row += 1
            else:
                notes_sheet.write(row, 0, "No significant warnings generated during analysis.")
                row += 1
            
            row += 2
            notes_sheet.write(row, 0, 'Data Sources:', workbook.add_format({'bold': True}))
            row += 1
            notes_sheet.write(row, 0, "• Real-time financial data: Yahoo Finance API")
            row += 1
            notes_sheet.write(row, 0, "• Peer identification: Finnhub API")
            row += 1
            notes_sheet.write(row, 0, "• Database: Local PostgreSQL database")
            row += 1
            
            # Add outlier information if available
            if hasattr(summary_stats, 'outlier_summary') and summary_stats.outlier_summary:
                row += 2
                notes_sheet.write(row, 0, 'Outliers Removed:', workbook.add_format({'bold': True}))
                row += 1
                
                for multiple, outlier_info in summary_stats.outlier_summary.items():
                    notes_sheet.write(row, 0, f"{multiple}:")
                    row += 1
                    for ticker, value in zip(outlier_info['outlier_tickers'], outlier_info['outlier_values']):
                        notes_sheet.write(row, 1, f"• {ticker}: {value:.2f}x")
                        row += 1
            
            # Auto-fit columns in notes sheet
            notes_sheet.set_column('A:A', 80)
            
            # Optional: Debug sheet with raw data
            if config.include_debug_sheet:
                logging.info("Creating Debug Information sheet...")
                debug_sheet = writer.book.add_worksheet('Debug Info')
                debug_sheet.write('A1', 'Debug Information', title_format)
                # Add configuration and debug information here
                row = 3
                debug_sheet.write(row, 0, 'Configuration Parameters:', workbook.add_format({'bold': True}))
                row += 1
                
                config_dict = {
                    'Max Initial Peers': config.max_initial_peers,
                    'Min Peers Required': config.min_peers_required,
                    'Enable Segment Filtering': config.enable_segment_filtering,
                    'Size Filter Lower Bound': f"{config.size_filter_lower_bound:.1%}",
                    'Size Filter Upper Bound': f"{config.size_filter_upper_bound:.1%}",
                    'Trimmed Mean Percentage': f"{config.trimmed_mean_percentage:.1%}",
                    'Max Outlier Threshold': f"{config.max_outlier_threshold} std dev"
                }
                
                for param, value in config_dict.items():
                    debug_sheet.write(row, 0, param)
                    debug_sheet.write(row, 1, str(value))
                    row += 1
        
            add_football_field_chart(workbook, val_sheet, valuation_df, target_ticker)
        
        logging.info(f"Excel report successfully generated: {filename}")
        return filename
        
    except Exception as e:
        logging.error(f"Error generating Excel report: {e}")
        raise

def add_football_field_chart(workbook, val_sheet, valuation_df: pd.DataFrame, target_ticker: str):
    """
    Add a professional Football Field valuation chart to the Excel workbook.
    
    Args:
        workbook: XlsxWriter workbook object
        val_sheet: Valuation Summary worksheet
        valuation_df: DataFrame with valuation scenarios
        target_ticker: Target company ticker
    """
    try:
        # Filter for valid share prices (exclude current market price row)
        chart_data = valuation_df[
            (valuation_df['Method'] != 'Current Market Price') & 
            (valuation_df['Implied Share Price'].notna()) &
            (valuation_df['Implied Share Price'] > 0)
        ].copy()
        
        if chart_data.empty:
            logging.warning("No valid data for Football Field chart")
            return
            
        # --- Create the main bar chart for valuation ranges ---
        bar_chart = workbook.add_chart({'type': 'bar'})
        
        # Calculate data range
        data_start_row = 4  # Data starts at row 4 (1-based index for headers + 1)
        data_end_row = data_start_row + len(chart_data) - 1
        
        # Add series for valuation range
        bar_chart.add_series({
            'name': 'Valuation Range',
            'categories': f"='Valuation Summary'!$A${data_start_row}:$A${data_end_row}",
            'values': f"='Valuation Summary'!$G${data_start_row}:$G${data_end_row}",  # Implied Share Price column
            'fill': {'color': '#4F81BD'},
            'border': {'color': '#2F5F8F', 'width': 1}
        })
        
        # --- Create a scatter chart for the vertical "Current Price" line ---
        current_price_row = valuation_df[valuation_df['Method'] == 'Current Market Price']
        if not current_price_row.empty:
            current_price = current_price_row.iloc[0]['Implied Share Price']
            if current_price and current_price > 0:
                # To create a vertical line, we need a scatter chart.
                # We'll create dummy data for it and then combine it.
                
                # We need to write the data for the scatter chart to the worksheet first.
                # Let's use some columns far to the right to hide them.
                scatter_col_x = 'Z'
                scatter_col_y = 'AA'
                
                # Write X values (the constant current price)
                val_sheet.write(f'{scatter_col_x}1', 'Current Price Line (X)')
                for i in range(len(chart_data)):
                    val_sheet.write(data_start_row + i -1, 25, current_price) # Col Z is 25

                # Write Y values (a sequence from 0.5 to len-0.5 to center the line on bars)
                val_sheet.write(f'{scatter_col_y}1', 'Current Price Line (Y)')
                for i in range(len(chart_data)):
                     val_sheet.write(data_start_row + i - 1, 26, i + 0.5) # Col AA is 26

                scatter_chart = workbook.add_chart({'type': 'scatter', 'subtype': 'straight'})
                
                scatter_chart.add_series({
                    'name': 'Current Price',
                    'categories': f"='Valuation Summary'!${scatter_col_x}${data_start_row}:${scatter_col_x}${data_end_row}",
                    'values': f"='Valuation Summary'!${scatter_col_y}${data_start_row}:${scatter_col_y}${data_end_row}",
                    'line': {'color': 'red', 'width': 2.5, 'dash_type': 'dash'},
                    'marker': {'type': 'none'},
                })

                # Hide the dummy data columns
                val_sheet.set_column('Z:AA', None, None, {'hidden': True})
                
                # Combine the bar and scatter charts
                bar_chart.combine(scatter_chart)

        # --- Chart Formatting ---
        min_price = chart_data['Implied Share Price'].min()
        max_price = chart_data['Implied Share Price'].max()
        
        # Add padding to chart range
        price_range = max_price - min_price if max_price > min_price else max_price
        chart_min = max(0, min_price - price_range * 0.2)
        chart_max = max_price + price_range * 0.2
        
        bar_chart.set_title({
            'name': f'{target_ticker} Valuation Football Field',
            'name_font': {'size': 14, 'bold': True}
        })
        
        bar_chart.set_x_axis({
            'name': 'Implied Share Price ($)',
            'name_font': {'size': 11, 'bold': True},
            'min': chart_min,
            'max': chart_max,
            'num_format': '$#,##0.00'
        })
        
        bar_chart.set_y_axis({
            'reverse': True,
            'name_font': {'size': 11}
        })
        
        bar_chart.set_legend({'position': 'bottom'})
        bar_chart.set_size({'width': 720, 'height': 450})
        
        # Insert chart into the worksheet
        val_sheet.insert_chart('I2', bar_chart)
        
        logging.info("Football Field chart added successfully")
        
    except Exception as e:
        logging.error(f"Error creating Football Field chart: {str(e)}")