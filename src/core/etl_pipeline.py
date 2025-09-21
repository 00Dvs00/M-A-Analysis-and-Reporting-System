import os
import logging
import time
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import re
import numpy as np

# --- Local Imports ---
from src.core.database_schema import Base, Company, FilingMetadata, ExtractedTextSection, RawFilingHtml, DATABASE_URL
from src.data.get_sp500_tickers import get_sp500_info

# --- Third-party Library Imports ---
from sec_edgar_downloader import Downloader
import sec_parser as sp

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()
SEC_EMAIL = os.getenv("SEC_EMAIL")
SEC_COMPANY = os.getenv("SEC_COMPANY_NAME")

# --- Database Setup ---
def get_db_session():
    """Initializes a connection to the database and returns a session object."""
    if not all([SEC_EMAIL, SEC_COMPANY]):
        logging.error("CRITICAL: SEC_EMAIL and SEC_COMPANY_NAME must be set in your .env file.")
        return None
    try:
        engine = create_engine(
            DATABASE_URL,
            connect_args={"options": "-c timezone=Asia/Kolkata"}
        )
        Session = sessionmaker(bind=engine)
        logging.info("Database session created successfully.")
        return Session()
    except Exception as e:
        logging.error(f"Failed to create database session: {e}")
        return None

# --- Core ETL Functions ---

def get_or_create_company(session, ticker, cik, name, sector, industry):
    """
    Checks if a company exists in the DB by its CIK. If not, it creates it.
    """
    company = session.query(Company).filter_by(cik=cik).first()
    if not company:
        logging.info(f"CIK {cik} not found. Creating new company record for ticker {ticker}.")
        company = Company(
            ticker=ticker,
            cik=cik,
            name=name,
            sector=sector,
            industry=industry
        )
        session.add(company)
        try:
            session.commit()
            logging.info(f"Successfully created company {name} with ticker {ticker}.")
        except Exception as e:
            logging.error(f"Error creating company {ticker}: {e}. Rolling back.")
            session.rollback()
            company = session.query(Company).filter_by(cik=cik).first()
    else:
        logging.info(f"Company with CIK {cik} (ticker: {company.ticker}) already exists. Proceeding with existing record.")
    return company

def transform_and_load_filing(session, company_obj, filing_html, metadata_dict):
    """
    Parses or archives a single filing and loads its data into the database.
    """
    accession_number = metadata_dict['accession_number']
    
    existing_filing = session.query(FilingMetadata).filter_by(accession_number=accession_number).first()
    
    if not existing_filing:
        logging.info(f"Processing new filing {accession_number} ({metadata_dict['form_type']}) for {company_obj.ticker}...")
        db_filing = FilingMetadata(
            company_id=company_obj.id,
            accession_number=accession_number,
            form_type=metadata_dict['form_type'],
            filing_date=datetime.strptime(metadata_dict['filing_date'], '%Y-%m-%d').date(),
            period_of_report=datetime.strptime(metadata_dict['period_of_report'], '%Y-%m-%d').date(),
            url=metadata_dict['url']
        )
        session.add(db_filing)
    else:
        if metadata_dict['form_type'] == '10-Q' and not existing_filing.text_sections:
            logging.warning(f"Found orphaned 10-Q {accession_number}. Re-parsing.")
            db_filing = existing_filing
        else:
            logging.info(f"Filing {accession_number} for {company_obj.ticker} already exists and is processed. Skipping.")
            return

    try:
        form_type = metadata_dict['form_type']
        if form_type == '10-Q':
            logging.info(f"Starting robust parsing for 10-Q {accession_number}...")
            elements = sp.Edgar10QParser().parse(filing_html)
            
            section_definitions = {
                'MD&A': { 'start_keywords': ["item 2.", "management's discussion"], 'end_keywords': ["item 3.", "quantitative and qualitative disclosures"] },
                'Risk Factors': { 'start_keywords': ["item 1a.", "risk factors"], 'end_keywords': ["item 1b.", "unresolved staff comments"] }
            }

            for section_name, defs in section_definitions.items():
                section_started = False
                section_elements = []
                
                for element in elements:
                    element_text_lower = str(element.text).lower().strip()
                    if not section_started and any(kw in element_text_lower for kw in defs['start_keywords']) and len(element_text_lower) < 200:
                        section_started = True
                    
                    if section_started:
                        if any(kw in element_text_lower for kw in defs['end_keywords']) and len(element_text_lower) < 200:
                            break
                        section_elements.append(element)

                if section_elements:
                    # --- NEW METHOD: Extract Plain Text ---
                    # Instead of rendering a structured tree, iterate through the parsed elements
                    # and join their raw text content. This gives us clean, plain text.
                    # We use a double newline to act as a paragraph separator.
                    plain_text_content = "\n\n".join(str(e.text) for e in section_elements)
                    # --- END OF NEW METHOD ---

                    new_section = ExtractedTextSection(
                        section=section_name, 
                        content=plain_text_content  # <-- Use the new plain text variable
                    )
                    db_filing.text_sections.append(new_section)
                    
                    logging.info(f"Successfully extracted and staged '{section_name}' for {accession_number}.")

        elif form_type == '10-K':
            if not db_filing.raw_html:
                logging.info(f"Archiving raw HTML for 10-K {accession_number}.")
                RawFilingHtml(html_content=filing_html, filing=db_filing)

        session.commit()
        logging.info(f"Successfully committed data for filing {accession_number} to the database.")

    except Exception as e:
        logging.error(f"Could not process filing {accession_number}. Rolling back. Error: {e}")
        session.rollback()

def process_downloaded_filings(session, company_obj, filings_dir, form_type):
    """
    Process the downloaded SEC filings from the specified directory.
    
    Args:
        session: The database session
        company_obj: The company object
        filings_dir: Directory containing the downloaded filings
        form_type: Type of filings ('10-K' or '10-Q')
    """
    if not os.path.exists(filings_dir):
        logging.error(f"Filings directory {filings_dir} does not exist.")
        return
    
    # List all subdirectories (accession numbers)
    accession_dirs = [d for d in os.listdir(filings_dir) if os.path.isdir(os.path.join(filings_dir, d))]
    
    for accession_dir in accession_dirs:
        try:
            accession_number = accession_dir
            
            # Handle transaction state
            session.rollback()  # Ensure clean transaction state
            
            # Check if filing already exists in the database - use a query that works regardless of schema version
            try:
                # Try with all columns
                query = f"SELECT id FROM filings_metadata WHERE accession_number = '{accession_number}'"
                result = session.execute(query).fetchone()
                existing_filing = result is not None
            except Exception:
                # Fallback to minimal query on error
                existing_filing = session.query(FilingMetadata.id).filter_by(accession_number=accession_number).first() is not None
            
            if existing_filing:
                logging.info(f"Filing {accession_number} already exists. Skipping.")
                continue
            
            # Path to the full submission file
            submission_file = os.path.join(filings_dir, accession_dir, 'full-submission.txt')
            
            if not os.path.exists(submission_file):
                logging.warning(f"Submission file not found for {accession_number}. Skipping.")
                continue
            
            # Read the submission file
            with open(submission_file, 'r', encoding='utf-8', errors='ignore') as f:
                filing_content = f.read()
            
            # Extract metadata from the submission file header
            filing_date, period_of_report = extract_metadata_from_content(filing_content)
            
            # Create a new filing record - only use columns we know exist
            new_filing = FilingMetadata(
                company_id=company_obj.id,
                accession_number=accession_number,
                form_type=form_type,
                filing_date=datetime.strptime(filing_date, '%Y-%m-%d').date(),
                period_of_report=datetime.strptime(period_of_report, '%Y-%m-%d').date(),
                url=f"https://www.sec.gov/Archives/edgar/data/{company_obj.cik}/{accession_number.replace('-', '')}/{accession_number}-index.htm"
            )
            
            session.add(new_filing)
            
            # Process content based on form type
            try:
                if form_type == '10-Q':
                    # Parse 10-Q content
                    elements = sp.Edgar10QParser().parse(filing_content)
                    
                    section_definitions = {
                        'MD&A': { 'start_keywords': ["item 2.", "management's discussion"], 'end_keywords': ["item 3.", "quantitative and qualitative disclosures"] },
                        'Risk Factors': { 'start_keywords': ["item 1a.", "risk factors"], 'end_keywords': ["item 1b.", "unresolved staff comments"] }
                    }
                    
                    for section_name, defs in section_definitions.items():
                        section_started = False
                        section_elements = []
                        
                        for element in elements:
                            element_text_lower = str(element.text).lower().strip()
                            if not section_started and any(kw in element_text_lower for kw in defs['start_keywords']) and len(element_text_lower) < 200:
                                section_started = True
                            
                            if section_started:
                                if any(kw in element_text_lower for kw in defs['end_keywords']) and len(element_text_lower) < 200:
                                    break
                                section_elements.append(element)
                        
                        if section_elements:
                            plain_text_content = "\n\n".join(str(e.text) for e in section_elements)
                            
                            new_section = ExtractedTextSection(
                                section=section_name, 
                                content=plain_text_content
                            )
                            new_filing.text_sections.append(new_section)
                            
                            logging.info(f"Successfully extracted '{section_name}' for {accession_number}.")
                
                elif form_type == '10-K':
                    # Store 10-K content as raw HTML
                    raw_html = RawFilingHtml(html_content=filing_content, filing=new_filing)
                    session.add(raw_html)
                    logging.info(f"Archived raw content for 10-K {accession_number}.")
                
                session.commit()
                logging.info(f"Successfully processed and committed filing {accession_number}.")
                
            except Exception as e:
                logging.error(f"Error processing content for filing {accession_number}: {e}")
                session.rollback()
                
        except Exception as e:
            logging.error(f"Error processing filing {accession_dir}: {e}")
            session.rollback()

def extract_metadata_from_content(content):
    """
    Extract metadata from the filing content.
    
    Args:
        content: The full submission file content
        
    Returns:
        Tuple of (filing_date, period_of_report)
    """
    # Default values
    filing_date = datetime.now().strftime('%Y-%m-%d')
    period_of_report = datetime.now().strftime('%Y-%m-%d')
    
    try:
        # Extract filing date
        filing_date_match = re.search(r'FILED AS OF DATE:\s+(\d{8})', content)
        if filing_date_match:
            date_str = filing_date_match.group(1)
            filing_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        # Extract period of report
        period_match = re.search(r'CONFORMED PERIOD OF REPORT:\s+(\d{8})', content)
        if period_match:
            date_str = period_match.group(1)
            period_of_report = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    except Exception as e:
        logging.error(f"Error extracting metadata: {e}")
    
    return filing_date, period_of_report
        
def run_etl_pipeline(tickers_to_process, num_filings_per_form=4):
    """Main ETL pipeline orchestrator."""
    session = get_db_session()
    if not session:
        return

    # --- Step 2: Initialize the SEC Downloader ---
    # Initialize the downloader with your company name and email.
    # This is a requirement from the SEC to identify who is making the requests.
    dl = Downloader(os.getenv("SEC_COMPANY_NAME"), os.getenv("SEC_EMAIL"))

    # --- Step 3: Process Each Ticker in the Batch ---
    for index, row in tickers_to_process.iterrows():
        ticker = row['ticker']
        cik = row['cik']
        
        # SEC EDGAR requires CIKs to be zero-padded to 10 digits
        padded_cik = str(cik).zfill(10)
        
        logging.info(f"--- Starting Batch Process for {ticker} ({index + 1}/{len(tickers_to_process)}) ---")

        # --- Step 3a: Get or Create the Company Record ---
        # Check if the company already exists in our database.
        company = session.query(Company).filter_by(cik=padded_cik).first()
        if not company:
            # If not, create a new record for it.
            logging.info(f"Creating new database entry for {ticker} (CIK: {padded_cik}).")
            company = Company(
                cik=padded_cik,
                ticker=ticker,
                name=row['name'],
                sector=row['sector'],
                industry=row['industry']
            )
            session.add(company)
            # We commit here so the company record gets an ID for foreign key relationships.
            session.commit()
        else:
            logging.info(f"Company with CIK {padded_cik} (ticker: {ticker}) already exists. Proceeding with existing record.")

        # --- Step 3b: Download Filings for Each Form Type ---
        for form_type in ['10-K', '10-Q']:
            try:
                logging.info(f"Fetching {num_filings_per_form} most recent '{form_type}' filings for {ticker}.")
                
                # Correct ticker format for SEC downloader (e.g., 'BRK.B' -> 'BRK-B')
                downloader_ticker = ticker.replace('.', '-')
                
                # Use the downloader to get the filings. The get method returns the number of filings downloaded.
                # The actual files are saved to disk.
                num_downloaded = dl.get(
                    form=form_type,
                    ticker_or_cik=downloader_ticker,
                    limit=num_filings_per_form
                )
                
                logging.info(f"Downloaded {num_downloaded} '{form_type}' filings for {ticker}.")
                
                # Process the downloaded filings
                if num_downloaded > 0:
                    # Path to the directory containing the downloaded filings
                    filings_dir = os.path.join(dl.download_folder, 'sec-edgar-filings', downloader_ticker, form_type)
                    process_downloaded_filings(session, company, filings_dir, form_type)
                    logging.info(f"Successfully processed '{form_type}' filings for {ticker}.")
                else:
                    logging.warning(f"No '{form_type}' filings found for {ticker}.")

            except Exception as e:
                logging.error(f"Could not process '{form_type}' for {ticker}. Error: {e}")
                continue 

    session.close()
    logging.info("--- ETL Batch Finished ---")

# --- Adjust Prices and Shares for Corporate Actions ---
def adjust_prices_and_shares(prices_df):
    """
    Adjusts prices and shares for corporate actions like splits and dividends.

    Args:
        prices_df (pd.DataFrame): DataFrame containing historical prices and corporate action data.

    Returns:
        pd.DataFrame: Adjusted prices and shares.
    """
    prices_df['adjusted_close'] = prices_df['close'] / prices_df['split_factor']
    prices_df['adjusted_volume'] = prices_df['volume'] * prices_df['split_factor']

    # Adjust for dividends if applicable
    if 'dividend' in prices_df.columns:
        prices_df['adjusted_close'] -= prices_df['dividend']

    return prices_df

# --- Normalize Currency and Units ---
def normalize_currency_and_units(prices_df, fx_rates):
    """
    Normalizes currency and units in the prices DataFrame.

    Args:
        prices_df (pd.DataFrame): DataFrame containing historical prices.
        fx_rates (dict): Dictionary mapping currency codes to exchange rates.

    Returns:
        pd.DataFrame: DataFrame with normalized prices.
    """
    # Assume prices_df has a 'currency' column indicating the currency of each row
    prices_df['normalized_close'] = prices_df.apply(
        lambda row: row['adjusted_close'] * fx_rates.get(row['currency'], 1), axis=1
    )
    prices_df['normalized_volume'] = prices_df['adjusted_volume']  # Assuming volume is already in base units

    return prices_df

# --- Align Fiscal Calendars ---
def align_fiscal_calendars(financials_df, fiscal_year_end):
    """
    Aligns financial data to a consistent fiscal calendar.

    Args:
        financials_df (pd.DataFrame): DataFrame containing financial data with reporting dates.
        fiscal_year_end (str): Month and day representing the fiscal year-end (e.g., '12-31').

    Returns:
        pd.DataFrame: DataFrame with aligned fiscal periods.
    """
    fiscal_year_end_date = pd.to_datetime(fiscal_year_end, format='%m-%d')

    def compute_fiscal_year(row):
        report_date = pd.to_datetime(row['report_date'])
        if report_date.month > fiscal_year_end_date.month or (
            report_date.month == fiscal_year_end_date.month and report_date.day > fiscal_year_end_date.day):
            return report_date.year + 1
        return report_date.year

    financials_df['fiscal_year'] = financials_df.apply(compute_fiscal_year, axis=1)
    return financials_df

# --- Normalize Financial Adjustments ---
def normalize_financial_adjustments(financials_df, adjustments):
    """
    Applies normalization adjustments to financial data.

    Args:
        financials_df (pd.DataFrame): DataFrame containing financial data.
        adjustments (dict): Dictionary specifying adjustments to apply (e.g., remove SBC, leases).

    Returns:
        pd.DataFrame: Adjusted financial data.
    """
    for adjustment, value in adjustments.items():
        if adjustment == 'remove_sbc':
            financials_df['adjusted_ebitda'] = financials_df['ebitda'] - financials_df['sbc']
        elif adjustment == 'treat_leases':
            financials_df['adjusted_ebitda'] = financials_df['adjusted_ebitda'] + financials_df['lease_expense']
        # Add more adjustments as needed

    return financials_df

def compute_enterprise_value(financials_df, config):
    """
    Computes Enterprise Value (EV) based on the given financial data and configuration.

    Args:
        financials_df (pd.DataFrame): DataFrame containing financial data.
        config (dict): Configuration dictionary specifying the components of EV.

    Returns:
        pd.DataFrame: DataFrame with an additional column for Enterprise Value.
    """
    # Extract configuration values
    market_cap_col = config.get('market_cap', 'market_cap')
    total_debt_col = config.get('total_debt', 'total_debt')
    lease_liabilities_col = config.get('lease_liabilities', 'lease_liabilities')
    minority_interest_col = config.get('minority_interest', 'minority_interest')
    cash_col = config.get('cash', 'cash')
    long_term_investments_col = config.get('long_term_investments', 'long_term_investments')

    # Compute EV
    financials_df['enterprise_value'] = (
        financials_df[market_cap_col] +
        financials_df[total_debt_col] +
        financials_df[lease_liabilities_col] +
        financials_df[minority_interest_col] -
        financials_df[cash_col] -
        financials_df[long_term_investments_col]
    )

    return financials_df

# --- Detect and Handle Outliers ---
def detect_and_handle_outliers(data_df, columns, method='IQR', lower_quantile=0.025, upper_quantile=0.975):
    """
    Detects and handles outliers in the specified columns of a DataFrame.

    Args:
        data_df (pd.DataFrame): The input DataFrame.
        columns (list): List of column names to process.
        method (str): Method to handle outliers ('IQR' or 'winsorize').
        lower_quantile (float): Lower quantile for winsorization (default: 2.5%).
        upper_quantile (float): Upper quantile for winsorization (default: 97.5%).

    Returns:
        pd.DataFrame: DataFrame with outliers handled.
    """
    for column in columns:
        if method == 'IQR':
            Q1 = data_df[column].quantile(0.25)
            Q3 = data_df[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            data_df[column] = data_df[column].clip(lower=lower_bound, upper=upper_bound)

        elif method == 'winsorize':
            lower_bound = data_df[column].quantile(lower_quantile)
            upper_bound = data_df[column].quantile(upper_quantile)

            data_df[column] = data_df[column].clip(lower=lower_bound, upper=upper_bound)

    return data_df

# --- Enhance Peer Selection ---
def enhance_peer_selection(companies_df, filters):
    """
    Enhances peer selection based on specified filters.

    Args:
        companies_df (pd.DataFrame): DataFrame containing company data.
        filters (dict): Dictionary specifying filters (e.g., GICS, size, geography).

    Returns:
        pd.DataFrame: Filtered DataFrame of peers.
    """
    filtered_df = companies_df.copy()

    # Apply GICS/SIC/NAICS matching
    if 'industry_code' in filters:
        filtered_df = filtered_df[filtered_df['industry_code'] == filters['industry_code']]

    # Apply size band filters
    if 'size_min' in filters:
        filtered_df = filtered_df[filtered_df['market_cap'] >= filters['size_min']]
    if 'size_max' in filters:
        filtered_df = filtered_df[filtered_df['market_cap'] <= filters['size_max']]

    # Apply geography filters
    if 'geography' in filters:
        filtered_df = filtered_df[filtered_df['region'] == filters['geography']]

    # Apply profitability screens
    if 'profitability_min' in filters:
        filtered_df = filtered_df[filtered_df['profit_margin'] >= filters['profitability_min']]

    return filtered_df

# --- Compute Similarity Scores ---
def compute_similarity_scores(companies_df, target_company, weights):
    """
    Computes similarity scores for a target company against a DataFrame of peers.

    Args:
        companies_df (pd.DataFrame): DataFrame containing peer company data.
        target_company (dict): Dictionary containing the target company's feature vector.
        weights (dict): Dictionary specifying weights for each feature.

    Returns:
        pd.DataFrame: DataFrame with similarity scores.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    # Normalize weights
    weight_vector = np.array([weights[feature] for feature in weights])
    weight_vector = weight_vector / np.linalg.norm(weight_vector)

    # Extract feature matrix and normalize
    feature_columns = list(weights.keys())
    feature_matrix = companies_df[feature_columns].values
    feature_matrix = feature_matrix / np.linalg.norm(feature_matrix, axis=1, keepdims=True)

    # Normalize target company features
    target_vector = np.array([target_company[feature] for feature in feature_columns])
    target_vector = target_vector / np.linalg.norm(target_vector)

    # Compute weighted cosine similarity
    weighted_matrix = feature_matrix * weight_vector
    weighted_target = target_vector * weight_vector
    similarity_scores = cosine_similarity(weighted_matrix, weighted_target.reshape(1, -1)).flatten()

    # Add similarity scores to DataFrame
    companies_df['similarity_score'] = similarity_scores
    return companies_df.sort_values(by='similarity_score', ascending=False)

# --- Metric Library ---
def calculate_ev_to_revenue(financials_df):
    """
    Calculates the EV/Revenue ratio.

    Args:
        financials_df (pd.DataFrame): DataFrame containing financial data with 'enterprise_value' and 'revenue'.

    Returns:
        pd.DataFrame: DataFrame with an additional column for EV/Revenue.
    """
    financials_df['ev_to_revenue'] = financials_df['enterprise_value'] / financials_df['revenue']
    return financials_df

def calculate_ev_to_ebitda(financials_df):
    """
    Calculates the EV/EBITDA ratio.

    Args:
        financials_df (pd.DataFrame): DataFrame containing financial data with 'enterprise_value' and 'ebitda'.

    Returns:
        pd.DataFrame: DataFrame with an additional column for EV/EBITDA.
    """
    financials_df['ev_to_ebitda'] = financials_df['enterprise_value'] / financials_df['ebitda']
    return financials_df

def calculate_pe_ratio(financials_df):
    """
    Calculates the Price-to-Earnings (P/E) ratio.

    Args:
        financials_df (pd.DataFrame): DataFrame containing financial data with 'market_cap' and 'net_income'.

    Returns:
        pd.DataFrame: DataFrame with an additional column for P/E ratio.
    """
    financials_df['pe_ratio'] = financials_df['market_cap'] / financials_df['net_income']
    return financials_df

def calculate_peg_ratio(financials_df):
    """
    Calculates the Price/Earnings-to-Growth (PEG) ratio.

    Args:
        financials_df (pd.DataFrame): DataFrame containing financial data with 'pe_ratio' and 'growth_rate'.

    Returns:
        pd.DataFrame: DataFrame with an additional column for PEG ratio.
    """
    # Ensure P/E ratio is calculated first
    if 'pe_ratio' not in financials_df.columns:
        financials_df = calculate_pe_ratio(financials_df)

    financials_df['peg_ratio'] = financials_df['pe_ratio'] / financials_df['growth_rate']
    return financials_df

def calculate_roic(financials_df):
    """
    Calculates the Return on Invested Capital (ROIC).

    Args:
        financials_df (pd.DataFrame): DataFrame containing financial data with 'net_income', 'total_equity', and 'total_debt'.

    Returns:
        pd.DataFrame: DataFrame with an additional column for ROIC.
    """
    financials_df['roic'] = financials_df['net_income'] / (financials_df['total_equity'] + financials_df['total_debt'])
    return financials_df

def calculate_fcf_yield(financials_df):
    """
    Calculates the Free Cash Flow (FCF) yield.

    Args:
        financials_df (pd.DataFrame): DataFrame containing financial data with 'free_cash_flow' and 'market_cap'.

    Returns:
        pd.DataFrame: DataFrame with an additional column for FCF yield.
    """
    financials_df['fcf_yield'] = financials_df['free_cash_flow'] / financials_df['market_cap']
    return financials_df

# --- Quality Flags and Completeness ---
def add_quality_flags_and_completeness(financials_df):
    """
    Adds quality flags and completeness metrics to the financials DataFrame.

    Args:
        financials_df (pd.DataFrame): DataFrame containing financial data.

    Returns:
        pd.DataFrame: DataFrame with additional columns for quality flags and null rates.
    """
    # Calculate null rates for each column
    for column in financials_df.columns:
        financials_df[f'{column}_null_rate'] = financials_df[column].isnull().mean()

    # Define quality flags based on presence of null values
    financials_df['quality_flag'] = financials_df.apply(
        lambda row: 'low_quality' if row.isnull().any() else 'high_quality',
        axis=1
    )

    return financials_df

def impute_missing_values(financials_df, method='median', peer_group=None):
    """
    Imputes missing values in the financials DataFrame based on the specified method.

    Args:
        financials_df (pd.DataFrame): DataFrame containing financial data.
        method (str): Imputation method ('none', 'median', 'peer_median').
        peer_group (pd.DataFrame, optional): DataFrame containing peer group data for peer median imputation.

    Returns:
        pd.DataFrame: DataFrame with imputed values.
    """
    if method == 'none':
        return financials_df

    for column in financials_df.columns:
        if financials_df[column].isnull().any():
            if method == 'median':
                financials_df[column].fillna(financials_df[column].median(), inplace=True)
            elif method == 'peer_median' and peer_group is not None:
                financials_df[column].fillna(peer_group[column].median(), inplace=True)

    return financials_df

# --- Historical Snapshots and Drift Measurement ---
def create_historical_snapshot(data, snapshot_date):
    """
    Create a historical snapshot of the data as of the given snapshot_date.

    Args:
        data (pd.DataFrame): The financial data to snapshot.
        snapshot_date (str): The date for the snapshot (YYYY-MM-DD).

    Returns:
        pd.DataFrame: A snapshot of the data as of the snapshot_date.
    """
    # Filter data to include only records available up to the snapshot_date
    snapshot = data[data['date'] <= snapshot_date].copy()

    # Ensure the snapshot is sorted by date and retains the latest record per entity
    snapshot = snapshot.sort_values(by=['entity', 'date']).drop_duplicates(subset=['entity'], keep='last')

    return snapshot

def measure_stability_and_drift(current_data, historical_snapshot):
    """
    Measure stability and drift between current data and a historical snapshot.

    Args:
        current_data (pd.DataFrame): The current financial data.
        historical_snapshot (pd.DataFrame): The historical snapshot to compare against.

    Returns:
        dict: Stability and drift metrics.
    """
    metrics = {}

    # Align current data and historical snapshot by entity
    merged = pd.merge(current_data, historical_snapshot, on='entity', suffixes=('_current', '_historical'))

    # Example: Compute drift in EV/EBITDA multiples
    for metric in ['ev_to_ebitda', 'ev_to_revenue']:
        current_col = f'{metric}_current'
        historical_col = f'{metric}_historical'

        if current_col in merged.columns and historical_col in merged.columns:
            # Avoid division by zero or NaN issues
            valid_rows = merged[historical_col] != 0
            drift = ((merged.loc[valid_rows, current_col] - merged.loc[valid_rows, historical_col]).abs() / merged.loc[valid_rows, historical_col]).mean()
            metrics[f'{metric}_drift'] = drift
        else:
            metrics[f'{metric}_drift'] = None  # Indicate missing data for drift calculation

    return metrics

# --- Bootstrap Confidence Intervals ---
def bootstrap_confidence_intervals(data, metric, num_resamples=1000, confidence_level=0.95):
    """
    Compute bootstrap confidence intervals for a given metric.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        metric (str): The column name of the metric to compute confidence intervals for.
        num_resamples (int): Number of bootstrap resamples.
        confidence_level (float): Confidence level for the intervals (e.g., 0.95 for 95%).

    Returns:
        tuple: Lower and upper bounds of the confidence interval.
    """
    if metric not in data.columns:
        raise ValueError(f"Metric '{metric}' not found in the data.")

    resampled_means = []
    for _ in range(num_resamples):
        resample = data[metric].dropna().sample(frac=1, replace=True)
        resampled_means.append(resample.mean())

    lower_bound = np.percentile(resampled_means, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(resampled_means, (1 + confidence_level) / 2 * 100)

    return lower_bound, upper_bound

# --- Reporting Templates and Charts ---
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import os

def generate_distribution_plot(data, column, output_path, title=None, bins=20):
    """
    Generate a distribution plot for a given column and save it as an image.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        column (str): The column to plot.
        output_path (str): Path to save the plot image.
        title (str, optional): Title for the plot. Defaults to None.
        bins (int, optional): Number of bins for the histogram. Defaults to 20.

    Returns:
        str: Path to the saved plot image.
    """
    plt.figure(figsize=(10, 6))
    
    # Use seaborn for better aesthetics
    sns.set_style("whitegrid")
    
    # Create the histogram
    ax = sns.histplot(data[column].dropna(), bins=bins, kde=True)
    
    # Add title and labels
    if title:
        plt.title(title, fontsize=14)
    else:
        plt.title(f'Distribution of {column}', fontsize=14)
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    # Add mean and median lines
    mean_val = data[column].mean()
    median_val = data[column].median()
    plt.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='g', linestyle='--', label=f'Median: {median_val:.2f}')
    
    # Add confidence intervals if available
    try:
        lower, upper = bootstrap_confidence_intervals(data, column)
        plt.axvline(lower, color='b', linestyle=':', label=f'95% CI Lower: {lower:.2f}')
        plt.axvline(upper, color='b', linestyle=':', label=f'95% CI Upper: {upper:.2f}')
        # Shade the confidence interval
        ymin, ymax = ax.get_ylim()
        plt.fill_between([lower, upper], [0, 0], [ymax, ymax], color='blue', alpha=0.1)
    except:
        pass
    
    plt.legend()
    plt.tight_layout()
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the plot
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return output_path

def generate_box_plot(data, column, output_path, title=None, by=None):
    """
    Generate a box plot for a given column and save it as an image.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        column (str): The column to plot.
        output_path (str): Path to save the plot image.
        title (str, optional): Title for the plot. Defaults to None.
        by (str, optional): Column to group by for multiple box plots. Defaults to None.

    Returns:
        str: Path to the saved plot image.
    """
    plt.figure(figsize=(10, 6))
    
    # Use seaborn for better aesthetics
    sns.set_style("whitegrid")
    
    # Create the box plot
    if by is not None and by in data.columns:
        sns.boxplot(x=by, y=column, data=data)
    else:
        sns.boxplot(y=column, data=data)
    
    # Add title and labels
    if title:
        plt.title(title, fontsize=14)
    else:
        plt.title(f'Box Plot of {column}', fontsize=14)
    
    plt.ylabel(column, fontsize=12)
    if by:
        plt.xlabel(by, fontsize=12)
    
    plt.tight_layout()
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the plot
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return output_path

def generate_pdf_report(title, sections, output_path):
    """
    Generate a PDF report with the given title and sections.

    Args:
        title (str): Title of the report.
        sections (list of tuples): Each tuple contains a section title and content (text or image path).
        output_path (str): Path to save the PDF report.

    Returns:
        str: Path to the saved PDF report.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, title, ln=True, align='C')
    pdf.ln(10)

    # Sections
    for section_title, content in sections:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, section_title, ln=True)
        pdf.ln(5)

        if isinstance(content, str) and content.endswith(('.png', '.jpg', '.jpeg')):
            # Adjust image to fit page width with some margin
            pdf.image(content, x=10, w=190)
        elif isinstance(content, pd.DataFrame):
            # Convert DataFrame to table
            pdf.set_font('Arial', '', 8)
            
            # Get the number of columns and column names
            cols = content.columns
            col_widths = [180 / len(cols)] * len(cols)
            
            # Table header
            for i, col in enumerate(cols):
                pdf.cell(col_widths[i], 10, str(col), 1, 0, 'C')
            pdf.ln()
            
            # Table data
            for i, row in content.iterrows():
                for j, col in enumerate(cols):
                    pdf.cell(col_widths[j], 10, str(row[col])[:20], 1, 0, 'L')
                pdf.ln()
        else:
            pdf.set_font('Arial', '', 12)
            pdf.multi_cell(0, 10, str(content))
        pdf.ln(10)

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the PDF
    pdf.output(output_path)
    
    return output_path

def  generate_comps_report(target_ticker, peers_data, metrics, output_dir='reports'):
    """
    Generate a comprehensive comparables analysis report.

    Args:
        target_ticker (str): Ticker symbol of the target company.
        peers_data (pd.DataFrame): DataFrame containing peer company data.
        metrics (list): List of metrics to include in the report.
        output_dir (str, optional): Directory to save the report files. Defaults to 'reports'.

    Returns:
        str: Path to the saved PDF report.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract target company data
    target_data = peers_data[peers_data['ticker'] == target_ticker].copy()
    
    if target_data.empty:
        raise ValueError(f"Target ticker {target_ticker} not found in the data.")
    
    # Format report title
    report_title = f"Comparables Analysis for {target_ticker} ({target_data['name'].iloc[0]})"
    
    # Initialize sections list
    sections = []
    
    # Add company overview section
    sections.append(("Company Overview", f"Target: {target_ticker} - {target_data['name'].iloc[0]}\n"
                    f"Sector: {target_data['sector'].iloc[0]}\n"
                    f"Industry: {target_data['industry'].iloc[0]}\n"
                    f"Market Cap: ${target_data['market_cap'].iloc[0]:,.2f} million\n"))
    
    # Add peer group overview
    top_peers = peers_data.sort_values(by='similarity_score', ascending=False).head(10)
    sections.append(("Peer Group Overview", top_peers[['ticker', 'name', 'similarity_score']]))
    
    # Generate plots for each metric
    for metric in metrics:
        if metric not in peers_data.columns:
            continue
        
        # Distribution plot
        dist_plot_path = os.path.join(output_dir, f"{target_ticker}_{metric}_distribution.png")
        generate_distribution_plot(peers_data, metric, dist_plot_path)
        sections.append((f"{metric.replace('_', ' ').title()} Distribution", dist_plot_path))
        
        # Box plot
        box_plot_path = os.path.join(output_dir, f"{target_ticker}_{metric}_boxplot.png")
        generate_box_plot(peers_data, metric, box_plot_path)
        sections.append((f"{metric.replace('_', ' ').title()} Box Plot", box_plot_path))
        
        # Add summary statistics
        mean_val = peers_data[metric].mean()
        median_val = peers_data[metric].median()
        target_val = target_data[metric].iloc[0] if not target_data[metric].isnull().iloc[0] else "N/A"
        
        # Calculate percentile of target within peer group
        if not target_data[metric].isnull().iloc[0]:
            percentile = (peers_data[metric] < target_val).mean() * 100
            percentile_text = f"Target is at the {percentile:.1f}th percentile of the peer group."
        else:
            percentile_text = "Percentile not available due to missing target value."
        
        # Bootstrap confidence intervals
        try:
            lower, upper = bootstrap_confidence_intervals(peers_data, metric)
            ci_text = f"95% Confidence Interval: [{lower:.2f}, {upper:.2f}]"
        except:
            ci_text = "Confidence interval could not be calculated."
        
        summary = (f"Mean: {mean_val:.2f}\n"
                  f"Median: {median_val:.2f}\n"
                  f"Target: {target_val}\n"
                  f"{percentile_text}\n"
                  f"{ci_text}")
        
        sections.append((f"{metric.replace('_', ' ').title()} Summary", summary))
    
    # Add quality and completeness information
    if 'quality_flag' in peers_data.columns:
        quality_summary = (f"Data Quality:\n"
                         f"High Quality Records: {(peers_data['quality_flag'] == 'high_quality').sum()}\n"
                         f"Low Quality Records: {(peers_data['quality_flag'] == 'low_quality').sum()}\n"
                         f"Target Quality: {target_data['quality_flag'].iloc[0]}")
        
        sections.append(("Data Quality Summary", quality_summary))
    
    # Generate PDF report
    report_path = os.path.join(output_dir, f"{target_ticker}_comps_analysis.pdf")
    generate_pdf_report(report_title, sections, report_path)
    
    return report_path

# Example usage:
# fx_rates = {'USD': 1, 'EUR': 1.1, 'JPY': 0.009}
# prices_df = normalize_currency_and_units(prices_df, fx_rates)
# financials_df = align_fiscal_calendars(financials_df, '12-31')
# adjustments = {'remove_sbc': True, 'treat_leases': True}
# financials_df = normalize_financial_adjustments(financials_df, adjustments)
# config = {'market_cap': 'market_cap', 'total_debt': 'total_debt', 'lease_liabilities': 'lease_liabilities', 'minority_interest': 'minority_interest', 'cash': 'cash', 'long_term_investments': 'long_term_investments'}
# financials_df = compute_enterprise_value(financials_df, config)

if __name__ == "__main__":
    sp500_df = get_sp500_info()
    
    if sp500_df is not None:
        BATCH_SIZE = 20
        START_INDEX = 0
        
        end_index = START_INDEX + BATCH_SIZE
        tickers_batch = sp500_df.iloc[START_INDEX:end_index]
        
        if not tickers_batch.empty:
            run_etl_pipeline(tickers_to_process=tickers_batch, num_filings_per_form=4)
        else:
            print("No more tickers to process.")