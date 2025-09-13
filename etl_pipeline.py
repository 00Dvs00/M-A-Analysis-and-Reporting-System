import os
import logging
import time
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import re

# --- Local Imports ---
from database_schema import Base, Company, FilingMetadata, ExtractedTextSection, RawFilingHtml, DATABASE_URL
from get_sp500_tickers import get_sp500_info

# --- Third-party Library Imports ---
from sec_downloader import Downloader
from sec_downloader.types import RequestedFilings
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
        
def run_etl_pipeline(tickers_to_process, num_filings_per_form=4):
    """Main ETL pipeline orchestrator."""
    session = get_db_session()
    if not session:
        return

    dl = Downloader(SEC_COMPANY, SEC_EMAIL)
    total_tickers = len(tickers_to_process)

    for index, row in tickers_to_process.iterrows():
        ticker = row['ticker']
        current_pos = tickers_to_process.index.get_loc(index)
        logging.info(f"--- Starting Batch Process for {ticker} ({current_pos + 1}/{total_tickers}) ---")
        
        company_obj = get_or_create_company(session, ticker, row['cik'], row['name'], row['sector'], row['industry'])
        
        forms_to_fetch = ["10-K", "10-Q"]
        for form_type in forms_to_fetch:
            try:
                logging.info(f"Fetching {num_filings_per_form} most recent '{form_type}' filings for {ticker}.")
                filings_metadata = dl.get_filing_metadatas(RequestedFilings(ticker_or_cik=ticker, form_type=form_type, limit=num_filings_per_form))
                
                for metadata in filings_metadata:
                    time.sleep(0.5)
                    filing_html = dl.download_filing(url=metadata.primary_doc_url).decode('utf-8', errors='ignore')
                    
                    metadata_dict = {
                        'accession_number': metadata.accession_number, 'form_type': metadata.form_type,
                        'filing_date': metadata.filing_date, 'period_of_report': metadata.report_date, 'url': metadata.primary_doc_url
                    }
                    transform_and_load_filing(session, company_obj, filing_html, metadata_dict)

            except Exception as e:
                logging.error(f"Could not process '{form_type}' for {ticker}. Error: {e}")
                continue 

    session.close()
    logging.info("--- ETL Batch Finished ---")


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