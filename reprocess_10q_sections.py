import os
import re
import logging
import time
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# --- Third-party Library Imports ---
from sec_downloader import Downloader
import sec_parser as sp

# --- Local Imports ---
from database_schema import Base, FilingMetadata, ExtractedTextSection, DATABASE_URL

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()
SEC_EMAIL = os.getenv("SEC_EMAIL")
SEC_COMPANY = os.getenv("SEC_COMPANY_NAME")

def get_db_session():
    """Initializes a connection to the database and returns a session object."""
    if not DATABASE_URL:
        logging.error("FATAL: DATABASE_URL not set. Check your .env file.")
        return None
    try:
        engine = create_engine(DATABASE_URL)
        Session = sessionmaker(bind=engine)
        logging.info("Database session created successfully.")
        return Session()
    except Exception as e:
        logging.error(f"Failed to create database session: {e}")
        return None

def reprocess_all_10q_filings():
    """
    Iterates through all 10-Q filings in the database, re-downloads them,
    re-parses them to extract clean plain text, and replaces the old records.
    """
    session = get_db_session()
    if not session or not SEC_EMAIL or not SEC_COMPANY:
        logging.error("Prerequisites not met. Aborting re-processing.")
        return

    logging.info("Starting the re-processing job for all existing 10-Q filings...")
    
    try:
        # 1. Fetch all 10-Q filings that need reprocessing.
        filings_to_process = session.query(FilingMetadata).filter_by(form_type='10-Q').all()
        total_filings = len(filings_to_process)
        logging.info(f"Found {total_filings} 10-Q filings to re-process.")

        dl = Downloader(SEC_COMPANY, SEC_EMAIL)
        
        for i, filing in enumerate(filings_to_process):
            logging.info(f"Processing {i+1}/{total_filings}: {filing.accession_number} for ticker {filing.company.ticker}")
            time.sleep(0.5) # Be respectful to SEC servers

            try:
                # 2. Re-download the filing's HTML
                filing_html = dl.download_filing(url=filing.url).decode('utf-8', errors='ignore')
                
                # 3. Delete the old, badly formatted sections for this filing
                session.query(ExtractedTextSection).filter_by(filing_id=filing.id).delete(synchronize_session=False)

                # 4. Re-run the parsing logic (same as the fixed ETL pipeline)
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
                        # 5. Extract plain text and create the new section object
                        plain_text_content = "\n\n".join(str(e.text) for e in section_elements)
                        new_section = ExtractedTextSection(
                            filing_id=filing.id,
                            section=section_name,
                            content=plain_text_content
                        )
                        session.add(new_section)
                
                # 6. Commit changes for this one filing to save progress
                session.commit()

            except Exception as e:
                logging.error(f"Could not process filing {filing.accession_number}. Rolling back. Error: {e}")
                session.rollback()
                continue # Move to the next filing

    except Exception as e:
        logging.error(f"A major error occurred during the job: {e}")
        session.rollback()
    finally:
        session.close()
        logging.info("--- Re-processing Job Finished ---")

if __name__ == "__main__":
    reprocess_all_10q_filings()