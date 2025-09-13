import os
import logging
import time
from sqlalchemy import create_engine, and_
from sqlalchemy.orm import sessionmaker, subqueryload
from dotenv import load_dotenv

# --- Third-party Library Imports ---
from sec_downloader import Downloader

# --- Local Imports ---
# Use our corrected and verified database schema
from database_schema import Base, FilingMetadata, RawFilingHtml, DATABASE_URL

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

def backfill_missing_html():
    """
    Finds 10-K filings in the database that are missing their raw HTML archive
    and downloads the content to complete the data set.
    """
    session = get_db_session()
    if not session or not SEC_EMAIL or not SEC_COMPANY:
        logging.error("Prerequisites not met. Aborting backfill job.")
        return

    logging.info("Starting the backfill job for missing 10-K HTML archives...")
    
    try:
        # 1. Efficiently query for 10-K filings that do NOT have a related RawFilingHtml record.
        #    The `outerjoin` and `filter` combination is the key to finding the missing ones.
        filings_to_backfill = session.query(FilingMetadata).outerjoin(RawFilingHtml).filter(
            and_(
                FilingMetadata.form_type == '10-K',
                RawFilingHtml.id == None
            )
        ).all()
        
        total_to_fix = len(filings_to_backfill)
        if total_to_fix == 0:
            logging.info("No missing 10-K archives found. Your data is already complete!")
            return

        logging.info(f"Found {total_to_fix} 10-K filings that need their HTML archived.")

        dl = Downloader(SEC_COMPANY, SEC_EMAIL)
        
        for i, filing in enumerate(filings_to_backfill):
            logging.info(f"Backfilling {i+1}/{total_to_fix}: {filing.accession_number}")
            time.sleep(0.5) # Be respectful to SEC servers

            try:
                # 2. Download the filing's HTML
                filing_html = dl.download_filing(url=filing.url).decode('utf-8', errors='ignore')
                
                # 3. Create the new RawFilingHtml object and link it to the filing
                new_html_archive = RawFilingHtml(
                    filing_id=filing.id,
                    html_content=filing_html
                )
                session.add(new_html_archive)
                
                # 4. Commit changes for this one filing to save progress incrementally
                session.commit()

            except Exception as e:
                logging.error(f"Could not backfill filing {filing.accession_number}. Rolling back. Error: {e}")
                session.rollback()
                continue # Move to the next filing

    except Exception as e:
        logging.error(f"A major error occurred during the backfill job: {e}")
        session.rollback()
    finally:
        session.close()
        logging.info("--- Backfill Job Finished ---")

if __name__ == "__main__":
    backfill_missing_html()