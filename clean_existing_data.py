import os
import re
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# --- Local Imports ---
# This script now imports the corrected schema definitions that perfectly match your database
from database_schema import Base, ExtractedTextSection, DATABASE_URL

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_db_session():
    """Initializes a connection to the database and returns a session object."""
    if not DATABASE_URL:
        logging.error("FATAL: DATABASE_URL environment variable not found.")
        logging.error("Please ensure your .env file is correct and in the project root.")
        return None
    try:
        engine = create_engine(DATABASE_URL)
        Session = sessionmaker(bind=engine)
        logging.info("Database session created successfully.")
        return Session()
    except Exception as e:
        logging.error(f"Failed to create database session: {e}")
        return None

def clean_all_extracted_sections():
    """
    Finds all records in the extracted_text_sections table, cleans the
    'content' field by removing ANSI escape codes, and updates the record.
    """
    session = get_db_session()
    if not session:
        return

    logging.info("Starting the data cleaning process for existing 10-Q sections...")
    
    # Define the regex pattern to find ANSI escape codes for terminal colors
    ansi_escape_pattern = re.compile(r'\x1b\[[0-9;]*m')
    
    try:
        # Fetch all records from the table. Note the use of the correct class name.
        all_sections = session.query(ExtractedTextSection).all()
        total_records = len(all_sections)
        
        if total_records == 0:
            logging.warning("No text sections found in the database to clean.")
            return

        logging.info(f"Found {total_records} records to check and clean.")
        
        cleaned_count = 0
        for record in all_sections:
            # Check if the content has ANSI codes to avoid unnecessary database writes
            if record.content and '\x1b[' in record.content:
                # Clean the content field
                cleaned_content = ansi_escape_pattern.sub('', record.content)
                record.content = cleaned_content
                cleaned_count += 1

        if cleaned_count > 0:
            logging.info(f"Cleaned {cleaned_count} records. Committing changes to the database...")
            # Commit the transaction to save all changes at once
            session.commit()
            logging.info("Successfully saved all cleaned data.")
        else:
            logging.info("No records contained ANSI codes. Data is already clean.")

    except Exception as e:
        logging.error(f"An error occurred during the cleaning process: {e}")
        session.rollback() # Roll back in case of an error
    finally:
        session.close() # Always close the session
        logging.info("--- Cleaning Process Finished ---")

# --- Execution Guard ---
if __name__ == "__main__":
    # Ensure the .env file is loaded before we do anything
    load_dotenv()
    clean_all_extracted_sections()