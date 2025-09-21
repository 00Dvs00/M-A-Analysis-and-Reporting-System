import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Database connection
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def check_database_status():
    """
    Comprehensive check of the database to see what data has been downloaded
    and processed according to the ETL scripts.
    """
    try:
        engine = create_engine(DATABASE_URL)
        logging.info("Successfully connected to the database")
        
        print("=" * 80)
        print("DATABASE STATUS REPORT")
        print("=" * 80)
        
        # 1. Check all tables in the database
        print("\n1. AVAILABLE TABLES:")
        tables_query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
        """
        tables_df = pd.read_sql(tables_query, engine)
        for table in tables_df['table_name']:
            print(f"   - {table}")
        
        # 2. Check companies table
        print("\n2. COMPANIES TABLE:")
        try:
            companies_query = """
            SELECT 
                COUNT(*) as total_companies,
                COUNT(DISTINCT sector) as total_sectors,
                COUNT(DISTINCT industry) as total_industries
            FROM companies;
            """
            companies_summary = pd.read_sql(companies_query, engine)
            print(f"   Total companies: {companies_summary['total_companies'].iloc[0]}")
            print(f"   Total sectors: {companies_summary['total_sectors'].iloc[0]}")
            print(f"   Total industries: {companies_summary['total_industries'].iloc[0]}")
            
            # Show sample companies
            sample_companies = pd.read_sql("SELECT ticker, name, sector FROM companies LIMIT 10", engine)
            print("\n   Sample companies:")
            for _, row in sample_companies.iterrows():
                print(f"   - {row['ticker']}: {row['name']} ({row['sector']})")
                
        except Exception as e:
            print(f"   Error checking companies table: {e}")
        
        # 3. Check filings metadata
        print("\n3. FILINGS METADATA:")
        try:
            filings_query = """
            SELECT 
                form_type,
                COUNT(*) as total_filings,
                COUNT(DISTINCT company_id) as companies_with_filings,
                MIN(filing_date) as earliest_filing,
                MAX(filing_date) as latest_filing
            FROM filings_metadata 
            GROUP BY form_type
            ORDER BY form_type;
            """
            filings_summary = pd.read_sql(filings_query, engine)
            for _, row in filings_summary.iterrows():
                print(f"   {row['form_type']}:")
                print(f"     - Total filings: {row['total_filings']}")
                print(f"     - Companies: {row['companies_with_filings']}")
                print(f"     - Date range: {row['earliest_filing']} to {row['latest_filing']}")
        except Exception as e:
            print(f"   Error checking filings metadata: {e}")
        
        # 4. Check extracted text sections (10-Q parsing)
        print("\n4. EXTRACTED TEXT SECTIONS (10-Q Parsing):")
        try:
            sections_query = """
            SELECT 
                section,
                COUNT(*) as total_sections,
                COUNT(DISTINCT filing_id) as filings_with_sections,
                AVG(LENGTH(content)) as avg_content_length
            FROM extracted_text_sections 
            GROUP BY section
            ORDER BY section;
            """
            sections_summary = pd.read_sql(sections_query, engine)
            for _, row in sections_summary.iterrows():
                print(f"   {row['section']}:")
                print(f"     - Total sections: {row['total_sections']}")
                print(f"     - Filings: {row['filings_with_sections']}")
                print(f"     - Avg content length: {row['avg_content_length']:.0f} characters")
        except Exception as e:
            print(f"   Error checking extracted text sections: {e}")
        
        # 5. Check raw filing HTML (10-K archiving)
        print("\n5. RAW FILING HTML (10-K Archiving):")
        try:
            html_query = """
            SELECT 
                COUNT(*) as total_html_archives,
                COUNT(DISTINCT filing_id) as filings_with_html,
                AVG(LENGTH(html_content)) as avg_html_length
            FROM raw_filing_html;
            """
            html_summary = pd.read_sql(html_query, engine)
            print(f"   Total HTML archives: {html_summary['total_html_archives'].iloc[0]}")
            print(f"   Filings with HTML: {html_summary['filings_with_html'].iloc[0]}")
            print(f"   Avg HTML length: {html_summary['avg_html_length'].iloc[0]:.0f} characters")
        except Exception as e:
            print(f"   Error checking raw filing HTML: {e}")
        
        # 6. Check data completeness vs S&P 500
        print("\n6. DATA COMPLETENESS VS S&P 500:")
        try:
            # Load S&P 500 master list
            sp500_df = pd.read_csv("sp500_master_list.csv")
            total_sp500 = len(sp500_df)
            
            # Check how many S&P 500 companies we have
            sp500_in_db_query = """
            SELECT COUNT(DISTINCT c.cik) as sp500_in_db
            FROM companies c
            WHERE c.cik IN ({})
            """.format(','.join(map(str, sp500_df['cik'].tolist())))
            
            sp500_in_db = pd.read_sql(sp500_in_db_query, engine)['sp500_in_db'].iloc[0]
            
            print(f"   Total S&P 500 companies: {total_sp500}")
            print(f"   S&P 500 companies in database: {sp500_in_db}")
            print(f"   Coverage: {(sp500_in_db/total_sp500)*100:.1f}%")
            
            # Find and display missing companies
            if sp500_in_db < total_sp500:
                # Create a temporary table of S&P 500 companies to compare against
                values_list = []
                for _, row in sp500_df.iterrows():
                    ticker = row['ticker'].replace("'", "''")
                    name = row['name'].replace("'", "''")
                    cik = row['cik']
                    sector = row['sector'].replace("'", "''")
                    industry = row['industry'].replace("'", "''")
                    values_list.append(f"('{ticker}', '{name}', {cik}, '{sector}', '{industry}')")
                
                values_string = ",\n".join(values_list)

                missing_companies_query = f"""
                WITH sp500_list (ticker, name, cik, sector, industry) AS (
                    VALUES {values_string}
                )
                SELECT s.ticker, s.name
                FROM sp500_list s
                LEFT JOIN companies c ON s.cik = c.cik
                WHERE c.cik IS NULL;
                """
                
                missing_companies_df = pd.read_sql(missing_companies_query, engine)
                
                print(f"\n   Missing {len(missing_companies_df)} S&P 500 companies:")
                if not missing_companies_df.empty:
                    # Save missing tickers to a file for the batcher
                    missing_tickers_path = "missing_tickers.txt"
                    with open(missing_tickers_path, "w") as f:
                        for ticker in missing_companies_df['ticker']:
                            f.write(f"{ticker}\n")
                    print(f"   Saved list of missing tickers to {missing_tickers_path}")

                    print("   Sample of missing companies:")
                    for _, row in missing_companies_df.head(10).iterrows():
                        print(f"   - {row['ticker']}: {row['name']}")

        except Exception as e:
            print(f"   Error checking S&P 500 completeness: {e}")
        
        # 7. Recent activity
        print("\n7. RECENT ACTIVITY:")
        try:
            recent_filings = pd.read_sql("""
            SELECT c.ticker, fm.form_type, fm.filing_date, fm.accession_number
            FROM filings_metadata fm
            JOIN companies c ON c.id = fm.company_id
            ORDER BY fm.filing_date DESC
            LIMIT 10;
            """, engine)
            
            print("   Most recent filings:")
            for _, row in recent_filings.iterrows():
                print(f"   - {row['ticker']} {row['form_type']} on {row['filing_date']} ({row['accession_number']})")
                
        except Exception as e:
            print(f"   Error checking recent activity: {e}")
        
        print("\n" + "=" * 80)
        print("END OF REPORT")
        print("=" * 80)
        
    except Exception as e:
        logging.error(f"Failed to connect to database or run queries: {e}")

if __name__ == "__main__":
    check_database_status()