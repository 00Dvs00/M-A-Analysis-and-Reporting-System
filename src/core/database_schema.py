import os
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    Date,
    Numeric,
    BigInteger,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import relationship, declarative_base, sessionmaker
from dotenv import load_dotenv

# --- Database Connection Setup ---
load_dotenv()

# Check if we're using Docker database (port 5433) or local database (port 5432)
# Docker database credentials
DOCKER_DB_USER = "sec_user"
DOCKER_DB_PASSWORD = "T1538"
DOCKER_DB_HOST = "localhost"
DOCKER_DB_PORT = "5433"
DOCKER_DB_NAME = "sec_data"

# Environment variables (for local database)
DB_USER = os.getenv("DB_USER", DOCKER_DB_USER)
DB_PASSWORD = os.getenv("DB_PASSWORD", DOCKER_DB_PASSWORD)
DB_HOST = os.getenv("DB_HOST", DOCKER_DB_HOST)
DB_PORT = os.getenv("DB_PORT", DOCKER_DB_PORT)
DB_NAME = os.getenv("DB_NAME", DOCKER_DB_NAME)

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
print(f"Using database connection: {DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db_session():
    """Provides a database session."""
    return SessionLocal()

# --- Model Definitions ---

class Company(Base):
    __tablename__ = 'companies'
    id = Column(Integer, primary_key=True)
    ticker = Column(String, unique=True, nullable=False, index=True)
    cik = Column(Integer, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    sector = Column(String, nullable=True)
    industry = Column(String, nullable=True)
    filings = relationship("FilingMetadata", back_populates="company", cascade="all, delete-orphan")
    active = Column(Integer, nullable=False, default=1)  # 1 for active, 0 for inactive
    inactive_date = Column(Date, nullable=True)
    delisted_date = Column(Date, nullable=True)
    membership_date = Column(Date, nullable=True)

class FilingMetadata(Base):
    __tablename__ = 'filings_metadata'
    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey('companies.id'), nullable=False, index=True)
    accession_number = Column(String, unique=True, nullable=False)
    form_type = Column(String, nullable=False, index=True)
    filing_date = Column(Date, nullable=False)
    period_of_report = Column(Date, nullable=True)
    url = Column(String, nullable=False)
    schema_version = Column(String, nullable=True)  # --- SCHEMA VERSION COLUMN ---
    run_id = Column(Integer, ForeignKey('etl_runs.id'), nullable=True, index=True)  # --- RUN ID FK ---
    company = relationship("Company", back_populates="filings")
    text_sections = relationship("ExtractedTextSection", back_populates="filing", cascade="all, delete-orphan")
    
    # --- NEW RELATIONSHIP FOR 10-K HTML ---
    # This creates a one-to-one relationship to our new table.
    raw_html = relationship("RawFilingHtml", back_populates="filing", uselist=False, cascade="all, delete-orphan")
    etl_run = relationship("ETLRun", back_populates="filings")  # --- RELATIONSHIP TO ETL RUN ---

class FinancialFact(Base):
    __tablename__ = 'financial_facts'
    id = Column(BigInteger, primary_key=True)
    filing_id = Column(Integer, ForeignKey('filings_metadata.id'), nullable=False, index=True)
    us_gaap_tag = Column(String, nullable=False, index=True)
    value = Column(Numeric, nullable=False)
    unit = Column(String, nullable=False)
    period_end_date = Column(Date, nullable=False)
    filing = relationship("FilingMetadata")

class ExtractedTextSection(Base):
    __tablename__ = 'extracted_text_sections'
    id = Column(Integer, primary_key=True)
    filing_id = Column(Integer, ForeignKey('filings_metadata.id'), nullable=False, index=True)
    section = Column(String, nullable=False, index=True)
    content = Column(Text, nullable=False)
    filing = relationship("FilingMetadata", back_populates="text_sections")

# --- NEW TABLE FOR RAW 10-K HTML ---
class RawFilingHtml(Base):
    """
    Represents the 'raw_filing_html' table, designed to store the complete
    HTML content of filings that we don't parse (like 10-Ks).
    """
    __tablename__ = 'raw_filing_html'
    id = Column(Integer, primary_key=True)
    filing_id = Column(Integer, ForeignKey('filings_metadata.id'), nullable=False, index=True)
    html_content = Column(Text, nullable=False)
    filing = relationship("FilingMetadata", back_populates="raw_html")

# --- NEW TABLE FOR ETL RUNS ---
class ETLRun(Base):
    __tablename__ = 'etl_runs'
    id = Column(Integer, primary_key=True)
    run_id = Column(String, unique=True, nullable=False, index=True)
    start_time = Column(Date, nullable=False)
    end_time = Column(Date, nullable=True)
    status = Column(String, nullable=False)
    source = Column(String, nullable=False)
    filings = relationship("FilingMetadata", back_populates="etl_run")  # --- RELATIONSHIP TO FILINGS ---


# --- Engine and Table Creation Function ---
def create_database_tables():
    """Connects to the database and creates all tables defined in the models."""
    try:
        # Explicitly set the timezone in the connection options
        engine = create_engine(
            DATABASE_URL,
            connect_args={"options": "-c timezone=Asia/Kolkata"}
        )
        print("--- Connecting to PostgreSQL database and creating tables... ---")
        Base.metadata.create_all(engine)
        print("--- Database tables created/verified successfully. ---")
        return engine
    except Exception as e:
        print(f"An error occurred while connecting or creating tables: {e}")
        return None

if __name__ == '__main__':
    create_database_tables()