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
from sqlalchemy.orm import relationship, declarative_base
from dotenv import load_dotenv

# --- Database Connection Setup (Unchanged) ---
load_dotenv()
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "sec_data")
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
Base = declarative_base()

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

class FilingMetadata(Base):
    __tablename__ = 'filings_metadata'
    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey('companies.id'), nullable=False, index=True)
    accession_number = Column(String, unique=True, nullable=False)
    form_type = Column(String, nullable=False, index=True)
    filing_date = Column(Date, nullable=False)
    period_of_report = Column(Date, nullable=True)
    url = Column(String, nullable=False)
    company = relationship("Company", back_populates="filings")
    text_sections = relationship("ExtractedTextSection", back_populates="filing", cascade="all, delete-orphan")
    
    # --- NEW RELATIONSHIP FOR 10-K HTML ---
    # This creates a one-to-one relationship to our new table.
    raw_html = relationship("RawFilingHtml", back_populates="filing", uselist=False, cascade="all, delete-orphan")

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