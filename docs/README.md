# M&A Analysis and Reporting System

A comprehensive, automated system for generating professional-grade comparable company analysis reports for investment banking and M&A purposes. This system downloads SEC filing data, processes financial information, and generates detailed Excel reports with valuation metrics, peer comparisons, and visualization charts.

## 🎯 Features

- **Automated Peer Selection**: Multi-tier peer identification using Finnhub API, database lookups, and intelligent fallback mechanisms
- **Advanced Financial Analysis**: Comprehensive ratio analysis, valuation multiples, and statistical benchmarking
- **Professional Excel Reports**: Formatted reports with multiple sheets, charts, and "Football Field" valuation visualization
- **Robust Data Pipeline**: SEC filing processing and financial data storage in PostgreSQL
- **End-to-End Data Validation**: Recalculates key figures and flags discrepancies over tight tolerances (0.1% for EV, 0.5% for EPS)
- **User-Friendly Interface**: Simple command-line interface for generating reports

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.9+** - [Download here](https://www.python.org/downloads/)
- **PostgreSQL 12+** - [Download here](https://www.postgresql.org/download/)
- **Docker Desktop** (Optional) - [Download here](https://www.docker.com/products/docker-desktop/)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd M-A-Analysis-and-Reporting-System
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory with the following content:

```env
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/ma_analysis

# SEC EDGAR Contact Information (Required by SEC)
SEC_EMAIL=your.email@domain.com
SEC_COMPANY_NAME=Your Company Name

# Finnhub API Configuration
FINNHUB_API_KEY=your_finnhub_api_key_here
```

**Important**: 
- Replace `username`, `password`, and database details with your PostgreSQL credentials
- Get your free Finnhub API key from [https://finnhub.io/](https://finnhub.io/)
- Use your real email for SEC compliance

### 4. Set Up Database

#### Option A: Using Docker (Recommended)

```bash
docker-compose up -d
```

This will create a PostgreSQL container with the database `ma_analysis`.

#### Option B: Manual PostgreSQL Setup

1. Create a new database called `ma_analysis`
2. Update the `DATABASE_URL` in your `.env` file with your database credentials

### 5. Initialize Data Pipeline

Run the ETL pipeline to populate your database with SEC filing data:

```bash
python etl_pipeline.py
```

**Note**: This process may take 30-60 minutes depending on your internet connection and the number of companies processed.

### 6. Generate Your First Report

```bash
python main.py
```

Enter a ticker symbol (e.g., `AAPL`, `MSFT`, `GOOGL`) and wait for your comprehensive analysis report!

## 📖 Detailed Usage

### Running the Analysis

The main entry point is `main.py`, which provides an interactive interface:

```bash
python main.py
```

**Example workflow:**
1. Script prompts: `Enter ticker symbol to analyze:`
2. You type: `AAPL`
3. System generates: `AAPL_Advanced_Comps.xlsx`

### Understanding the Output

Each analysis generates an Excel file with multiple sheets:

1. **Executive Summary** - Key metrics and valuation ranges
2. **Target Company** - Detailed financial profile of analyzed company
3. **Peer Analysis** - Comprehensive peer company data and metrics
4. **Summary Statistics** - Statistical analysis and benchmarking
5. **Valuation Summary** - Multiple valuation methodologies and ranges
6. **Football Field Chart** - Visual valuation range analysis

### Updating Data

To refresh your database with the latest SEC filings:

```bash
python etl_pipeline.py
```

Run this periodically (weekly/monthly) to keep your analysis current.

## 🔧 Advanced Configuration

### Custom Analysis Parameters

You can customize the analysis by modifying the `CompsConfig` class in `comps_analysis.py`:

```python
custom_config = CompsConfig(
    max_initial_peers=30,           # Reduce peer count
    min_peers_required=5,           # Require more peers
    enable_size_filtering=False,    # Disable size filtering
    # ... other parameters
)

generate_comps_report("AAPL", custom_config)
```

### Database Schema

The system uses the following main tables:
- `companies` - Company master data
- `filing_metadata` - SEC filing information
- `extracted_text_sections` - Processed filing content
- `raw_filing_html` - Raw HTML filing data

### API Rate Limits

- **Finnhub**: 60 calls/minute (free tier)
- **SEC EDGAR**: 10 requests/second maximum
- **Yahoo Finance**: No official limits, but be respectful

## 🛠 Troubleshooting

### Common Issues

#### "Database connection failed"
- Verify PostgreSQL is running
- Check `DATABASE_URL` in `.env` file
- Ensure database `ma_analysis` exists

#### "Finnhub API key invalid"
- Verify your API key in `.env` file
- Check if you've exceeded rate limits
- Ensure key is from [finnhub.io](https://finnhub.io/)

#### "No peers found for ticker"
- Ticker might be delisted or invalid
- Try with well-known tickers (AAPL, MSFT, GOOGL)
- Check if database contains sufficient data

#### "Excel file generation failed"
- Ensure you have write permissions in project directory
- Check if Excel file is open in another application
- Verify xlsxwriter installation: `pip install xlsxwriter`

### Debug Mode

Enable detailed logging by modifying `comps_analysis.py`:

```python
logging.basicConfig(level=logging.DEBUG)
```

### Getting Help

1. Check the log output for detailed error messages
2. Verify all environment variables are set correctly
3. Ensure all dependencies are installed: `pip install -r requirements.txt`
4. Test database connection independently

## 📁 Project Structure

```
M&A Analysis System/
├── main.py                 # Main entry point - start here!
├── comps_analysis.py       # Core analysis engine
├── etl_pipeline.py         # Data extraction and loading
├── database_schema.py      # Database structure definition
├── validation_engine.py    # Data validation system
├── validation_reporting.py # Validation results reporting
├── get_sp500_tickers.py    # S&P 500 ticker utilities
├── config/                 # Configuration files
│   └── default_config.yaml # System configuration
├── tests/                  # Test cases
│   └── test_validation_engine.py # Validation tests
├── requirements.txt        # Python dependencies
├── docker-compose.yml      # Docker database setup
├── .env                    # Environment configuration
├── validation_documentation.md # Validation system docs
└── README.md              # This file
```

## 🔄 Typical Workflow

### First-Time Setup
1. Install Python and PostgreSQL
2. Clone repository and install dependencies
3. Configure `.env` file
4. Set up database (Docker or manual)
5. Run ETL pipeline to populate data
6. Test with sample analysis

### Daily Usage
1. Run `python main.py`
2. Enter ticker symbol
3. Wait for analysis completion
4. Review generated Excel report
5. Repeat for additional companies

### Maintenance
- Update data weekly: `python etl_pipeline.py`
- Monitor API usage and rate limits
- Backup database periodically

## 🔎 Validation System

The system includes a robust validation engine to ensure data integrity:

### Key Features
- **Independent Recalculation**: Recalculates Market Cap, EV, EPS, and multiples from raw components
- **Configurable Tolerances**: Tight tolerance thresholds (0.1% for EV, 0.5% for EPS)
- **Multi-level Severity**: INFO, WARNING, ERROR, and CRITICAL severity levels
- **Processing Control**: Halts processing when critical thresholds are exceeded
- **Validation Reporting**: Detailed validation results in Excel reports with color-coding
- **Comprehensive Logging**: Audit trail of all validation checks and issues

### Configuration
Validation thresholds can be customized in `config/default_config.yaml`. See `validation_documentation.md` for detailed information on the validation system.

### Validation Process
1. Data is fetched from external sources
2. Key metrics are independently recalculated
3. Recalculated values are compared against source values
4. Discrepancies are flagged based on configured thresholds
5. Processing continues or halts based on severity
6. Results are included in Excel reports

## 🤝 Contributing

This system is designed for educational and research purposes. When using for commercial applications:

1. Ensure compliance with SEC data usage policies
2. Respect API rate limits and terms of service
3. Verify all financial data independently
4. Consider professional data providers for production use

## 📄 License

This project is for educational and research purposes. Please ensure compliance with all applicable financial data regulations and API terms of service.

## 🔗 Resources

- **SEC EDGAR Database**: [https://www.sec.gov/edgar](https://www.sec.gov/edgar)
- **Finnhub Financial Data**: [https://finnhub.io/](https://finnhub.io/)
- **PostgreSQL Documentation**: [https://www.postgresql.org/docs/](https://www.postgresql.org/docs/)
- **Pandas Documentation**: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)

---

**Ready to generate your first professional M&A analysis report? Run `python main.py` and get started! 🚀**
SEC_COMPANY_NAME="Your Company or Project Name"
SEC_EMAIL="your.email@example.com"

# Database Connection Details
DB_USER=sec_user
DB_PASSWORD=your_secure_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=sec_data

# API Keys (Optional, for other data sources)
FMP_API_KEY=
FINNHUB_API_KEY=

# Timezone
TIMEZONE=Asia/Kolkata
```
**Note:** Replace `"your.email@example.com"` and `"Your Company or Project Name"` with your actual information. Choose a secure password for `DB_PASSWORD`.

### 2. Set Up the Python Environment

It is highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On macOS/Linux:
# source .venv/bin/activate
```

### 3. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Start the Database

This project uses Docker to run a PostgreSQL database. The `docker-compose.yml` file is pre-configured to work with an existing data volume named `pg_sec_data`.

**Important:** If you are running this for the first time and do not have this volume, Docker will show an error. To create it, run `docker volume create pg_sec_data` before proceeding.

Start the database container in the background:
```bash
docker-compose up -d
```
The database will now be running on `localhost:5432`.

## How to Run the Project

Once the setup is complete, you can use the following scripts to manage and populate your database.

### Initial Data Load

To perform the initial download of all S&P 500 company data, run the `etl_pipeline.py` script directly. This will fetch all companies and their recent filings.

```bash
python etl_pipeline.py
```
**Note:** This process can take a significant amount of time as it downloads a large volume of data.

### Check Data Status

To get a report on the current state of your database, including data coverage and recent activity, run the `check_data_status.py` script.

```bash
python check_data_status.py
```

### Backfill Missing Data

If the data status check reveals that some companies are missing, you can run the `backfill.py` script. First, create a file named `missing_tickers.txt` and add the ticker symbols of the missing companies, one per line. Then, run the script:

```bash
python backfill.py
```
This will download and process data only for the companies listed in the file.

## Project Structure

Here is a brief overview of the key files in this project:

| File | Description |
| --- | --- |
| `etl_pipeline.py` | The core script for the ETL (Extract, Transform, Load) process. |
| `database_schema.py`| Defines the database tables using SQLAlchemy ORM. |
| `get_sp500_tickers.py`| Utility to fetch the current list of S&P 500 companies. |
| `check_data_status.py`| Generates a health and completeness report of the database. |
| `backfill.py` | A targeted script to download data for specific missing companies. |
| `docker-compose.yml`| Configuration file for running the PostgreSQL database in Docker. |
| `.env` | Stores environment variables and secrets. **Do not commit this file.** |
| `requirements.txt` | A list of all Python packages required for the project. |
| `sp500_master_list.csv`| A cached master list of S&P 500 companies. |

---

# --- Data Source SLAs and Retry Strategies ---
# 1. S&P 500 Tickers:
#    - Source: Wikipedia
#    - SLA: Ensure data freshness by scraping daily.
#    - Retry: 3 attempts with exponential backoff (2s, 4s, 8s).
#    - Fallback: Use a cached version of the last successful scrape.
#
# 2. SEC Filings:
#    - Source: SEC EDGAR database
#    - SLA: Ensure filings are fetched within 24 hours of availability.
#    - Retry: 5 attempts with linear backoff (5s each).
#    - Fallback: Notify admin if retries fail; consider manual intervention.
#
# 3. Peer Data:
#    - Source: Finnhub API
#    - SLA: Ensure API key is valid and rate limits are respected.
#    - Retry: 3 attempts with exponential backoff (1s, 2s, 4s).
#    - Fallback: Use a static peer list or notify admin for manual update.
