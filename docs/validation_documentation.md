# Validation Engine Documentation

## Overview

The Validation Engine is a key component of the M&A Analysis and Reporting System that ensures data integrity by performing end-to-end validation of critical financial figures. It recalculates key metrics independently and compares them against ingested data, flagging any discrepancies that exceed configurable tolerance thresholds.

## Key Features

- **Independent Recalculation**: Recalculates Market Cap, Enterprise Value (EV), EPS, and key multiples from raw components
- **Configurable Tolerances**: Tight tolerance thresholds (0.1% for EV, 0.5% for EPS) with configurable values
- **Blocking Thresholds**: Configurable blocking thresholds that halt further processing when exceeded
- **Validation Reporting**: Detailed validation reports in Excel with visual indicators
- **Audit Logging**: Comprehensive validation logs for audit and troubleshooting

## Configuration

The validation engine is configured in `config/default_config.yaml`:

```yaml
# Data validation configuration
validation:
  enabled: true  # Master switch to enable/disable validation
  
  # Warning thresholds - Issues that exceed these thresholds will be logged as warnings
  thresholds:
    market_cap: 0.005   # 0.5% tolerance for Market Cap
    enterprise_value: 0.001  # 0.1% tolerance for Enterprise Value
    eps: 0.005  # 0.5% tolerance for EPS
    multiples: 0.01  # 1% tolerance for multiples (EV/Sales, EV/EBITDA, P/E)
  
  # Blocking thresholds - Issues that exceed these thresholds will halt processing
  blocking_thresholds:
    market_cap: 0.01   # 1% blocking threshold for Market Cap
    enterprise_value: 0.002  # 0.2% blocking threshold for Enterprise Value
    eps: 0.01  # 1% blocking threshold for EPS
    multiples: 0.02  # 2% blocking threshold for multiples
  
  # Metrics to validate
  metrics_to_validate:
    - Market Cap
    - EV
    - EPS Forward
    - P/E Forward
    - EV/Sales LTM
    - EV/EBITDA LTM
  
  # Validation reporting
  reporting:
    log_to_file: true
    log_file: validation.log
    save_results: true
    results_dir: validation_results
```

## How It Works

### 1. Data Ingestion and Initial Validation

When financial data is fetched from external sources in `get_comps_data()`, the validation engine:
- Performs immediate validation after data retrieval
- Calculates expected values independently
- Compares expected values against retrieved values
- Flags discrepancies that exceed configured thresholds

### 2. Validation Results Tracking

The validation results are:
- Stored with each company's data
- Tracked as the data flows through the pipeline
- Used to make process control decisions

### 3. Process Control

Based on validation results:
- Minor discrepancies (below warning thresholds): Processing continues normally
- Medium discrepancies (above warning but below blocking thresholds): Processing continues with warnings
- Severe discrepancies (above blocking thresholds): Processing may be halted

### 4. Reporting and Auditing

The validation engine generates:
- A dedicated "Validation Results" tab in Excel reports
- Color-coded status indicators (green for passed, yellow for warnings, red for errors)
- Detailed validation logs for audit purposes

## Validation Formulas

The validation engine uses these formulas to recalculate key metrics:

### Market Cap
```
Market Cap = Price × Shares Outstanding
```

### Enterprise Value (EV)
```
EV = Market Cap + Debt - Cash
```

### P/E Ratio
```
P/E = Price / EPS
```

### EV/Sales Ratio
```
EV/Sales = Enterprise Value / Revenue
```

### EV/EBITDA Ratio
```
EV/EBITDA = Enterprise Value / EBITDA
```

## Severity Levels

The validation engine uses four severity levels:

1. **INFO**: No issues, or minor discrepancies within acceptable ranges
2. **WARNING**: Discrepancies exceed warning thresholds but are below blocking thresholds
3. **ERROR**: Discrepancies exceed blocking thresholds, requiring attention
4. **CRITICAL**: Severe issues that could indicate data corruption or major problems

## Integration Points

The validation engine is integrated at these key points:

1. **Data Ingestion**: In `get_comps_data()` function
2. **Analysis Pipeline**: In `generate_comps_report()` function
3. **Reporting**: In `generate_excel_report()` function

## Usage Example

```python
from validation_engine import ValidationEngine

# Initialize with default config
validator = ValidationEngine()

# Validate company data
result = validator.validate_company_data(company_data)

# Check if validation passed
if result.passed:
    # Proceed with analysis
    proceed_with_analysis(company_data)
else:
    # Handle validation failures
    for issue in result.issues:
        print(f"Validation issue: {issue.metric}, expected={issue.expected}, actual={issue.actual}")
```

## Validation Output Example

Example validation output in the Excel report:

| Ticker | Company Name | Validation Status | Issues Count | Blocking Issues |
|--------|--------------|------------------|--------------|-----------------|
| AAPL   | Apple Inc.   | PASSED           | 0            | 0               |
| MSFT   | Microsoft    | WARNING          | 2            | 0               |
| GOOGL  | Alphabet     | FAILED           | 3            | 1               |

## Troubleshooting

If validation fails:

1. Check the "Validation Results" tab in the Excel report
2. Review the detailed validation log file
3. Inspect the raw data sources for potential issues
4. Adjust tolerance thresholds if necessary