#!/usr/bin/env python3
"""
Enhancement for Excel Reports to Include Validation Results
==========================================================

This module adds the validation results tab to Excel reports to provide
transparency about data quality and validation checks.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Any
import os
from datetime import datetime
from src.validation.validation_engine import ValidationResult

def add_validation_tab(writer, validation_results: Dict[str, ValidationResult], 
                     peer_data: pd.DataFrame):
    """
    Adds a validation results tab to an Excel report.
    
    Args:
        writer: The Excel writer object
        validation_results: Dictionary mapping tickers to ValidationResult objects
        peer_data: DataFrame containing peer data
    """
    if not validation_results:
        logging.info("No validation results to add to report")
        return
    
    workbook = writer.book
    
    # Create formats
    header_format = workbook.add_format({
        'bold': True,
        'font_size': 12,
        'bg_color': '#4F81BD',
        'font_color': 'white',
        'border': 1,
        'align': 'center'
    })
    
    cell_format = workbook.add_format({
        'border': 1,
        'align': 'center'
    })
    
    error_format = workbook.add_format({
        'bg_color': '#FFC7CE',
        'font_color': '#9C0006',
        'border': 1,
        'align': 'center'
    })
    
    warning_format = workbook.add_format({
        'bg_color': '#FFEB9C',
        'font_color': '#9C6500',
        'border': 1,
        'align': 'center'
    })
    
    success_format = workbook.add_format({
        'bg_color': '#C6EFCE',
        'font_color': '#006100',
        'border': 1,
        'align': 'center'
    })
    
    percentage_format = workbook.add_format({
        'num_format': '0.00%',
        'border': 1,
        'align': 'right'
    })
    
    title_format = workbook.add_format({
        'bold': True,
        'font_size': 16,
        'font_color': '#4F81BD'
    })
    
    # Create validation overview tab
    validation_sheet = workbook.add_worksheet('Validation Results')
    
    # Add title
    validation_sheet.write('A1', 'Data Validation Results', title_format)
    validation_sheet.write('A2', 'This sheet shows the results of data validation checks performed on the companies in this report.', cell_format)
    
    # Headers for validation summary
    validation_sheet.write('A4', 'Ticker', header_format)
    validation_sheet.write('B4', 'Company Name', header_format)
    validation_sheet.write('C4', 'Validation Status', header_format)
    validation_sheet.write('D4', 'Issues Count', header_format)
    validation_sheet.write('E4', 'Blocking Issues', header_format)
    
    # Add data
    row = 4
    for ticker, result in validation_results.items():
        company_name = peer_data.loc[peer_data['Ticker'] == ticker, 'Company Name'].iloc[0] if ticker in peer_data['Ticker'].values else ticker
        
        validation_sheet.write(row, 0, ticker, cell_format)
        validation_sheet.write(row, 1, company_name, cell_format)
        
        # Status with conditional formatting
        if result.passed:
            validation_sheet.write(row, 2, 'PASSED', success_format)
        else:
            validation_sheet.write(row, 2, 'FAILED', error_format)
        
        # Issues count
        validation_sheet.write(row, 3, len(result.issues), cell_format)
        
        # Blocking issues
        blocking_issues = sum(1 for issue in result.issues if issue.severity.value in ('ERROR', 'CRITICAL'))
        if blocking_issues > 0:
            validation_sheet.write(row, 4, blocking_issues, error_format)
        else:
            validation_sheet.write(row, 4, blocking_issues, cell_format)
        
        row += 1
    
    # Add detailed issues table
    row += 2
    validation_sheet.write(row, 0, 'Detailed Validation Issues', title_format)
    row += 1
    
    # Headers for issues
    validation_sheet.write(row, 0, 'Ticker', header_format)
    validation_sheet.write(row, 1, 'Metric', header_format)
    validation_sheet.write(row, 2, 'Expected Value', header_format)
    validation_sheet.write(row, 3, 'Actual Value', header_format)
    validation_sheet.write(row, 4, 'Difference (%)', header_format)
    validation_sheet.write(row, 5, 'Threshold (%)', header_format)
    validation_sheet.write(row, 6, 'Severity', header_format)
    
    # Add issue data
    row += 1
    for ticker, result in validation_results.items():
        for issue in result.issues:
            validation_sheet.write(row, 0, ticker, cell_format)
            validation_sheet.write(row, 1, issue.metric, cell_format)
            validation_sheet.write(row, 2, issue.expected, cell_format)
            validation_sheet.write(row, 3, issue.actual, cell_format)
            validation_sheet.write(row, 4, issue.pct_difference, percentage_format)
            validation_sheet.write(row, 5, issue.threshold, percentage_format)
            
            # Format severity
            if issue.severity.value == 'ERROR' or issue.severity.value == 'CRITICAL':
                validation_sheet.write(row, 6, issue.severity.value, error_format)
            elif issue.severity.value == 'WARNING':
                validation_sheet.write(row, 6, issue.severity.value, warning_format)
            else:
                validation_sheet.write(row, 6, issue.severity.value, cell_format)
            
            row += 1
    
    # Set column widths
    validation_sheet.set_column('A:A', 10)
    validation_sheet.set_column('B:B', 25)
    validation_sheet.set_column('C:C', 15)
    validation_sheet.set_column('D:D', 12)
    validation_sheet.set_column('E:E', 15)
    validation_sheet.set_column('F:F', 15)
    validation_sheet.set_column('G:G', 12)
    
    logging.info(f"Added validation results tab with {sum(len(r.issues) for r in validation_results.values())} total issues")

def create_validation_log_file(validation_results: Dict[str, ValidationResult], 
                             output_dir: str = 'validation_logs'):
    """
    Creates a detailed validation log file for audit purposes.
    
    Args:
        validation_results: Dictionary mapping tickers to ValidationResult objects
        output_dir: Directory to save the log file
        
    Returns:
        str: Path to the created log file
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"validation_log_{timestamp}.txt")
    
    with open(log_path, 'w') as f:
        f.write(f"VALIDATION LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Summary stats
        companies_validated = len(validation_results)
        companies_passed = sum(1 for r in validation_results.values() if r.passed)
        companies_failed = companies_validated - companies_passed
        total_issues = sum(len(r.issues) for r in validation_results.values())
        
        f.write(f"Companies Validated: {companies_validated}\n")
        f.write(f"Companies Passed: {companies_passed}\n")
        f.write(f"Companies Failed: {companies_failed}\n")
        f.write(f"Total Issues Found: {total_issues}\n\n")
        
        f.write("DETAILED RESULTS BY COMPANY\n")
        f.write("-"*80 + "\n\n")
        
        # Company by company details
        for ticker, result in validation_results.items():
            f.write(f"TICKER: {ticker}\n")
            f.write(f"Status: {'PASSED' if result.passed else 'FAILED'}\n")
            f.write(f"Issues: {len(result.issues)}\n")
            
            if result.issues:
                f.write("\nDetailed Issues:\n")
                for i, issue in enumerate(result.issues, 1):
                    f.write(f"  {i}. {issue.metric}:\n")
                    f.write(f"     Expected: {issue.expected}\n")
                    f.write(f"     Actual: {issue.actual}\n")
                    f.write(f"     Difference: {issue.pct_difference:.2%}\n")
                    f.write(f"     Severity: {issue.severity.value}\n")
                    if issue.details:
                        f.write(f"     Details: {issue.details}\n")
                    f.write("\n")
            
            f.write("="*80 + "\n\n")
    
    logging.info(f"Created validation log file: {log_path}")
    return log_path