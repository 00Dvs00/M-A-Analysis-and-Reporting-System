import os
import pandas as pd
import numpy as np
from datetime import datetime
from etl_pipeline import (
    generate_distribution_plot,
    generate_box_plot,
    generate_pdf_report,
    generate_comps_report,
    bootstrap_confidence_intervals,
    add_quality_flags_and_completeness
)

def enhanced_comps_report(financials_df, target_ticker=None, output_dir='reports'):
    """
    Generate an enhanced comparables analysis report using the data processing pipeline.
    
    Args:
        financials_df (pd.DataFrame): DataFrame containing financial data for companies.
        target_ticker (str, optional): Ticker of the target company. If None, uses the first company.
        output_dir (str, optional): Directory to save reports. Defaults to 'reports'.
        
    Returns:
        str: Path to the generated report
        list: Warnings or issues encountered during report generation
    """
    warnings = []
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the target ticker if not provided
    if target_ticker is None and not financials_df.empty:
        target_ticker = financials_df['ticker'].iloc[0]
        warnings.append(f"No target ticker specified, using {target_ticker} as default")
    
    try:
        # Add quality flags and completeness metrics
        financials_df = add_quality_flags_and_completeness(financials_df)
        
        # Define metrics to include in the report
        metrics = [col for col in financials_df.columns if any(x in col for x in ['ev_to_', 'pe_', 'peg_', 'roic', 'yield'])]
        
        if not metrics:
            warnings.append("No valuation metrics found in the data. Report may be limited.")
            
        # Generate the report
        report_path = generate_comps_report(target_ticker, financials_df, metrics, output_dir)
        
        return report_path, warnings
    
    except Exception as e:
        warnings.append(f"Error generating report: {str(e)}")
        return None, warnings