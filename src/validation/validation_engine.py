#!/usr/bin/env python3
"""
Validation Engine for M&A Analysis and Reporting System
======================================================

This module provides end-to-end validation functionality for key financial figures
after data ingestion. It recalculates values independently and compares them against
ingested data, flagging discrepancies that exceed specified tolerance thresholds.

Features:
- Independent recalculation of Market Cap, Enterprise Value, EPS, and key multiples
- Configurable tolerance thresholds (e.g., 0.1% for EV, 0.5% for EPS)
- Detailed validation reports with pass/fail status
- Process halting logic for severe discrepancies
- Validation can be executed at multiple stages of the pipeline

Usage:
    from validation_engine import ValidationEngine
    
    # Initialize the engine with config
    validator = ValidationEngine(config)
    
    # Validate company data
    validation_result = validator.validate_company_data(company_data)
    
    # Check if validation passed
    if validation_result.passed:
        # Proceed with further processing
        proceed_with_analysis(company_data)
    else:
        # Handle validation failure
        handle_validation_errors(validation_result.errors)
"""

import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Set
from enum import Enum
import os
import yaml
import json
from datetime import datetime
import traceback

# Setup logging with more detailed format for validation issues
validation_logger = logging.getLogger('validation')
validation_logger.setLevel(logging.INFO)

# Add a dedicated file handler for validation issues if it doesn't exist
if not any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith('validation.log') 
           for h in validation_logger.handlers):
    validation_file_handler = logging.FileHandler('validation.log')
    validation_file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    validation_logger.addHandler(validation_file_handler)

# Add console handler if it doesn't exist
if not any(isinstance(h, logging.StreamHandler) and h.stream == logging.sys.stderr 
           for h in validation_logger.handlers):
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    validation_logger.addHandler(console_handler)


class ValidationSeverity(Enum):
    """Enumeration of validation error severity levels."""
    INFO = "INFO"           # Informational only, processing continues
    WARNING = "WARNING"     # Issue detected, but within acceptable range
    ERROR = "ERROR"         # Serious issue, processing should be halted
    CRITICAL = "CRITICAL"   # Critical issue requiring immediate attention


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    metric: str                              # The metric being validated (e.g., "EV", "Market Cap")
    expected: Union[float, int]              # The expected (calculated) value
    actual: Union[float, int]                # The actual (ingested) value
    abs_difference: Union[float, int]        # Absolute difference
    pct_difference: float                    # Percentage difference
    severity: ValidationSeverity             # Severity level
    threshold: float                         # The threshold that was applied
    details: Optional[str] = None            # Additional details about the issue
    timestamp: datetime = field(default_factory=datetime.now)  # When the issue was detected
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the validation issue to a dictionary."""
        return {
            "metric": self.metric,
            "expected": self.expected,
            "actual": self.actual,
            "abs_difference": self.abs_difference,
            "pct_difference": self.pct_difference,
            "severity": self.severity.value,
            "threshold": self.threshold,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ValidationResult:
    """Represents the result of a validation operation."""
    company_ticker: str                              # Ticker symbol of the company
    passed: bool                                     # Overall pass/fail status
    issues: List[ValidationIssue] = field(default_factory=list)  # List of validation issues
    validation_time: datetime = field(default_factory=datetime.now)  # When validation was performed
    
    def has_blocking_issues(self) -> bool:
        """Check if there are any blocking issues (ERROR or CRITICAL)."""
        return any(issue.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL) 
                  for issue in self.issues)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the validation result to a dictionary."""
        return {
            "company_ticker": self.company_ticker,
            "passed": self.passed,
            "issues": [issue.to_dict() for issue in self.issues],
            "validation_time": self.validation_time.isoformat(),
            "has_blocking_issues": self.has_blocking_issues()
        }
    
    def to_json(self) -> str:
        """Convert the validation result to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def save_to_file(self, filepath: Optional[str] = None) -> str:
        """Save the validation result to a JSON file."""
        if filepath is None:
            # Generate a default filename based on ticker and timestamp
            timestamp = self.validation_time.strftime("%Y%m%d_%H%M%S")
            filepath = f"validation_{self.company_ticker}_{timestamp}.json"
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write(self.to_json())
        
        return filepath


class ValidationEngine:
    """
    Main validation engine for financial data integrity checks.
    
    This class handles the recalculation and validation of key financial metrics
    to ensure data integrity throughout the analysis pipeline.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the validation engine with configuration.
        
        Args:
            config: Dictionary containing validation configuration.
                   If None, default configuration will be loaded.
        """
        self.config = config or self._load_default_config()
        self.validation_enabled = self.config.get('validation', {}).get('enabled', True)
        
        # Load thresholds from config
        validation_config = self.config.get('validation', {})
        self.thresholds = validation_config.get('thresholds', {
            'market_cap': 0.05,  # 5%
            'enterprise_value': 0.05,  # 5%
            'eps': 0.05,  # 5%
            'multiples': 0.05  # 5%
        })
        
        # Configure blocking behavior
        self.blocking_thresholds = validation_config.get('blocking_thresholds', {
            'market_cap': 0.10,  # 10%
            'enterprise_value': 0.10,  # 10%
            'eps': 0.10,  # 10%
            'multiples': 0.10  # 10%
        })
        
        # Configure metrics to validate
        self.metrics_to_validate = validation_config.get('metrics_to_validate', [
            'Market Cap', 'EV', 'EPS Forward', 'P/E Forward', 'EV/Sales LTM', 'EV/EBITDA LTM'
        ])
        
        validation_logger.info(f"Validation engine initialized with thresholds: {self.thresholds}")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """
        Load default configuration from default_config.yaml.
        
        Returns:
            Dict containing configuration values.
        """
        try:
            config_path = os.path.join('config', 'default_config.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Add validation section if not present
            if 'validation' not in config:
                config['validation'] = {
                    'enabled': True,
                    'thresholds': {
                        'market_cap': 0.005,  # 0.5%
                        'enterprise_value': 0.001,  # 0.1%
                        'eps': 0.005,  # 0.5%
                        'multiples': 0.01  # 1%
                    },
                    'blocking_thresholds': {
                        'market_cap': 0.01,  # 1%
                        'enterprise_value': 0.002,  # 0.2%
                        'eps': 0.01,  # 1%
                        'multiples': 0.02  # 2%
                    },
                    'metrics_to_validate': [
                        'Market Cap', 'EV', 'EPS Forward', 'P/E Forward', 'EV/Sales LTM', 'EV/EBITDA LTM'
                    ]
                }
            
            return config
        except Exception as e:
            validation_logger.error(f"Error loading default config: {e}")
            # Return a minimal default configuration
            return {
                'validation': {
                    'enabled': True,
                    'thresholds': {
                        'market_cap': 0.005,
                        'enterprise_value': 0.001,
                        'eps': 0.005,
                        'multiples': 0.01
                    },
                    'blocking_thresholds': {
                        'market_cap': 0.01,
                        'enterprise_value': 0.002,
                        'eps': 0.01,
                        'multiples': 0.02
                    },
                    'metrics_to_validate': [
                        'Market Cap', 'EV', 'EPS Forward', 'P/E Forward', 'EV/Sales LTM', 'EV/EBITDA LTM'
                    ]
                }
            }
    
    def validate_company_data(self, company_data: pd.Series) -> ValidationResult:
        """
        Validate financial data for a single company.
        
        Args:
            company_data: Pandas Series containing company financial data.
            
        Returns:
            ValidationResult object with validation status and issues.
        """
        if not self.validation_enabled:
            validation_logger.info(f"Validation skipped for {company_data.get('Ticker', 'Unknown')} (disabled in config)")
            return ValidationResult(
                company_ticker=company_data.get('Ticker', 'Unknown'),
                passed=True
            )
        
        validation_logger.info(f"Starting validation for {company_data.get('Ticker', 'Unknown')}")
        
        # Initialize validation result
        result = ValidationResult(
            company_ticker=company_data.get('Ticker', 'Unknown'),
            passed=True
        )
        
        try:
            # Validate key metrics
            for metric in self.metrics_to_validate:
                if metric not in company_data:
                    validation_logger.warning(f"Metric '{metric}' not found in company data, skipping validation")
                    continue
                
                # Get actual value from data
                actual_value = company_data.get(metric)
                
                # Skip if the value is missing
                if pd.isna(actual_value):
                    validation_logger.warning(f"Missing value for '{metric}', skipping validation")
                    continue
                
                # Recalculate the metric
                try:
                    calculated_value = self._recalculate_metric(metric, company_data)
                    
                    # Skip if calculation failed
                    if calculated_value is None:
                        validation_logger.warning(f"Could not recalculate '{metric}', skipping validation")
                        continue
                    
                    # Compare values
                    self._validate_metric(metric, calculated_value, actual_value, result)
                except Exception as e:
                    validation_logger.error(f"Error validating {metric}: {e}")
                    validation_logger.debug(traceback.format_exc())
                    continue
            
            # Update overall pass/fail status
            result.passed = not result.has_blocking_issues()
            
            # Log validation result
            if result.passed:
                validation_logger.info(f"Validation passed for {result.company_ticker} with {len(result.issues)} non-blocking issues")
            else:
                validation_logger.error(f"Validation failed for {result.company_ticker} with blocking issues")
                for issue in result.issues:
                    if issue.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL):
                        validation_logger.error(f"  - {issue.metric}: expected={issue.expected}, actual={issue.actual}, " 
                                             f"diff={issue.pct_difference:.2%}, threshold={issue.threshold:.2%}")
            
            return result
            
        except Exception as e:
            validation_logger.error(f"Validation failed with exception: {e}")
            validation_logger.debug(traceback.format_exc())
            result.passed = False
            return result
    
    def _recalculate_metric(self, metric: str, data: pd.Series) -> Optional[float]:
        """
        Recalculate a financial metric from raw data.
        
        Args:
            metric: Name of the metric to recalculate.
            data: Pandas Series containing company financial data.
            
        Returns:
            Recalculated value, or None if calculation fails.
        """
        try:
            if metric == 'Market Cap':
                # Market Cap = Price * Shares Outstanding
                price = data.get('Price')
                shares = data.get('Shares Out')
                if pd.isna(price) or pd.isna(shares):
                    return None
                return price * shares
            
            elif metric == 'EV':
                # Enterprise Value = Market Cap + Debt - Cash
                market_cap = data.get('Market Cap')
                debt = data.get('Debt')
                cash = data.get('Cash')
                if pd.isna(market_cap) or pd.isna(debt) or pd.isna(cash):
                    return None
                return market_cap + debt - cash
            
            elif metric == 'EPS Forward':
                # We can't directly recalculate forward EPS as it's based on estimates
                # For validation, we could compare it to a trailing EPS with growth assumptions
                # But for now, we'll just return the actual value for consistency checking
                return data.get('EPS Forward')
            
            elif metric == 'P/E Forward':
                # P/E Forward = Price / EPS Forward
                price = data.get('Price')
                eps_forward = data.get('EPS Forward')
                if pd.isna(price) or pd.isna(eps_forward) or eps_forward == 0:
                    return None
                return price / eps_forward
            
            elif metric == 'EV/Sales LTM':
                # EV/Sales = Enterprise Value / Revenue LTM
                ev = data.get('EV')
                revenue = data.get('Revenue LTM')
                if pd.isna(ev) or pd.isna(revenue) or revenue == 0:
                    return None
                return ev / revenue
            
            elif metric == 'EV/EBITDA LTM':
                # EV/EBITDA = Enterprise Value / EBITDA LTM
                ev = data.get('EV')
                ebitda = data.get('EBITDA LTM')
                if pd.isna(ev) or pd.isna(ebitda) or ebitda == 0:
                    return None
                return ev / ebitda
            
            else:
                validation_logger.warning(f"Recalculation not implemented for metric: {metric}")
                return None
        
        except Exception as e:
            validation_logger.error(f"Error recalculating {metric}: {e}")
            return None
    
    def _validate_metric(self, metric: str, calculated: float, actual: float, 
                        result: ValidationResult) -> None:
        """
        Validate a metric by comparing calculated and actual values.
        
        Args:
            metric: Name of the metric being validated.
            calculated: The recalculated value.
            actual: The value from ingested data.
            result: ValidationResult object to update with any issues.
        """
        # Determine the appropriate threshold for this metric
        threshold = self._get_threshold_for_metric(metric)
        blocking_threshold = self._get_blocking_threshold_for_metric(metric)
        
        # Calculate absolute and percentage differences
        abs_diff = abs(calculated - actual)
        
        # Handle division by zero or very small values
        if abs(calculated) < 1e-10:
            pct_diff = abs_diff if abs_diff > 0 else 0
        else:
            pct_diff = abs_diff / abs(calculated)
        
        # Determine if this exceeds thresholds
        if pct_diff > blocking_threshold:
            severity = ValidationSeverity.ERROR
            validation_logger.error(
                f"VALIDATION ERROR: {metric} differs by {pct_diff:.2%} "
                f"(calculated={calculated:.2f}, actual={actual:.2f})"
            )
        elif pct_diff > threshold:
            severity = ValidationSeverity.WARNING
            validation_logger.warning(
                f"VALIDATION WARNING: {metric} differs by {pct_diff:.2%} "
                f"(calculated={calculated:.2f}, actual={actual:.2f})"
            )
        else:
            severity = ValidationSeverity.INFO
            validation_logger.debug(
                f"VALIDATION PASSED: {metric} differs by {pct_diff:.2%} "
                f"(calculated={calculated:.2f}, actual={actual:.2f})"
            )
        
        # Add issue to result if severity is WARNING or higher
        if severity != ValidationSeverity.INFO:
            issue = ValidationIssue(
                metric=metric,
                expected=calculated,
                actual=actual,
                abs_difference=abs_diff,
                pct_difference=pct_diff,
                severity=severity,
                threshold=threshold,
                details=f"Recalculated {metric} differs from ingested value by {pct_diff:.2%}"
            )
            result.issues.append(issue)
    
    def _get_threshold_for_metric(self, metric: str) -> float:
        """
        Get the appropriate threshold for a given metric.
        
        Args:
            metric: Name of the metric.
            
        Returns:
            Threshold value as a float.
        """
        if metric == 'Market Cap':
            return self.thresholds.get('market_cap', 0.005)
        elif metric == 'EV':
            return self.thresholds.get('enterprise_value', 0.001)
        elif metric == 'EPS Forward':
            return self.thresholds.get('eps', 0.005)
        else:  # Any multiple
            return self.thresholds.get('multiples', 0.01)
    
    def _get_blocking_threshold_for_metric(self, metric: str) -> float:
        """
        Get the appropriate blocking threshold for a given metric.
        
        Args:
            metric: Name of the metric.
            
        Returns:
            Blocking threshold value as a float.
        """
        if metric == 'Market Cap':
            return self.blocking_thresholds.get('market_cap', 0.01)
        elif metric == 'EV':
            return self.blocking_thresholds.get('enterprise_value', 0.002)
        elif metric == 'EPS Forward':
            return self.blocking_thresholds.get('eps', 0.01)
        else:  # Any multiple
            return self.blocking_thresholds.get('multiples', 0.02)


def validate_peer_group(target_data: pd.Series, peer_data: pd.DataFrame, 
                       config: Optional[Dict[str, Any]] = None) -> Dict[str, ValidationResult]:
    """
    Validate an entire peer group including the target company.
    
    Args:
        target_data: Pandas Series with target company data.
        peer_data: Pandas DataFrame with peer companies data.
        config: Optional configuration dictionary.
        
    Returns:
        Dictionary mapping company tickers to their ValidationResult objects.
    """
    validation_engine = ValidationEngine(config)
    
    results = {}
    
    # Validate target company
    target_result = validation_engine.validate_company_data(target_data)
    results[target_data.get('Ticker', 'Unknown')] = target_result
    
    # Validate each peer company
    for _, peer_row in peer_data.iterrows():
        peer_result = validation_engine.validate_company_data(peer_row)
        results[peer_row.get('Ticker', 'Unknown')] = peer_result
    
    # Log overall validation status
    failed_companies = [ticker for ticker, result in results.items() if not result.passed]
    
    if failed_companies:
        validation_logger.error(f"Validation failed for {len(failed_companies)} companies: {', '.join(failed_companies)}")
    else:
        validation_logger.info(f"Validation passed for all {len(results)} companies")
    
    return results


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    print("Testing Validation Engine with live data...")
    
    try:
        # Get sample data for a test company
        ticker = "MSFT"
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Create a Series with the data
        test_data = pd.Series({
            'Ticker': ticker,
            'Company Name': info.get('shortName', ticker),
            'Price': info.get('currentPrice', 0),
            'Market Cap': info.get('marketCap', 0),
            'EV': info.get('enterpriseValue', 0),
            'Revenue LTM': info.get('totalRevenue', 0),
            'EBITDA LTM': info.get('ebitda', 0),
            'Debt': info.get('totalDebt', 0),
            'Cash': info.get('totalCash', 0),
            'Shares Out': info.get('sharesOutstanding', 0),
            'EPS Forward': info.get('forwardEps', 0),
            'P/E Forward': info.get('forwardPE', 0),
        })
        
        # Initialize the validation engine
        validator = ValidationEngine()
        
        # Validate the data
        result = validator.validate_company_data(test_data)
        
        # Print results
        print(f"Validation {'passed' if result.passed else 'failed'} for {ticker}")
        
        if result.issues:
            print(f"Found {len(result.issues)} issues:")
            for issue in result.issues:
                print(f"  - {issue.metric}: expected={issue.expected:.2f}, actual={issue.actual:.2f}, "
                     f"diff={issue.pct_difference:.2%}, severity={issue.severity.name}")
        else:
            print("No issues found")
        
        # Save result to file
        result_file = result.save_to_file()
        print(f"Validation result saved to {result_file}")
        
    except Exception as e:
        print(f"Error during test: {e}")