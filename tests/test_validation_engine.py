#!/usr/bin/env python3
"""
Test Module for Validation Engine
================================

This script tests the validation engine functionality by creating example data
with known discrepancies and validating the results.
"""

import pandas as pd
import numpy as np
import logging
import os
import sys
import yaml
from datetime import datetime
import pytest

# Add parent directory to path to import local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import validation engine
from src.validation.validation_engine import ValidationEngine, ValidationResult, ValidationSeverity, validate_peer_group

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create test directory if it doesn't exist
os.makedirs('test_results', exist_ok=True)


def create_test_company_data():
    """Create test company data with intentional discrepancies."""
    
    # Base data with no discrepancies
    clean_data = pd.Series({
        'Ticker': 'TEST',
        'Company Name': 'Test Company',
        'Price': 100.0,
        'Shares Out': 1000000,  # 1M shares
        'Market Cap': 100000000.0,  # 100M = Price * Shares
        'Debt': 50000000.0,  # 50M
        'Cash': 20000000.0,  # 20M
        'EV': 130000000.0,  # 130M = Market Cap + Debt - Cash
        'Revenue LTM': 30000000.0,  # 30M
        'EBITDA LTM': 10000000.0,  # 10M
        'EPS Forward': 5.0,
        'P/E Forward': 20.0,  # = Price / EPS Forward
        'EV/Sales LTM': 4.333333,  # = EV / Revenue
        'EV/EBITDA LTM': 13.0,  # = EV / EBITDA
    })
    
    # Create variants with different severity of discrepancies
    
    # 1. Small discrepancies (within thresholds)
    small_discrepancy = clean_data.copy()
    small_discrepancy['Market Cap'] = 100900000.0  # 0.9% off
    small_discrepancy['EV'] = 130650000.0  # 0.5% off
    small_discrepancy['EPS Forward'] = 5.05  # 1% off
    small_discrepancy['Ticker'] = 'SMALL'

    # 2. Medium discrepancies (warnings but not blocking)
    medium_discrepancy = clean_data.copy()
    medium_discrepancy['Market Cap'] = 101000000.0  # 1% off (within blocking threshold)
    medium_discrepancy['EV'] = 130000100.0  # 0.00008% off (well within blocking threshold)
    medium_discrepancy['EPS Forward'] = 5.1  # 2% off
    medium_discrepancy['Ticker'] = 'MEDIUM'

    # 3. Large discrepancies (blocking)
    large_discrepancy = clean_data.copy()
    large_discrepancy['Market Cap'] = 103000000.0  # 3% off
    large_discrepancy['EV'] = 132000000.0  # 2% off
    large_discrepancy['EPS Forward'] = 5.5  # 10% off
    large_discrepancy['Ticker'] = 'LARGE'
    
    # Return as dictionary of Series
    return {
        'clean': clean_data,
        'small': small_discrepancy,
        'medium': medium_discrepancy,
        'large': large_discrepancy
    }


def test_validation_engine():
    """Test the validation engine with various data scenarios."""
    
    # Create test data
    test_data = create_test_company_data()
    
    # Load test configuration
    config = {
        'validation': {
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
    }
    
    # Initialize validation engine
    validator = ValidationEngine(config)
    
    # Test with clean data
    logger.info("Testing with clean data")
    clean_result = validator.validate_company_data(test_data['clean'])
    assert clean_result.passed
    assert len(clean_result.issues) == 0
    
    # Test with small discrepancies
    logger.info("Testing with small discrepancies")
    small_result = validator.validate_company_data(test_data['small'])
    assert small_result.passed
    # Expect some INFO level issues
    
    # Test with medium discrepancies
    logger.info("Testing with medium discrepancies")
    medium_result = validator.validate_company_data(test_data['medium'])
    assert not medium_result.passed  # We expect this to fail due to EV difference exceeding threshold
    assert any(issue.severity == ValidationSeverity.ERROR for issue in medium_result.issues)
    
    # Test with large discrepancies
    logger.info("Testing with large discrepancies")
    large_result = validator.validate_company_data(test_data['large'])
    assert not large_result.passed
    # Should have ERROR level issues and fail
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_result.save_to_file(f"test_results/clean_validation_{timestamp}.json")
    small_result.save_to_file(f"test_results/small_validation_{timestamp}.json")
    medium_result.save_to_file(f"test_results/medium_validation_{timestamp}.json")
    large_result.save_to_file(f"test_results/large_validation_{timestamp}.json")
    
    # Return results for further analysis
    return {
        'clean': clean_result,
        'small': small_result,
        'medium': medium_result,
        'large': large_result
    }


def test_peer_group_validation():
    """Test validation of a peer group."""
    
    # Create test data
    test_data = create_test_company_data()
    
    # Create a target and peer DataFrame
    target = test_data['clean']
    peers_dict = {
        'small': test_data['small'],
        'medium': test_data['medium'],
        'large': test_data['large']
    }
    
    peers_df = pd.DataFrame([peers_dict['small'], peers_dict['medium'], peers_dict['large']])
    
    # Validate the peer group
    results = validate_peer_group(target, peers_df)
    
    # Check results
    assert results['TEST'].passed
    assert results['SMALL'].passed
    assert results['MEDIUM'].passed
    assert not results['LARGE'].passed
    
    # Print summary
    logger.info("Peer Group Validation Results:")
    for ticker, result in results.items():
        logger.info(f"{ticker}: {'PASSED' if result.passed else 'FAILED'} with {len(result.issues)} issues")
    
    return results


def intentionally_break_calculations():
    """Test to ensure calculation validation logic works correctly."""
    
    # Create data with inconsistent calculations
    broken_data = pd.Series({
        'Ticker': 'BROKEN',
        'Company Name': 'Broken Calculations Inc.',
        'Price': 100.0,
        'Shares Out': 1000000,  # 1M shares
        # Market Cap should be 100M but is incorrectly set to 110M (10% off)
        'Market Cap': 110000000.0,
        'Debt': 50000000.0,
        'Cash': 20000000.0,
        # EV should be 130M but is incorrectly set to 150M (15.4% off)
        'EV': 150000000.0,
        'Revenue LTM': 30000000.0,
        'EBITDA LTM': 10000000.0,
        'EPS Forward': 5.0,
        # P/E Forward should be 20 but is incorrectly set to 25 (25% off)
        'P/E Forward': 25.0,
        # EV/Sales should be 4.33 but is incorrectly set to 6 (38.5% off)
        'EV/Sales LTM': 6.0,
        # EV/EBITDA should be 13 but is incorrectly set to 18 (38.5% off)
        'EV/EBITDA LTM': 18.0,
    })
    
    # Initialize validation engine with default config
    validator = ValidationEngine()
    
    # Validate the broken data
    result = validator.validate_company_data(broken_data)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result.save_to_file(f"test_results/broken_validation_{timestamp}.json")
    
    # Print results
    logger.info(f"Broken Data Validation: {'PASSED' if result.passed else 'FAILED'}")
    logger.info(f"Issues found: {len(result.issues)}")
    
    for issue in result.issues:
        logger.info(f"  - {issue.metric}: expected={issue.expected:.2f}, actual={issue.actual:.2f}, "
                   f"diff={issue.pct_difference:.2%}, severity={issue.severity.name}")
    
    # Assertions
    assert not result.passed
    assert len(result.issues) >= 5  # Should have at least 5 issues (one for each metric)
    
    return result


if __name__ == "__main__":
    print("Running Validation Engine Tests...")
    
    try:
        # Run tests
        validation_results = test_validation_engine()
        peer_results = test_peer_group_validation()
        broken_results = intentionally_break_calculations()
        
        # Print test summary
        print("\n===== TEST SUMMARY =====")
        print(f"Clean data validation: {'PASSED' if validation_results['clean'].passed else 'FAILED'}")
        print(f"Small discrepancy validation: {'PASSED' if validation_results['small'].passed else 'FAILED'}")
        print(f"Medium discrepancy validation: {'PASSED' if validation_results['medium'].passed else 'FAILED'}")
        print(f"Large discrepancy validation: {'PASSED' if validation_results['large'].passed else 'FAILED'}")
        print(f"Peer group validation: {'PASSED' if any(r.passed for r in peer_results.values()) else 'FAILED'}")
        print(f"Broken calculations: {'FAILED AS EXPECTED' if not broken_results.passed else 'ERROR - DID NOT FAIL'}")
        
        print("\nTests completed successfully.")
        
    except Exception as e:
        print(f"Error during tests: {e}")
        import traceback
        traceback.print_exc()