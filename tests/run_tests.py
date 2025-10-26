#!/usr/bin/env python3
"""
Test runner for Multi-Agent Crypto Trading System
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def run_tests(test_type=None, verbose=False):
    """Run tests with specified options"""
    
    # Add src directory to Python path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    # Add test type filter
    if test_type:
        if test_type == "unit":
            cmd.extend(["-m", "unit"])
        elif test_type == "integration":
            cmd.extend(["-m", "integration"])
        elif test_type == "edge":
            cmd.extend(["-m", "edge"])
        elif test_type == "cycle":
            cmd.extend(["-m", "cycle"])
        elif test_type == "all":
            pass  # Run all tests
        else:
            print(f"Unknown test type: {test_type}")
            return False
    
    # Add test directory
    cmd.append("tests/")
    
    print(f"Running tests with command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Tests failed with exit code: {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run tests for Multi-Agent Crypto Trading System")
    parser.add_argument(
        "--type", 
        choices=["unit", "integration", "edge", "cycle", "all"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Multi-Agent Crypto Trading System - Test Suite")
    print("=" * 60)
    
    success = run_tests(args.type, args.verbose)
    
    if success:
        print("\nAll tests completed successfully!")
        sys.exit(0)
    else:
        print("\nSome tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 