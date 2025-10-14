#!/usr/bin/env python3
"""
Historical Data Loading Entry Point
"""

import sys
import os
import asyncio

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from analysis.historical_data_loader import main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down historical data loader...")
    except Exception as e:
        print(f"\nError in historical data loading: {str(e)}") 