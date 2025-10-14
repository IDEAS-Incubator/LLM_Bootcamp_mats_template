#!/usr/bin/env python3
"""
Main entry point for the Multi-Agent Trading System (MATS)
"""

import sys
import os
import asyncio
from datetime import datetime
import logging
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agents.multi_agent_system import main as trading_main


def setup_logging():
    """Setup logging for the main application"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "trading_system.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


async def main():
    """Main function that runs the Multi-Agent Trading System"""
    logger = setup_logging()
    logger.info("Starting Multi-Agent Trading System (MATS)...")
    
    print("=" * 60)
    print("MULTI-AGENT TRADING SYSTEM (MATS) STARTING")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Features: Moving Average Strategy + Real Data Fetching")
    print("Teams: BTC, SOL, DOGE")
    print("Database: MongoDB (if configured)")
    print("=" * 60)

    # Run the main trading system
    try:
        await trading_main()
    except Exception as e:
        logger.error(f"Error in trading system: {str(e)}")
        print(f"\nError in trading system: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down Multi-Agent Trading System...")
        print("Thank you for using MATS!")
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        print("Please check the logs for more details.")
