from typing import Optional, List, Dict
from datetime import datetime
from core.data_models import PriceSnapshot, PriceDataPoint
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
import os
from dotenv import load_dotenv
import time
from utils.logging_utils import trading_logger

load_dotenv()

class MongoDBService:
    def __init__(self):
        # Get MongoDB connection details from environment variables
        mongo_uri = os.getenv("MONGODB_URI")
        db_name = os.getenv("MONGODB_DB_NAME")
        
        # Initialize MongoDB client
        self.client = MongoClient(mongo_uri)
        self.db: Database = self.client[db_name]
        
        # Initialize collections
        self.price_snapshots: Collection = self.db.price_snapshots
        self.optimized_parameters: Collection = self.db.optimized_parameters
        self.portfolio_config: Collection = self.db.portfolio_config
        self.orders: Collection = self.db.orders
        
        # Create indexes
        self._create_indexes()

    def _create_indexes(self):
        """Create necessary indexes for efficient querying"""
        # Create unique index on token_symbol to ensure one entry per token
        self.price_snapshots.create_index(
            [("token_symbol", 1)],
            unique=True
        )
        
        # Create index on timestamp for optimized parameters
        self.optimized_parameters.create_index(
            [("timestamp", -1)]
        )
        
        # Create unique index on token for portfolio config
        self.portfolio_config.create_index(
            [("token", 1)],
            unique=True
        )

    def save_price_snapshot(self, snapshot: PriceSnapshot) -> str:
        """Save or update a price snapshot for a token"""
        # Get existing snapshot if it exists
        existing_snapshot = self.price_snapshots.find_one(
            {"token_symbol": snapshot.token_symbol}
        )
        
        if existing_snapshot:
            # Convert existing price data to PriceDataPoint objects
            existing_price_data = [
                PriceDataPoint(**data) for data in existing_snapshot["price_data"]
            ]
            
            # Append new price data point
            existing_price_data.append(snapshot.price_data[0])
            
            # Update the snapshot with new price data
            result = self.price_snapshots.update_one(
                {"token_symbol": snapshot.token_symbol},
                {
                    "$set": {
                        "price_data": [data.dict() for data in existing_price_data],
                        "updated_at": datetime.utcnow().isoformat()
                    }
                }
            )
            return str(existing_snapshot["_id"])
        else:
            # Convert PriceSnapshot to dictionary for new document
            snapshot_dict = snapshot.dict()
            snapshot_dict["created_at"] = snapshot_dict["created_at"].isoformat()
            snapshot_dict["updated_at"] = snapshot_dict["updated_at"].isoformat()
            
            # Insert new document
            result = self.price_snapshots.insert_one(snapshot_dict)
            return str(result.inserted_id)

    def get_latest_snapshot(self, token_symbol: str) -> Optional[PriceSnapshot]:
        """Get the price snapshot for a token"""
        # Get the snapshot
        snapshot_doc = self.price_snapshots.find_one(
            {"token_symbol": token_symbol}
        )
        
        if not snapshot_doc:
            return None
        
        # Convert to PriceSnapshot object
        return PriceSnapshot(
            token_symbol=snapshot_doc["token_symbol"],
            source=snapshot_doc["source"],
            metadata=snapshot_doc["metadata"],
            price_data=[
                PriceDataPoint(
                    datetime=dp["datetime"],
                    price=dp["price"],
                    open=dp["open"],
                    high=dp["high"],
                    low=dp["low"],
                    volume=dp["volume"],
                    price_change=dp["price_change"]
                ) for dp in snapshot_doc["price_data"]
            ]
        )

    def get_optimized_parameters(self, symbol: str) -> Optional[Dict]:
        """Get the latest optimized parameters for a symbol"""
        return self.optimized_parameters.find_one(
            {"symbol": symbol},
            sort=[("timestamp", -1)]  # Get the most recent
        )

    def get_token_allocation(self, token: str) -> Optional[Dict]:
        """Get token allocation from portfolio config"""
        return self.portfolio_config.find_one({"token": token})

    def update_token_allocation(self, token: str, update_data: Dict) -> None:
        """Update token allocation in portfolio config"""
        try:
            trading_logger.log_message(
                "database",
                f"Attempting to update token allocation for {token} with data: {update_data}",
                "INFO"
            )
            
            # First verify the token exists
            existing = self.portfolio_config.find_one({"token": token})
            if not existing:
                trading_logger.log_message(
                    "database",
                    f"Token {token} not found in portfolio config",
                    "ERROR"
                )
                return
            
            # Log current values before update
            trading_logger.log_message(
                "database",
                f"Current values for {token}: available_capital={existing.get('available_capital')}, current_position={existing.get('current_position')}",
                "INFO"
            )
            
            # Perform the update
            result = self.portfolio_config.update_one(
                {"token": token},
                {"$set": update_data}
            )
            
            # Verify the update was successful
            if result.modified_count == 1:
                trading_logger.log_message(
                    "database",
                    f"Successfully updated token allocation for {token}",
                    "INFO"
                )
                
                # Log the new values
                updated = self.portfolio_config.find_one({"token": token})
                trading_logger.log_message(
                    "database",
                    f"New values for {token}: available_capital={updated.get('available_capital')}, current_position={updated.get('current_position')}",
                    "INFO"
                )
            else:
                trading_logger.log_message(
                    "database",
                    f"Failed to update token allocation for {token}. Modified count: {result.modified_count}",
                    "ERROR"
                )
                
        except Exception as e:
            trading_logger.log_message(
                "database",
                f"Error updating token allocation for {token}: {str(e)}",
                "ERROR"
            )
            raise  # Re-raise the exception to be handled by the caller

    def save_order(self, order_data: Dict) -> str:
        """Save an order to the database"""
        result = self.orders.insert_one(order_data)
        return str(result.inserted_id)

    def close(self):
        """Close the MongoDB connection"""
        self.client.close()

    def initialize_portfolio(self, token: str, initial_capital: float = 10000.0) -> None:
        """Initialize portfolio configuration for a token if it doesn't exist"""
        try:
            # Check if token exists
            existing = self.portfolio_config.find_one({"token": token})
            if existing:
                trading_logger.log_message(
                    "database",
                    f"Portfolio already exists for {token}",
                    "INFO"
                )
                return
                
            # Create initial portfolio configuration
            portfolio_config = {
                "token": token,
                "available_capital": initial_capital,
                "current_position": 0.0,
                "last_trade_price": 0.0,
                "last_trade_type": "none",
                "last_trade_time": datetime.utcnow().isoformat()
            }
            
            # Insert the configuration
            result = self.portfolio_config.insert_one(portfolio_config)
            
            trading_logger.log_message(
                "database",
                f"Initialized portfolio for {token} with ID {result.inserted_id}",
                "INFO"
            )
            
        except Exception as e:
            trading_logger.log_message(
                "database",
                f"Error initializing portfolio for {token}: {str(e)}",
                "ERROR"
            )
            raise 