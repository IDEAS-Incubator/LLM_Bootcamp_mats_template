import yaml
from pathlib import Path
from datetime import datetime
from services.database_service import MongoDBService
from utils.logging_utils import trading_logger

def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / "config" / "trading_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def store_config():
    """Store token allocations from config.yaml into MongoDB as separate documents per token"""
    config = load_config()
    db_service = MongoDBService()
    
    try:
        # Get the timestamp for all documents
        timestamp = datetime.utcnow().isoformat()
        
        # Store each token's configuration as a separate document
        for token, allocation in config["capital_model"]["token_allocations"].items():
            token_doc = {
                "token": token,
                "allocated_capital": allocation["allocated_capital"],
                "trade_size": allocation["trade_size"],
                "current_position": 0.0,
                "available_capital": allocation["allocated_capital"],
                "last_trade_price": 0.0,
                "last_trade_type": "none",
                "last_trade_time": timestamp
            }
            
            # Update existing document or insert new one
            db_service.db.portfolio_config.update_one(
                {"token": token},
                {"$set": token_doc},
                upsert=True  # Create new document if it doesn't exist
            )
        
        trading_logger.log_message(
            "portfolio_manager",
            "Successfully updated token allocations in MongoDB",
            "INFO"
        )
        print("Token allocations updated successfully!")
        print("\nUpdated Token Allocations:")
        for token, allocation in config['capital_model']['token_allocations'].items():
            print(f"{token}:")
            print(f"  Allocated Capital: ${allocation['allocated_capital']}")
            print(f"  Trade Size: ${allocation['trade_size']}")
        
    except Exception as e:
        trading_logger.log_message(
            "portfolio_manager",
            f"Error updating token allocations: {str(e)}",
            "ERROR"
        )
        print(f"Error updating token allocations: {str(e)}")

if __name__ == "__main__":
    # Store token allocations in MongoDB
    store_config() 