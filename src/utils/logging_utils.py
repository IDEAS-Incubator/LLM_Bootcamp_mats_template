import logging
from datetime import datetime
from pathlib import Path

class TradingLogger:
    def __init__(self):
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "trading.log"),
                logging.StreamHandler()  # Also print to console
            ]
        )
        
        self.logger = logging.getLogger("trading")
    
    def log(self, agent_name: str, message: str):
        """Simple logging method for all trading activities"""
        self.logger.info(f"{agent_name}: {message}")
    
    def error(self, agent_name: str, message: str):
        """Log error messages"""
        self.logger.error(f"{agent_name}: {message}")
    
    def log_message(self, team: str, message: str, level: str = "INFO"):
        """Log a message for a specific team (compatibility with existing code)"""
        log_level = getattr(logging, level.upper())
        self.logger.log(log_level, f"{team}: {message}")
    
    def log_state_update(self, team: str, state: dict):
        """Log a state update in a more readable format"""
        # Extract relevant information
        current_agent = state.get('current_agent', 'Unknown')
        workflow = state.get('current_workflow', 'Unknown')
        task_status = state.get('task_status', {})
        
        # Format the message
        status_message = f"State Update - Team: {team}, Current Agent: {current_agent}, Workflow: {workflow}"
        
        # Add task status if available
        if task_status:
            completed_tasks = [task for task, completed in task_status.items() if completed]
            if completed_tasks:
                status_message += f"\nCompleted Tasks: {', '.join(completed_tasks)}"
        
        self.logger.info(status_message)

# Create a singleton instance
trading_logger = TradingLogger() 