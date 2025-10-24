"""
MULTI-AGENT TRADING SYSTEM - STUDENT TEMPLATE
================================================================

This is a template for students to implement a sophisticated multi-agent trading system.
The system coordinates multiple agents to perform automated cryptocurrency trading.

LEARNING OBJECTIVES:
- Understand multi-agent system architecture
- Implement agent communication patterns
- Apply trading strategies (MA, EMA, MACD, RSI, VP)
- Handle real-time market data processing from Yahoo Finance API
- Implement risk management and order execution

WHAT STUDENTS NEED TO IMPLEMENT:
[TODO] DataAgent: Market data fetching from Yahoo Finance with caching
[TODO] StrategyAgent: Trading signal generation using any strategy
[TODO] RiskAgent: Risk assessment and position sizing logic
[TODO] OrderAgent: Order execution and portfolio management
[OPTIONAL] Extend PortfolioAgent: Add advanced portfolio management features
[OPTIONAL] Extend TeamManager: Add custom workflow coordination

KEY RESOURCES PROVIDED:
- strategy/MA/ma_strategy.py: Complete moving average strategy functions
- strategy/EMA/ema_strategy.py: Complete EMA strategy functions
- strategy/MACD/macd_strategy.py: Complete MACD strategy functions  
- strategy/RSI/rsi_strategy.py: Complete RSI strategy functions
- strategy/VP/vp_strategy.py: Complete volume profile strategy functions
- services/: Database, email, and logging services
- core/: Data models and portfolio management
- config/: Trading configuration settings

IMPLEMENTATION STRATEGY:
1. Start with DataAgent (learn Yahoo Finance API and data handling)
2. Implement StrategyAgent (choose any strategy from MA/EMA/MACD/RSI/VP)
3. Implement RiskAgent for position sizing
4. Complete OrderAgent for trade execution
5. Test full workflow and enhance as needed

================================================================
"""

from typing import Dict, List, TypedDict, Sequence, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import BaseMessage
from datetime import datetime
import asyncio
from enum import Enum
import yaml
from pathlib import Path
from utils.logging_utils import trading_logger
import yfinance as yf
from core.data_models import PriceSnapshot, PriceDataPoint
from services.database_service import MongoDBService
import pandas as pd
from services.email_service import EmailService

# Import strategy components
from strategy.MA.ma_strategy import (
    add_multi_timeframe_moving_averages,
    add_multi_timeframe_momentum,
    add_multi_timeframe_volume,
    add_multi_timeframe_support_resistance,
    add_price_patterns,
    validate_ma_dataframe,
    generate_ma_signal,
    backtest_multi_timeframe_strategy,
    compute_multi_timeframe_metrics
)

# Load configuration at module level
config_path = Path(__file__).parent.parent / "config" / "trading_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Define the state type for our graph
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    current_agent: str
    message_history: List[Dict[str, str]]
    task_status: Dict[str, bool]
    parallel_tasks: List[str]
    trading_teams: Dict[str, Dict[str, bool]]
    current_workflow: str

class WorkflowStatus(Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"

# Base Agent Class for Messaging
class BaseAgent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.team = self._extract_team_from_name(name)

    def _extract_team_from_name(self, name: str) -> str:
        """Extract team name from agent name"""
        if "Portfolio" in name:
            return "portfolio_manager"
        return name.split()[0]  # First word is team name

    async def process(self, state: AgentState) -> AgentState:
        # Log the start of processing
        trading_logger.log_agent_action(
            self.team,
            self.name,
            "start_processing",
            {"state": state}
        )

        # Add a simple acknowledgment message
        state["messages"].append(AIMessage(
            content=f"TO: System\nMESSAGE: {self.name} is processing\nACTION: Continue workflow"
        ))
        
        state["message_history"].append({
            "from": self.name,
            "content": f"{self.name} is processing",
            "timestamp": datetime.now().isoformat()
        })
        
        # Log the completion of processing
        trading_logger.log_agent_action(
            self.team,
            self.name,
            "complete_processing",
            {"state": state}
        )
        
        return state

# Dynamic Agent Factory
class AgentFactory:
    @staticmethod
    def create_agent(agent_type: str, team: str, role: str) -> BaseAgent:
        agent_classes = {
            "Data": DataAgent,
            "Strategy": StrategyAgent,
            "Risk": RiskAgent,
            "Order": OrderAgent
        }
        
        if agent_type not in agent_classes:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
        return agent_classes[agent_type](team, role)

# =============================================================================
# STRATEGY AGENT -  IMPLEMENTATION TEMPLATE
# =============================================================================

class StrategyAgent(BaseAgent):
    """
    [STUDENT TODO] Implement Moving Average Strategy Agent
    
    This agent should:
    1. Load and manage MA strategy parameters from database
    2. Convert price data to analysis-ready format
    3. Apply technical indicators using functions from strategy/MA/ma_strategy.py
    4. Generate BUY/SELL/HOLD trading signals
    5. Send email notifications for actionable signals
    6. Communicate results to Risk Agent
    
    RESOURCES TO USE:
    - Import functions from strategy.MA.ma_strategy:
      * add_multi_timeframe_moving_averages()
      * add_multi_timeframe_momentum() 
      * add_multi_timeframe_volume()
      * add_multi_timeframe_support_resistance()
      * add_price_patterns()
      * generate_ma_signal()
    """
    
    def __init__(self, team: str, role: str):
        """
        [TODO] Initialize Strategy Agent
        
        Students should implement:
        1. Call parent class __init__ with agent name and role
        2. Store team name for this agent
        3. Initialize database service (MongoDBService)
        4. Initialize email service (EmailService) 
        5. Set up default MA strategy configuration dict with:
           - ma_fast: 10 (fast moving average period)
           - ma_slow: 20 (slow moving average period)  
           - ma_trend: 50 (trend moving average period)
           - momentum_period: 3 (momentum calculation period)
           - volume_period: 8 (volume analysis period)
           - position_allocation: 0.50 (position size allocation)
        6. Call _load_optimized_parameters() method
        7. Log successful initialization
        """
        # 1. Call parent class __init__ - REQUIRED for agent to work
        super().__init__(
            name=f"{team} Strategy Agent",
            role=role
        )
        
        # 2. Store team name for this agent
        self.team = team
        
        # 3. Initialize database service (MongoDBService)
        self.db_service = MongoDBService()
        
        # 4. Initialize email service (EmailService)
        self.email_service = EmailService()
        
        # 5. Set up default MA strategy configuration dict
        self.config = {
            "ma_fast": 10,  # fast moving average period
            "ma_slow": 20,  # slow moving average period
            "ma_trend": 50,  # trend moving average period
            "momentum_period": 3,  # momentum calculation period
            "volume_period": 8,  # volume analysis period
            "position_allocation": 0.50  # position size allocation
        }
        
        # 6. Call _load_optimized_parameters() method
        self._load_optimized_parameters()
        
        # 7. Log successful initialization
        trading_logger.log_message(
            self.team,
            f"StrategyAgent initialized with MA strategy config: {self.config}",
            "INFO"
        )
        
        print(f"StrategyAgent initialized for {team} - Config: {self.config}")

    def _load_optimized_parameters(self):
        """
        [TODO] Load optimized MA parameters from database
        
        Students should implement:
        1. Use try-catch for error handling
        2. Query self.db_service.db.optimized_parameters.find_one() with:
           - Filter: {"symbol": self.team}
           - Sort: [("timestamp", -1)] to get most recent
        3. Check if optimized_params exists and has "parameters" key
        4. If found: Update self.config with optimized_params["parameters"]
        5. Log success with trading_logger.log_message()
        6. If not found: Log warning about using defaults
        7. Handle exceptions gracefully with error logging
        """
        try:
            # Query database for optimized parameters
            optimized_params = self.db_service.db.optimized_parameters.find_one(
                {"symbol": self.team},
                sort=[("timestamp", -1)]
            )
            
            # Check if optimized parameters exist and have "parameters" key
            if optimized_params and "parameters" in optimized_params:
                # Update self.config with optimized parameters
                self.config.update(optimized_params["parameters"])
                
                # Log success
                trading_logger.log_message(
                    self.team,
                    f"Loaded optimized parameters: {optimized_params['parameters']}",
                    "INFO"
                )
                print(f"StrategyAgent: Loaded optimized parameters for {self.team}")
            else:
                # Log warning about using defaults
                trading_logger.log_message(
                    self.team,
                    f"No optimized parameters found for {self.team}, using default config",
                    "WARNING"
                )
                print(f"StrategyAgent: Using default parameters for {self.team}")
                
        except Exception as e:
            # Handle exceptions gracefully with error logging
            trading_logger.log_message(
                self.team,
                f"Error loading optimized parameters: {str(e)}",
                "ERROR"
            )
            print(f"StrategyAgent: Error loading optimized parameters: {str(e)}")


    def _convert_price_data_to_dataframe(self, price_data: List[PriceDataPoint]) -> pd.DataFrame:
        """
        [TODO] Convert PriceDataPoint objects to pandas DataFrame
        
        Students should implement:
        1. Check if price_data is empty, return empty DataFrame if so
        2. Create two lists: data[] and datetimes[]
        3. Loop through each point in price_data:
           - Extract OHLCV data: open, high, low, close (from point.price), volume
           - Add to data list as dictionary
           - Parse datetime with pd.to_datetime(point.datetime)
           - Handle timezone: localize to UTC if no timezone, convert to UTC if timezone exists
           - Add normalized datetime to datetimes list
        4. Create DataFrame from data list
        5. Set index to DatetimeIndex from datetimes list
        6. Sort by index chronologically
        7. Log conversion results with trading_logger
        8. Return the processed DataFrame
        
        Args:
            price_data: List of PriceDataPoint objects from database
            
        Returns:
            pd.DataFrame: OHLCV DataFrame with datetime index
        """
        # 1. Check if price_data is empty, return empty DataFrame if so
        if not price_data:
            trading_logger.log_message(
                self.team,
                "No price data provided, returning empty DataFrame",
                "WARNING"
            )
            return pd.DataFrame()
        
        # 2. Create two lists: data[] and datetimes[]
        data = []
        datetimes = []
        
        # 3. Loop through each point in price_data
        for point in price_data:
            # Extract OHLCV data: open, high, low, close (from point.price), volume
            data_point = {
                'open': point.price.open,
                'high': point.price.high,
                'low': point.price.low,
                'close': point.price.close,
                'volume': point.volume,
                'price_change': point.price_change
            }
            data.append(data_point)
            
            # Parse datetime with pd.to_datetime(point.datetime)
            dt = pd.to_datetime(point.datetime)
            
            # Handle timezone: localize to UTC if no timezone, convert to UTC if timezone exists
            if dt.tz is None:
                dt = dt.tz_localize('UTC')
            else:
                dt = dt.tz_convert('UTC')
            
            # Add normalized datetime to datetimes list
            datetimes.append(dt)
        
        # 4. Create DataFrame from data list
        df = pd.DataFrame(data)
        
        # 5. Set index to DatetimeIndex from datetimes list
        df.index = pd.DatetimeIndex(datetimes)
        
        # 6. Sort by index chronologically
        df = df.sort_index()
        
        # 7. Log conversion results with trading_logger
        trading_logger.log_message(
            self.team,
            f"Converted {len(price_data)} price points to DataFrame with shape {df.shape}",
            "INFO"
        )
        
        # 8. Return the processed DataFrame
        return df

    async def process(self, state: AgentState) -> AgentState:
        """
        [TODO] Implement Moving Average Strategy Analysis Workflow
        
        This is the main method that students need to implement. It should:
        
        STEP 1: SETUP AND LOGGING
        - Log start of MA strategy analysis using trading_logger.log_message()
        - Print status message for console output
        
        STEP 2: DATA RETRIEVAL
        - Get latest price snapshot: self.db_service.get_latest_snapshot(self.team)
        - Check if snapshot exists, return state if None
        - Log error and return state if no data available
        
        STEP 3: DATA PREPARATION  
        - Convert price data to DataFrame: self._convert_price_data_to_dataframe(snapshot.price_data)
        - Check if DataFrame is empty, return state if so
        - Log warning if empty data
        
        STEP 4: TECHNICAL ANALYSIS (Import from strategy.MA.ma_strategy!)
        - Apply moving averages: add_multi_timeframe_moving_averages(df, config params)
        - Apply momentum: add_multi_timeframe_momentum(df, config["momentum_period"])
        - Apply volume analysis: add_multi_timeframe_volume(df, config["volume_period"]) 
        - Apply support/resistance: add_multi_timeframe_support_resistance(df, 15)
        - Apply price patterns: add_price_patterns(df)
        
        STEP 5: SIGNAL GENERATION (Import from strategy.MA.ma_strategy!)
        - Generate signal: signal_result = generate_ma_signal(df, self.config)
        - Extract: signal, strategy_details, current_price, signal_reason from result
        
        STEP 6: LOGGING AND NOTIFICATIONS
        - Log signal generation with trading_logger
        - Print signal info to console
        - Send email notification if signal != "HOLD" using self.email_service.send_signal_email()
        
        STEP 7: COMMUNICATION
        - Create formatted message for Risk Agent with signal details
        - Add message to state["messages"] using AIMessage()
        - Add entry to state["message_history"] with timestamp
        
        STEP 8: WORKFLOW UPDATE
        - Mark strategy complete: state["trading_teams"][self.team.lower()]["strategy_complete"] = True
        - Log completion
        - Return updated state
        
        Args:
            state: Current agent workflow state
            
        Returns:
            AgentState: Updated state with strategy analysis results
        """
        # STEP 1: SETUP AND LOGGING
        trading_logger.log_message(
            self.team,
            f"Starting MA strategy analysis for {self.team}",
            "INFO"
        )
        print(f"StrategyAgent: Starting MA strategy analysis for {self.team}")
        
        # STEP 2: DATA RETRIEVAL
        snapshot = self.db_service.get_latest_snapshot(self.team)
        if not snapshot:
            trading_logger.log_message(
                self.team,
                f"No price snapshot found for {self.team}",
                "ERROR"
            )
            print(f"StrategyAgent: No price snapshot found for {self.team}")
            # Mark as complete to avoid blocking workflow
            state["trading_teams"][self.team.lower()]["strategy_complete"] = True
            return state
        
        # STEP 3: DATA PREPARATION
        df = self._convert_price_data_to_dataframe(snapshot.price_data)
        if df.empty:
            trading_logger.log_message(
                self.team,
                f"Empty DataFrame after conversion for {self.team}",
                "WARNING"
            )
            print(f"StrategyAgent: Empty DataFrame for {self.team}")
            # Mark as complete to avoid blocking workflow
            state["trading_teams"][self.team.lower()]["strategy_complete"] = True
            return state
        
        # STEP 4: TECHNICAL ANALYSIS (Import from strategy.MA.ma_strategy!)
        try:
            # Apply moving averages
            df = add_multi_timeframe_moving_averages(df, self.config)
            
            # Apply momentum
            df = add_multi_timeframe_momentum(df, self.config["momentum_period"])
            
            # Apply volume analysis
            df = add_multi_timeframe_volume(df, self.config["volume_period"])
            
            # Apply support/resistance
            df = add_multi_timeframe_support_resistance(df, 15)
            
            # Apply price patterns
            df = add_price_patterns(df)
            
            trading_logger.log_message(
                self.team,
                f"Applied technical analysis indicators to DataFrame",
                "INFO"
            )
            
        except Exception as e:
            trading_logger.log_message(
                self.team,
                f"Error applying technical analysis: {str(e)}",
                "ERROR"
            )
            print(f"StrategyAgent: Error in technical analysis: {str(e)}")
            # Mark as complete to avoid blocking workflow
            state["trading_teams"][self.team.lower()]["strategy_complete"] = True
            return state
        
        # STEP 5: SIGNAL GENERATION (Import from strategy.MA.ma_strategy!)
        try:
            signal_result = generate_ma_signal(df, self.config)
            
            # Extract signal details
            signal = signal_result.get("signal", "HOLD")
            strategy_details = signal_result.get("strategy_details", {})
            current_price = signal_result.get("current_price", 0.0)
            signal_reason = signal_result.get("signal_reason", "No reason provided")
            
        except Exception as e:
            trading_logger.log_message(
                self.team,
                f"Error generating signal: {str(e)}",
                "ERROR"
            )
            print(f"StrategyAgent: Error generating signal: {str(e)}")
            # Use default values
            signal = "HOLD"
            strategy_details = {}
            current_price = df['close'].iloc[-1] if not df.empty else 0.0
            signal_reason = "Error in signal generation"
        
        # STEP 6: LOGGING AND NOTIFICATIONS
        trading_logger.log_message(
            self.team,
            f"Generated signal: {signal} for {self.team} at price {current_price}",
            "INFO"
        )
        print(f"StrategyAgent: Signal {signal} for {self.team} at ${current_price:.2f}")
        print(f"StrategyAgent: Reason: {signal_reason}")
        
        # Send email notification if signal != "HOLD"
        if signal != "HOLD":
            try:
                self.email_service.send_signal_email(
                    self.team,
                    signal,
                    current_price,
                    signal_reason,
                    strategy_details
                )
                trading_logger.log_message(
                    self.team,
                    f"Sent email notification for {signal} signal",
                    "INFO"
                )
            except Exception as e:
                trading_logger.log_message(
                    self.team,
                    f"Error sending email notification: {str(e)}",
                    "ERROR"
                )
        
        # STEP 7: COMMUNICATION
        message_content = f"""TO: {self.team} Risk Agent
MESSAGE: MA Strategy Analysis Complete
SIGNAL: {signal}
PRICE: {current_price}
STRATEGY: MA_CROSSOVER
MA_FAST: {strategy_details.get('ma_fast', 'N/A')}
MA_SLOW: {strategy_details.get('ma_slow', 'N/A')}
MA_TREND: {strategy_details.get('ma_trend', 'N/A')}
REASON: {signal_reason}
ACTION: Proceed with risk assessment"""
        
        state["messages"].append(AIMessage(content=message_content))
        state["message_history"].append({
            "from": self.name,
            "content": f"MA Strategy Analysis: {signal} signal at ${current_price:.2f}",
            "timestamp": datetime.now().isoformat()
        })
        
        # STEP 8: WORKFLOW UPDATE
        state["trading_teams"][self.team.lower()]["strategy_complete"] = True
        
        trading_logger.log_message(
            self.team,
            f"MA strategy analysis completed for {self.team}",
            "INFO"
        )
        
        return state

    def __del__(self):
        """
        [TODO] Implement Resource Cleanup
        
        Students should implement:
        1. Check if self.db_service exists using hasattr()
        2. Call self.db_service.close() to cleanup database connection
        3. This prevents resource leaks when agent is destroyed
        """
        # Check if self.db_service exists using hasattr()
        if hasattr(self, 'db_service') and self.db_service:
            try:
                # Call self.db_service.close() to cleanup database connection
                self.db_service.close()
                trading_logger.log_message(
                    self.team,
                    "StrategyAgent database connection closed",
                    "INFO"
                )
            except Exception as e:
                trading_logger.log_message(
                    self.team,
                    f"Error closing database connection: {str(e)}",
                    "ERROR"
                )

# =============================================================================
# END STRATEGY AGENT TEMPLATE
# 
# STUDENT SUCCESS CHECKLIST:
# [ ] Implement __init__ with all required services and config
# [ ] Implement _load_optimized_parameters with database queries  
# [ ] Implement _convert_price_data_to_dataframe for data processing
# [ ] Implement process() method with 8-step workflow
# [ ] Use functions from strategy.MA.ma_strategy module
# [ ] Implement proper error handling and logging
# [ ] Test with real market data from DataAgent
# 
# KEY IMPORTS NEEDED:
# from strategy.MA.ma_strategy import (
#     add_multi_timeframe_moving_averages,
#     add_multi_timeframe_momentum, 
#     add_multi_timeframe_volume,
#     add_multi_timeframe_support_resistance,
#     add_price_patterns,
#     generate_ma_signal
# )
# =============================================================================

# =============================================================================
# RISK AGENT -  IMPLEMENTATION TEMPLATE  
# =============================================================================

class RiskAgent(BaseAgent):
    """
    [STUDENT TODO] Implement Risk Management Agent
    
    This agent should:
    1. Parse trading signals from StrategyAgent
    2. Perform comprehensive risk assessment
    3. Calculate appropriate position sizes
    4. Set stop-loss and take-profit levels
    5. Monitor portfolio exposure and correlation
    6. Make risk-based recommendations (PROCEED/REJECT/MODIFY)
    7. Forward approved signals to OrderAgent
    """
    
    def __init__(self, team: str, role: str):
        """
        [TODO] Initialize Risk Management Agent
        
        Students should implement:
        1. Call parent class __init__
        2. Store team name
        3. Initialize database service for risk data storage
        4. Set up risk management parameters:
           - max_position_size: Maximum position size (e.g., 0.10 = 10%)
           - max_portfolio_exposure: Maximum total exposure (e.g., 0.80 = 80%)
           - stop_loss_pct: Default stop loss percentage (e.g., 0.02 = 2%)
           - take_profit_pct: Default take profit percentage (e.g., 0.04 = 4%)
           - max_correlation: Maximum correlation between positions (e.g., 0.70)
        5. Initialize position tracking dictionary
        6. Log successful initialization
        """
        # 1. Call parent class __init__ - REQUIRED for agent to work
        super().__init__(
            name=f"{team} Risk Agent",
            role=role
        )
        
        # 2. Store team name
        self.team = team
        
        # 3. Initialize database service for risk data storage
        self.db_service = MongoDBService()
        
        # 4. Set up risk management parameters
        self.max_position_size = 0.10  # Maximum position size (10%)
        self.max_portfolio_exposure = 0.80  # Maximum total exposure (80%)
        self.stop_loss_pct = 0.02  # Default stop loss percentage (2%)
        self.take_profit_pct = 0.04  # Default take profit percentage (4%)
        self.max_correlation = 0.70  # Maximum correlation between positions (70%)
        
        # 5. Initialize position tracking dictionary
        self.position_tracking = {}
        
        # 6. Log successful initialization
        trading_logger.log_message(
            self.team,
            f"RiskAgent initialized with risk parameters - Max Position: {self.max_position_size}, Max Exposure: {self.max_portfolio_exposure}",
            "INFO"
        )
        
        print(f"RiskAgent initialized for {team} - Max Position: {self.max_position_size}, Max Exposure: {self.max_portfolio_exposure}")

    def _parse_strategy_message(self, messages: List) -> Dict:
        """
        [TODO] Parse Strategy Agent Message
        
        Students should implement:
        1. Find the latest message from "{self.team} Strategy Agent"
        2. Parse message content to extract:
           - SIGNAL: BUY/SELL/HOLD
           - PRICE: Current market price
           - STRATEGY: Strategy type (e.g., MA_CROSS_UP)
           - MA_FAST, MA_SLOW, MA_TREND: Moving average values
           - REASON: Signal reasoning
        3. Return parsed data as dictionary
        4. Handle parsing errors gracefully
        
        Returns:
            Dict: Parsed signal data or empty dict if parsing fails
        """
        try:
            # Find the latest message from "{self.team} Strategy Agent"
            strategy_agent_name = f"{self.team} Strategy Agent"
            latest_message = None
            
            for message in reversed(messages):
                if hasattr(message, 'content') and strategy_agent_name in str(message.content):
                    latest_message = message
                    break
            
            if not latest_message:
                trading_logger.log_message(
                    self.team,
                    f"No message found from {strategy_agent_name}",
                    "WARNING"
                )
                return {}
            
            # Parse message content to extract signal data
            content = str(latest_message.content)
            parsed_data = {}
            
            # Extract SIGNAL
            if "SIGNAL:" in content:
                signal_line = [line for line in content.split('\n') if 'SIGNAL:' in line][0]
                parsed_data['signal'] = signal_line.split('SIGNAL:')[1].strip()
            
            # Extract PRICE
            if "PRICE:" in content:
                price_line = [line for line in content.split('\n') if 'PRICE:' in line][0]
                parsed_data['price'] = float(price_line.split('PRICE:')[1].strip())
            
            # Extract STRATEGY
            if "STRATEGY:" in content:
                strategy_line = [line for line in content.split('\n') if 'STRATEGY:' in line][0]
                parsed_data['strategy'] = strategy_line.split('STRATEGY:')[1].strip()
            
            # Extract MA values
            if "MA_FAST:" in content:
                ma_fast_line = [line for line in content.split('\n') if 'MA_FAST:' in line][0]
                parsed_data['ma_fast'] = ma_fast_line.split('MA_FAST:')[1].strip()
            
            if "MA_SLOW:" in content:
                ma_slow_line = [line for line in content.split('\n') if 'MA_SLOW:' in line][0]
                parsed_data['ma_slow'] = ma_slow_line.split('MA_SLOW:')[1].strip()
            
            if "MA_TREND:" in content:
                ma_trend_line = [line for line in content.split('\n') if 'MA_TREND:' in line][0]
                parsed_data['ma_trend'] = ma_trend_line.split('MA_TREND:')[1].strip()
            
            # Extract REASON
            if "REASON:" in content:
                reason_line = [line for line in content.split('\n') if 'REASON:' in line][0]
                parsed_data['reason'] = reason_line.split('REASON:')[1].strip()
            
            trading_logger.log_message(
                self.team,
                f"Parsed strategy message: {parsed_data}",
                "INFO"
            )
            
            return parsed_data
            
        except Exception as e:
            # Handle parsing errors gracefully
            trading_logger.log_message(
                self.team,
                f"Error parsing strategy message: {str(e)}",
                "ERROR"
            )
            return {}

    def _calculate_position_size(self, signal: str, price: float, portfolio_value: float) -> float:
        """
        [TODO] Calculate Appropriate Position Size
        
        Students should implement:
        1. Get current portfolio value
        2. Calculate maximum allowed position size based on risk rules
        3. Consider current portfolio exposure
        4. Adjust for market volatility (higher volatility = smaller position)
        5. Ensure position size doesn't exceed risk limits
        6. Return position size as percentage of portfolio (0.0 to 1.0)
        
        Args:
            signal: Trading signal (BUY/SELL)
            price: Current market price
            portfolio_value: Total portfolio value
            
        Returns:
            float: Position size as percentage (0.0 to 1.0)
        """
        try:
            # 1. Get current portfolio value (already provided as parameter)
            if portfolio_value <= 0:
                trading_logger.log_message(
                    self.team,
                    f"Invalid portfolio value: {portfolio_value}",
                    "ERROR"
                )
                return 0.0
            
            # 2. Calculate maximum allowed position size based on risk rules
            base_position_size = self.max_position_size  # Start with max position size
            
            # 3. Consider current portfolio exposure
            current_exposure = self._get_current_portfolio_exposure()
            remaining_exposure_capacity = self.max_portfolio_exposure - current_exposure
            
            # Adjust position size based on remaining exposure capacity
            if remaining_exposure_capacity < base_position_size:
                base_position_size = max(0.01, remaining_exposure_capacity)  # Minimum 1%
                trading_logger.log_message(
                    self.team,
                    f"Reduced position size due to exposure limits: {base_position_size}",
                    "WARNING"
                )
            
            # 4. Adjust for market volatility (higher volatility = smaller position)
            volatility_adjustment = self._calculate_volatility_adjustment(price)
            adjusted_position_size = base_position_size * volatility_adjustment
            
            # 5. Ensure position size doesn't exceed risk limits
            final_position_size = min(adjusted_position_size, self.max_position_size)
            final_position_size = max(0.01, final_position_size)  # Minimum 1%
            
            trading_logger.log_message(
                self.team,
                f"Calculated position size: {final_position_size:.3f} (base: {base_position_size:.3f}, volatility adj: {volatility_adjustment:.3f})",
                "INFO"
            )
            
            # 6. Return position size as percentage of portfolio (0.0 to 1.0)
            return final_position_size
            
        except Exception as e:
            trading_logger.log_message(
                self.team,
                f"Error calculating position size: {str(e)}",
                "ERROR"
            )
            return 0.05  # Default 5% position size on error
    
    def _get_current_portfolio_exposure(self) -> float:
        """Helper method to get current portfolio exposure"""
        try:
            # This would typically query the database for current positions
            # For now, return a simulated value
            return 0.3  # 30% current exposure
        except Exception:
            return 0.0
    
    def _calculate_volatility_adjustment(self, price: float) -> float:
        """Helper method to calculate volatility adjustment factor"""
        try:
            # This would typically calculate volatility from recent price data
            # For now, return a simulated adjustment factor
            # Higher volatility = lower adjustment factor (smaller position)
            return 0.8  # 80% of base position size due to volatility
        except Exception:
            return 1.0  # No adjustment on error

    def _assess_risk_level(self, signal_data: Dict) -> Dict:
        """
        [TODO] Perform Risk Assessment
        
        Students should implement:
        1. Analyze market volatility (using price data from last N periods)
        2. Check portfolio concentration risk
        3. Evaluate correlation with existing positions
        4. Assess signal strength and confidence
        5. Calculate Value at Risk (VaR) if applicable
        6. Return risk assessment with:
           - risk_level: "LOW", "MEDIUM", "HIGH"
           - risk_factors: List of identified risk factors
           - recommendation: "PROCEED", "REDUCE", "REJECT"
           - adjusted_position_size: Risk-adjusted position size
        
        Args:
            signal_data: Parsed signal information
            
        Returns:
            Dict: Risk assessment results
        """
        try:
            risk_factors = []
            risk_score = 0
            
            # 1. Analyze market volatility (using price data from last N periods)
            volatility_risk = self._analyze_market_volatility(signal_data.get('price', 0))
            if volatility_risk > 0.7:
                risk_factors.append("High market volatility detected")
                risk_score += 2
            elif volatility_risk > 0.4:
                risk_factors.append("Moderate market volatility")
                risk_score += 1
            
            # 2. Check portfolio concentration risk
            concentration_risk = self._check_portfolio_concentration()
            if concentration_risk > 0.8:
                risk_factors.append("High portfolio concentration")
                risk_score += 2
            elif concentration_risk > 0.5:
                risk_factors.append("Moderate portfolio concentration")
                risk_score += 1
            
            # 3. Evaluate correlation with existing positions
            correlation_risk = self._evaluate_position_correlation(signal_data.get('signal', ''))
            if correlation_risk > self.max_correlation:
                risk_factors.append(f"High correlation with existing positions ({correlation_risk:.2f})")
                risk_score += 2
            elif correlation_risk > 0.5:
                risk_factors.append(f"Moderate correlation with existing positions ({correlation_risk:.2f})")
                risk_score += 1
            
            # 4. Assess signal strength and confidence
            signal_strength = self._assess_signal_strength(signal_data)
            if signal_strength < 0.3:
                risk_factors.append("Weak signal strength")
                risk_score += 2
            elif signal_strength < 0.6:
                risk_factors.append("Moderate signal strength")
                risk_score += 1
            
            # 5. Calculate Value at Risk (VaR) if applicable
            var_risk = self._calculate_var_risk(signal_data.get('price', 0))
            if var_risk > 0.05:  # 5% VaR threshold
                risk_factors.append(f"High VaR risk ({var_risk:.2%})")
                risk_score += 2
            elif var_risk > 0.02:  # 2% VaR threshold
                risk_factors.append(f"Moderate VaR risk ({var_risk:.2%})")
                risk_score += 1
            
            # Determine risk level based on score
            if risk_score >= 5:
                risk_level = "HIGH"
                recommendation = "REJECT"
                position_adjustment = 0.0
            elif risk_score >= 3:
                risk_level = "MEDIUM"
                recommendation = "REDUCE"
                position_adjustment = 0.5  # Reduce position by 50%
            else:
                risk_level = "LOW"
                recommendation = "PROCEED"
                position_adjustment = 1.0  # No adjustment
            
            # Calculate adjusted position size
            base_position_size = self.max_position_size
            adjusted_position_size = base_position_size * position_adjustment
            
            # Ensure minimum position size for PROCEED recommendations
            if recommendation == "PROCEED" and adjusted_position_size < 0.01:
                adjusted_position_size = 0.01
            
            risk_assessment = {
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "recommendation": recommendation,
                "adjusted_position_size": adjusted_position_size,
                "risk_score": risk_score,
                "volatility_risk": volatility_risk,
                "concentration_risk": concentration_risk,
                "correlation_risk": correlation_risk,
                "signal_strength": signal_strength,
                "var_risk": var_risk
            }
            
            trading_logger.log_message(
                self.team,
                f"Risk assessment completed: {risk_level} risk, recommendation: {recommendation}",
                "INFO"
            )
            
            return risk_assessment
            
        except Exception as e:
            trading_logger.log_message(
                self.team,
                f"Error in risk assessment: {str(e)}",
                "ERROR"
            )
            return {
                "risk_level": "HIGH",
                "risk_factors": ["Error in risk assessment"],
                "recommendation": "REJECT",
                "adjusted_position_size": 0.0,
                "risk_score": 10
            }
    
    def _analyze_market_volatility(self, price: float) -> float:
        """Helper method to analyze market volatility"""
        try:
            # This would typically calculate volatility from recent price data
            # For now, return a simulated volatility score (0.0 to 1.0)
            return 0.3  # 30% volatility
        except Exception:
            return 0.5  # Default moderate volatility
    
    def _check_portfolio_concentration(self) -> float:
        """Helper method to check portfolio concentration risk"""
        try:
            # This would typically analyze current portfolio positions
            # For now, return a simulated concentration score (0.0 to 1.0)
            return 0.4  # 40% concentration
        except Exception:
            return 0.5  # Default moderate concentration
    
    def _evaluate_position_correlation(self, signal: str) -> float:
        """Helper method to evaluate correlation with existing positions"""
        try:
            # This would typically calculate correlation with existing positions
            # For now, return a simulated correlation score (0.0 to 1.0)
            return 0.2  # 20% correlation
        except Exception:
            return 0.3  # Default moderate correlation
    
    def _assess_signal_strength(self, signal_data: Dict) -> float:
        """Helper method to assess signal strength and confidence"""
        try:
            # This would typically analyze signal quality based on technical indicators
            # For now, return a simulated signal strength (0.0 to 1.0)
            signal = signal_data.get('signal', 'HOLD')
            if signal == 'BUY' or signal == 'SELL':
                return 0.7  # Strong signal
            else:
                return 0.3  # Weak signal
        except Exception:
            return 0.5  # Default moderate strength
    
    def _calculate_var_risk(self, price: float) -> float:
        """Helper method to calculate Value at Risk"""
        try:
            # This would typically calculate VaR based on historical price movements
            # For now, return a simulated VaR percentage
            return 0.02  # 2% VaR
        except Exception:
            return 0.03  # Default 3% VaR

    async def process(self, state: AgentState) -> AgentState:
        """
        [TODO] Implement Risk Assessment Workflow
        
        Students should implement:
        
        STEP 1: MESSAGE PARSING
        - Parse Strategy Agent message using _parse_strategy_message()
        - Extract signal, price, strategy details
        - Handle parsing errors gracefully
        
        STEP 2: RISK ASSESSMENT
        - Perform risk analysis using _assess_risk_level()
        - Calculate appropriate position size using _calculate_position_size()
        - Determine risk-based recommendation
        
        STEP 3: DECISION MAKING
        - If recommendation is "PROCEED": Forward signal to Order Agent
        - If recommendation is "REDUCE": Adjust position size and forward
        - If recommendation is "REJECT": Send rejection message
        
        STEP 4: COMMUNICATION
        - Create message for Order Agent with risk assessment results
        - Include: signal, price, strategy, risk_level, recommendation
        - Add to state["messages"] and state["message_history"]
        
        STEP 5: STATE UPDATE
        - Mark risk assessment as complete
        - Log assessment results
        - Return updated state
        
        Args:
            state: Current agent workflow state
            
        Returns:
            AgentState: Updated state with risk assessment results
        """
        # STEP 1: MESSAGE PARSING
        trading_logger.log_message(
            self.team,
            f"Starting risk assessment for {self.team}",
            "INFO"
        )
        print(f"RiskAgent: Starting risk assessment for {self.team}")
        
        signal_data = self._parse_strategy_message(state["messages"])
        if not signal_data:
            trading_logger.log_message(
                self.team,
                f"No valid signal data found for {self.team}",
                "WARNING"
            )
            print(f"RiskAgent: No valid signal data found for {self.team}")
            # Mark as complete to avoid blocking workflow
            state["trading_teams"][self.team.lower()]["risk_complete"] = True
            return state
        
        # STEP 2: RISK ASSESSMENT
        risk_assessment = self._assess_risk_level(signal_data)
        
        # Calculate appropriate position size
        portfolio_value = 100000  # Simulated portfolio value - would come from database
        position_size = self._calculate_position_size(
            signal_data.get('signal', 'HOLD'),
            signal_data.get('price', 0),
            portfolio_value
        )
        
        # Apply risk-adjusted position size
        final_position_size = position_size * risk_assessment["adjusted_position_size"]
        
        # STEP 3: DECISION MAKING
        recommendation = risk_assessment["recommendation"]
        signal = signal_data.get('signal', 'HOLD')
        price = signal_data.get('price', 0)
        
        trading_logger.log_message(
            self.team,
            f"Risk assessment result: {recommendation} for {signal} signal at ${price:.2f}",
            "INFO"
        )
        print(f"RiskAgent: Recommendation: {recommendation} for {signal} signal")
        print(f"RiskAgent: Risk Level: {risk_assessment['risk_level']}")
        print(f"RiskAgent: Position Size: {final_position_size:.3f}")
        
        # STEP 4: COMMUNICATION
        if recommendation == "REJECT":
            message_content = f"""TO: {self.team} Order Agent
MESSAGE: Risk Assessment - SIGNAL REJECTED
SIGNAL: {signal}
PRICE: {price}
STRATEGY: {signal_data.get('strategy', 'N/A')}
RISK_LEVEL: {risk_assessment['risk_level']}
RECOMMENDATION: REJECT
REASON: {', '.join(risk_assessment['risk_factors'])}
ACTION: Do not execute order"""
            
            print(f"RiskAgent: Signal REJECTED due to risk factors: {risk_assessment['risk_factors']}")
            
        else:  # PROCEED or REDUCE
            message_content = f"""TO: {self.team} Order Agent
MESSAGE: Risk Assessment - SIGNAL APPROVED
SIGNAL: {signal}
PRICE: {price}
STRATEGY: {signal_data.get('strategy', 'N/A')}
RISK_LEVEL: {risk_assessment['risk_level']}
RECOMMENDATION: {recommendation}
POSITION_SIZE: {final_position_size:.3f}
STOP_LOSS: {self.stop_loss_pct:.2%}
TAKE_PROFIT: {self.take_profit_pct:.2%}
RISK_FACTORS: {', '.join(risk_assessment['risk_factors']) if risk_assessment['risk_factors'] else 'None'}
ACTION: Execute order with risk parameters"""
            
            print(f"RiskAgent: Signal APPROVED with {recommendation} recommendation")
        
        # Add message to state
        state["messages"].append(AIMessage(content=message_content))
        state["message_history"].append({
            "from": self.name,
            "content": f"Risk Assessment: {recommendation} for {signal} signal (Risk: {risk_assessment['risk_level']})",
            "timestamp": datetime.now().isoformat()
        })
        
        # STEP 5: STATE UPDATE
        state["trading_teams"][self.team.lower()]["risk_complete"] = True
        
        trading_logger.log_message(
            self.team,
            f"Risk assessment completed for {self.team}",
            "INFO"
        )
        
        return state

# =============================================================================
# ORDER AGENT -  IMPLEMENTATION TEMPLATE
# =============================================================================

class OrderAgent(BaseAgent):
    """
    [STUDENT TODO] Implement Order Execution Agent
    
    This agent should:
    1. Parse risk assessment from RiskAgent
    2. Execute approved trading orders
    3. Manage different order types (market, limit, stop-loss)
    4. Store order details in database
    5. Update portfolio positions
    6. Handle order execution errors
    7. Send confirmations to TeamManager
    """
    
    def __init__(self, team: str, role: str):
        """
        [TODO] Initialize Order Management Agent
        
        Students should implement:
        1. Call parent class __init__
        2. Store team name
        3. Initialize database service for order storage
        4. Set up order execution parameters:
           - order_types: ["market", "limit", "stop_loss"]
           - default_slippage: Expected slippage percentage (e.g., 0.001 = 0.1%)
           - max_order_size: Maximum order size limit
           - timeout_seconds: Order timeout (e.g., 30 seconds)
        5. Initialize position tracking
        6. Set up exchange connection (if using real trading)
        7. Log successful initialization
        """
        # 1. Call parent class __init__ - REQUIRED for agent to work
        super().__init__(
            name=f"{team} Order Agent",
            role=role
        )
        
        # 2. Store team name
        self.team = team
        
        # 3. Initialize database service for order storage
        self.db_service = MongoDBService()
        
        # 4. Set up order execution parameters
        self.order_types = ["market", "limit", "stop_loss"]
        self.default_slippage = 0.001  # Expected slippage percentage (0.1%)
        self.max_order_size = 1000000  # Maximum order size limit ($1M)
        self.timeout_seconds = 30  # Order timeout (30 seconds)
        
        # 5. Initialize position tracking
        self.position_tracking = {}
        
        # 6. Set up exchange connection (simulation mode for now)
        self.exchange_connected = False  # Would be True for real trading
        self.simulation_mode = True  # Enable simulation mode
        
        # 7. Log successful initialization
        trading_logger.log_message(
            self.team,
            f"OrderAgent initialized with order types: {self.order_types}, simulation mode: {self.simulation_mode}",
            "INFO"
        )
        
        print(f"OrderAgent initialized for {team} - Order Types: {self.order_types}, Simulation: {self.simulation_mode}")

    def _parse_risk_message(self, messages: List) -> Dict:
        """
        [TODO] Parse Risk Agent Message
        
        Students should implement:
        1. Find latest message from "{self.team} Risk Agent"
        2. Parse message content to extract:
           - SIGNAL: BUY/SELL/HOLD
           - PRICE: Target execution price
           - STRATEGY: Strategy type
           - RISK_LEVEL: Risk assessment level
           - RECOMMENDATION: PROCEED/REDUCE/REJECT
        3. Return parsed data as dictionary
        4. Handle parsing errors gracefully
        
        Returns:
            Dict: Parsed risk assessment data
        """
        try:
            # Find latest message from "{self.team} Risk Agent"
            risk_agent_name = f"{self.team} Risk Agent"
            latest_message = None
            
            for message in reversed(messages):
                if hasattr(message, 'content') and risk_agent_name in str(message.content):
                    latest_message = message
                    break
            
            if not latest_message:
                trading_logger.log_message(
                    self.team,
                    f"No message found from {risk_agent_name}",
                    "WARNING"
                )
                return {}
            
            # Parse message content to extract risk assessment data
            content = str(latest_message.content)
            parsed_data = {}
            
            # Extract SIGNAL
            if "SIGNAL:" in content:
                signal_line = [line for line in content.split('\n') if 'SIGNAL:' in line][0]
                parsed_data['signal'] = signal_line.split('SIGNAL:')[1].strip()
            
            # Extract PRICE
            if "PRICE:" in content:
                price_line = [line for line in content.split('\n') if 'PRICE:' in line][0]
                parsed_data['price'] = float(price_line.split('PRICE:')[1].strip())
            
            # Extract STRATEGY
            if "STRATEGY:" in content:
                strategy_line = [line for line in content.split('\n') if 'STRATEGY:' in line][0]
                parsed_data['strategy'] = strategy_line.split('STRATEGY:')[1].strip()
            
            # Extract RISK_LEVEL
            if "RISK_LEVEL:" in content:
                risk_line = [line for line in content.split('\n') if 'RISK_LEVEL:' in line][0]
                parsed_data['risk_level'] = risk_line.split('RISK_LEVEL:')[1].strip()
            
            # Extract RECOMMENDATION
            if "RECOMMENDATION:" in content:
                rec_line = [line for line in content.split('\n') if 'RECOMMENDATION:' in line][0]
                parsed_data['recommendation'] = rec_line.split('RECOMMENDATION:')[1].strip()
            
            # Extract POSITION_SIZE
            if "POSITION_SIZE:" in content:
                pos_line = [line for line in content.split('\n') if 'POSITION_SIZE:' in line][0]
                parsed_data['position_size'] = float(pos_line.split('POSITION_SIZE:')[1].strip())
            
            # Extract STOP_LOSS
            if "STOP_LOSS:" in content:
                sl_line = [line for line in content.split('\n') if 'STOP_LOSS:' in line][0]
                parsed_data['stop_loss'] = sl_line.split('STOP_LOSS:')[1].strip()
            
            # Extract TAKE_PROFIT
            if "TAKE_PROFIT:" in content:
                tp_line = [line for line in content.split('\n') if 'TAKE_PROFIT:' in line][0]
                parsed_data['take_profit'] = tp_line.split('TAKE_PROFIT:')[1].strip()
            
            # Extract RISK_FACTORS
            if "RISK_FACTORS:" in content:
                rf_line = [line for line in content.split('\n') if 'RISK_FACTORS:' in line][0]
                parsed_data['risk_factors'] = rf_line.split('RISK_FACTORS:')[1].strip()
            
            trading_logger.log_message(
                self.team,
                f"Parsed risk message: {parsed_data}",
                "INFO"
            )
            
            return parsed_data
            
        except Exception as e:
            # Handle parsing errors gracefully
            trading_logger.log_message(
                self.team,
                f"Error parsing risk message: {str(e)}",
                "ERROR"
            )
            return {}

    def _validate_order(self, order_data: Dict) -> Dict:
        """
        [TODO] Validate Order Before Execution
        
        Students should implement:
        1. Check if recommendation is "PROCEED"
        2. Validate order parameters:
           - Signal is valid (BUY/SELL)
           - Price is reasonable (not too far from current market)
           - Position size is within limits
           - Sufficient portfolio balance
        3. Check for duplicate orders
        4. Verify risk limits are not exceeded
        5. Return validation result with details
        
        Args:
            order_data: Parsed order information
            
        Returns:
            Dict: Validation result with is_valid flag and reasons
        """
        try:
            validation_result = {
                "is_valid": True,
                "reasons": [],
                "warnings": []
            }
            
            # 1. Check if recommendation is "PROCEED"
            recommendation = order_data.get('recommendation', '')
            if recommendation != "PROCEED":
                validation_result["is_valid"] = False
                validation_result["reasons"].append(f"Recommendation is {recommendation}, not PROCEED")
                return validation_result
            
            # 2. Validate order parameters
            signal = order_data.get('signal', '')
            price = order_data.get('price', 0)
            position_size = order_data.get('position_size', 0)
            
            # Check signal is valid (BUY/SELL)
            if signal not in ['BUY', 'SELL']:
                validation_result["is_valid"] = False
                validation_result["reasons"].append(f"Invalid signal: {signal}")
            
            # Check price is reasonable (not zero or negative)
            if price <= 0:
                validation_result["is_valid"] = False
                validation_result["reasons"].append(f"Invalid price: {price}")
            
            # Check position size is within limits
            if position_size <= 0:
                validation_result["is_valid"] = False
                validation_result["reasons"].append(f"Invalid position size: {position_size}")
            elif position_size > self.max_order_size:
                validation_result["is_valid"] = False
                validation_result["reasons"].append(f"Position size exceeds maximum: {position_size} > {self.max_order_size}")
            
            # Check sufficient portfolio balance (simulated)
            portfolio_balance = self._get_portfolio_balance()
            order_value = price * position_size
            if signal == 'BUY' and order_value > portfolio_balance:
                validation_result["is_valid"] = False
                validation_result["reasons"].append(f"Insufficient balance: {order_value} > {portfolio_balance}")
            
            # 3. Check for duplicate orders
            if self._check_duplicate_orders(order_data):
                validation_result["is_valid"] = False
                validation_result["reasons"].append("Duplicate order detected")
            
            # 4. Verify risk limits are not exceeded
            risk_level = order_data.get('risk_level', '')
            if risk_level == 'HIGH':
                validation_result["warnings"].append("High risk level detected")
            
            # Additional validations
            if order_value < 100:  # Minimum order size
                validation_result["warnings"].append("Order value below minimum threshold")
            
            trading_logger.log_message(
                self.team,
                f"Order validation result: {validation_result['is_valid']}, reasons: {validation_result['reasons']}",
                "INFO"
            )
            
            return validation_result
            
        except Exception as e:
            trading_logger.log_message(
                self.team,
                f"Error validating order: {str(e)}",
                "ERROR"
            )
            return {
                "is_valid": False,
                "reasons": [f"Validation error: {str(e)}"],
                "warnings": []
            }
    
    def _get_portfolio_balance(self) -> float:
        """Helper method to get current portfolio balance"""
        try:
            # This would typically query the database for current balance
            # For now, return a simulated balance
            return 100000  # $100,000 simulated balance
        except Exception:
            return 0.0
    
    def _check_duplicate_orders(self, order_data: Dict) -> bool:
        """Helper method to check for duplicate orders"""
        try:
            # This would typically check recent orders in the database
            # For now, return False (no duplicates)
            return False
        except Exception:
            return True  # Assume duplicate on error

    def _execute_market_order(self, signal: str, price: float, position_size: float) -> Dict:
        """
        [TODO] Execute Market Order
        
        Students should implement:
        1. Create order object with:
           - order_type: "market"
           - signal: BUY/SELL
           - quantity: Calculate from position_size and price
           - timestamp: Current time
           - expected_price: Current market price
        2. For simulation: Apply slippage to execution price
        3. For real trading: Send order to exchange API
        4. Store order in database
        5. Update portfolio positions
        6. Return execution result
        
        Args:
            signal: BUY or SELL
            price: Market price
            position_size: Position size as percentage
            
        Returns:
            Dict: Execution result with success flag and details
        """
        try:
            # 1. Create order object
            order_id = f"sim_{self.team}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            timestamp = datetime.now()
            
            # Calculate quantity from position_size and price
            portfolio_value = self._get_portfolio_balance()
            order_value = portfolio_value * position_size
            quantity = order_value / price
            
            order_object = {
                "order_id": order_id,
                "order_type": "market",
                "signal": signal,
                "quantity": quantity,
                "timestamp": timestamp,
                "expected_price": price,
                "team": self.team,
                "status": "pending"
            }
            
            # 2. For simulation: Apply slippage to execution price
            if self.simulation_mode:
                slippage_factor = 1 + (self.default_slippage if signal == 'BUY' else -self.default_slippage)
                executed_price = price * slippage_factor
                
                # Simulate execution delay
                import time
                time.sleep(0.1)  # Simulate 100ms execution time
                
                execution_result = {
                    "success": True,
                    "order_id": order_id,
                    "executed_price": executed_price,
                    "quantity": quantity,
                    "fees": order_value * 0.001,  # 0.1% fee
                    "slippage": abs(executed_price - price),
                    "execution_time": datetime.now(),
                    "simulation": True
                }
                
                trading_logger.log_message(
                    self.team,
                    f"Simulated market order executed: {signal} {quantity:.4f} @ ${executed_price:.2f}",
                    "INFO"
                )
                
            else:
                # 3. For real trading: Send order to exchange API
                # This would integrate with actual exchange APIs
                execution_result = {
                    "success": False,
                    "order_id": order_id,
                    "error": "Real trading not implemented",
                    "simulation": False
                }
                
                trading_logger.log_message(
                    self.team,
                    f"Real trading not implemented for order {order_id}",
                    "WARNING"
                )
            
            # 4. Store order in database
            if execution_result["success"]:
                order_record = {
                    **order_object,
                    "executed_price": execution_result["executed_price"],
                    "fees": execution_result["fees"],
                    "status": "executed"
                }
                
                # Store in database (simulated)
                self._store_order_in_database(order_record)
                
                # 5. Update portfolio positions
                self._update_position_tracking(signal, quantity, execution_result["executed_price"])
            
            return execution_result
            
        except Exception as e:
            trading_logger.log_message(
                self.team,
                f"Error executing market order: {str(e)}",
                "ERROR"
            )
            return {
                "success": False,
                "order_id": f"error_{datetime.now().timestamp()}",
                "error": str(e),
                "executed_price": 0,
                "quantity": 0,
                "fees": 0
            }
    
    def _store_order_in_database(self, order_record: Dict) -> None:
        """Helper method to store order in database"""
        try:
            # This would typically store in MongoDB
            # For now, just log the order
            trading_logger.log_message(
                self.team,
                f"Order stored in database: {order_record['order_id']}",
                "INFO"
            )
        except Exception as e:
            trading_logger.log_message(
                self.team,
                f"Error storing order in database: {str(e)}",
                "ERROR"
            )
    
    def _update_position_tracking(self, signal: str, quantity: float, price: float) -> None:
        """Helper method to update position tracking"""
        try:
            if signal == 'BUY':
                self.position_tracking[self.team] = self.position_tracking.get(self.team, 0) + quantity
            elif signal == 'SELL':
                self.position_tracking[self.team] = self.position_tracking.get(self.team, 0) - quantity
            
            trading_logger.log_message(
                self.team,
                f"Position updated: {signal} {quantity:.4f} @ ${price:.2f}, Total: {self.position_tracking.get(self.team, 0):.4f}",
                "INFO"
            )
        except Exception as e:
            trading_logger.log_message(
                self.team,
                f"Error updating position tracking: {str(e)}",
                "ERROR"
            )

    def _update_portfolio_positions(self, order_result: Dict, signal: str) -> None:
        """
        [TODO] Update Portfolio Positions
        
        Students should implement:
        1. Load current portfolio positions from database
        2. Update position based on executed order:
           - BUY: Add to position (or reduce short position)
           - SELL: Reduce position (or add to short position)
        3. Calculate new average cost basis
        4. Update portfolio value and cash balance
        5. Store updated positions in database
        6. Log position changes
        
        Args:
            order_result: Result from order execution
            signal: BUY or SELL signal
        """
        try:
            if not order_result.get("success", False):
                trading_logger.log_message(
                    self.team,
                    "Cannot update portfolio positions - order execution failed",
                    "WARNING"
                )
                return
            
            # 1. Load current portfolio positions from database
            current_positions = self._load_portfolio_positions()
            
            # Extract order details
            quantity = order_result.get("quantity", 0)
            executed_price = order_result.get("executed_price", 0)
            fees = order_result.get("fees", 0)
            
            if quantity <= 0 or executed_price <= 0:
                trading_logger.log_message(
                    self.team,
                    f"Invalid order details for position update: quantity={quantity}, price={executed_price}",
                    "ERROR"
                )
                return
            
            # 2. Update position based on executed order
            position_key = f"{self.team}_position"
            current_position = current_positions.get(position_key, {
                "quantity": 0,
                "average_cost": 0,
                "total_cost": 0,
                "unrealized_pnl": 0
            })
            
            if signal == 'BUY':
                # BUY: Add to position (or reduce short position)
                new_quantity = current_position["quantity"] + quantity
                new_total_cost = current_position["total_cost"] + (quantity * executed_price) + fees
                
                if new_quantity != 0:
                    new_average_cost = new_total_cost / new_quantity
                else:
                    new_average_cost = 0
                
                trading_logger.log_message(
                    self.team,
                    f"BUY position update: +{quantity:.4f} @ ${executed_price:.2f}, New total: {new_quantity:.4f}",
                    "INFO"
                )
                
            elif signal == 'SELL':
                # SELL: Reduce position (or add to short position)
                new_quantity = current_position["quantity"] - quantity
                
                # Calculate realized PnL for the sold portion
                sold_cost_basis = quantity * current_position["average_cost"]
                sale_proceeds = (quantity * executed_price) - fees
                realized_pnl = sale_proceeds - sold_cost_basis
                
                # Update total cost (reduce by sold cost basis)
                new_total_cost = current_position["total_cost"] - sold_cost_basis
                
                if new_quantity != 0:
                    new_average_cost = new_total_cost / new_quantity
                else:
                    new_average_cost = 0
                
                trading_logger.log_message(
                    self.team,
                    f"SELL position update: -{quantity:.4f} @ ${executed_price:.2f}, Realized PnL: ${realized_pnl:.2f}",
                    "INFO"
                )
                
            else:
                trading_logger.log_message(
                    self.team,
                    f"Invalid signal for position update: {signal}",
                    "ERROR"
                )
                return
            
            # 3. Calculate new average cost basis (already calculated above)
            
            # 4. Update portfolio value and cash balance
            updated_position = {
                "quantity": new_quantity,
                "average_cost": new_average_cost,
                "total_cost": new_total_cost,
                "unrealized_pnl": 0,  # Would calculate based on current market price
                "last_updated": datetime.now()
            }
            
            current_positions[position_key] = updated_position
            
            # 5. Store updated positions in database
            self._store_portfolio_positions(current_positions)
            
            # 6. Log position changes
            trading_logger.log_message(
                self.team,
                f"Portfolio position updated: {signal} {quantity:.4f} @ ${executed_price:.2f}, "
                f"New position: {new_quantity:.4f} @ ${new_average_cost:.2f}",
                "INFO"
            )
            
        except Exception as e:
            trading_logger.log_message(
                self.team,
                f"Error updating portfolio positions: {str(e)}",
                "ERROR"
            )
    
    def _load_portfolio_positions(self) -> Dict:
        """Helper method to load current portfolio positions"""
        try:
            # This would typically load from MongoDB
            # For now, return simulated positions
            return {
                f"{self.team}_position": {
                    "quantity": 0,
                    "average_cost": 0,
                    "total_cost": 0,
                    "unrealized_pnl": 0
                }
            }
        except Exception as e:
            trading_logger.log_message(
                self.team,
                f"Error loading portfolio positions: {str(e)}",
                "ERROR"
            )
            return {}
    
    def _store_portfolio_positions(self, positions: Dict) -> None:
        """Helper method to store portfolio positions"""
        try:
            # This would typically store in MongoDB
            trading_logger.log_message(
                self.team,
                f"Portfolio positions stored: {len(positions)} positions",
                "INFO"
            )
        except Exception as e:
            trading_logger.log_message(
                self.team,
                f"Error storing portfolio positions: {str(e)}",
                "ERROR"
            )

    def _store_order_record(self, order_data: Dict, execution_result: Dict) -> str:
        """
        [TODO] Store Order Record in Database
        
        Students should implement:
        1. Create order record with:
           - team: Trading team
           - timestamp: Execution time
           - signal: BUY/SELL
           - strategy: Strategy type
           - price: Execution price
           - quantity: Order quantity
           - fees: Trading fees
           - order_id: Unique order identifier
           - status: "executed", "failed", "rejected"
        2. Store in database orders collection
        3. Return order record ID
        
        Args:
            order_data: Original order data
            execution_result: Order execution result
            
        Returns:
            str: Database record ID
        """
        try:
            # 1. Create order record
            order_record = {
                "team": self.team,
                "timestamp": execution_result.get("execution_time", datetime.now()),
                "signal": order_data.get("signal", ""),
                "strategy": order_data.get("strategy", ""),
                "price": execution_result.get("executed_price", 0),
                "quantity": execution_result.get("quantity", 0),
                "fees": execution_result.get("fees", 0),
                "order_id": execution_result.get("order_id", ""),
                "status": "executed" if execution_result.get("success", False) else "failed",
                "risk_level": order_data.get("risk_level", ""),
                "position_size": order_data.get("position_size", 0),
                "stop_loss": order_data.get("stop_loss", ""),
                "take_profit": order_data.get("take_profit", ""),
                "slippage": execution_result.get("slippage", 0),
                "simulation": execution_result.get("simulation", True),
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
            
            # 2. Store in database orders collection
            record_id = self._save_order_to_database(order_record)
            
            # 3. Return order record ID
            trading_logger.log_message(
                self.team,
                f"Order record stored with ID: {record_id}",
                "INFO"
            )
            
            return record_id
            
        except Exception as e:
            trading_logger.log_message(
                self.team,
                f"Error storing order record: {str(e)}",
                "ERROR"
            )
            return f"error_record_{datetime.now().timestamp()}"
    
    def _save_order_to_database(self, order_record: Dict) -> str:
        """Helper method to save order to database"""
        try:
            # This would typically save to MongoDB orders collection
            # For now, simulate database save and return record ID
            record_id = f"order_{self.team}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            trading_logger.log_message(
                self.team,
                f"Order saved to database: {record_id} - {order_record['signal']} {order_record['quantity']:.4f} @ ${order_record['price']:.2f}",
                "INFO"
            )
            
            return record_id
            
        except Exception as e:
            trading_logger.log_message(
                self.team,
                f"Error saving order to database: {str(e)}",
                "ERROR"
            )
            return f"error_{datetime.now().timestamp()}"

    async def process(self, state: AgentState) -> AgentState:
        """
        [TODO] Implement Order Execution Workflow
        
        Students should implement:
        
        STEP 1: MESSAGE PARSING
        - Parse Risk Agent message using _parse_risk_message()
        - Extract signal, price, risk assessment
        - Handle parsing errors gracefully
        
        STEP 2: ORDER VALIDATION
        - Validate order using _validate_order()
        - Check recommendation is "PROCEED"
        - Verify all order parameters are valid
        
        STEP 3: ORDER EXECUTION (if validation passes)
        - Execute market order using _execute_market_order()
        - Handle execution errors and retries
        - Apply slippage and fees for simulation
        
        STEP 4: POSITION MANAGEMENT
        - Update portfolio positions using _update_portfolio_positions()
        - Store order record using _store_order_record()
        - Calculate new portfolio metrics
        
        STEP 5: COMMUNICATION
        - Send execution confirmation to Team Manager
        - Include: order_id, executed_price, quantity, fees
        - Add to state["messages"] and state["message_history"]
        
        STEP 6: STATE UPDATE
        - Mark order processing as complete
        - Log execution results
        - Return updated state
        
        Args:
            state: Current agent workflow state
            
        Returns:
            AgentState: Updated state with order execution results
        """
        # STEP 1: MESSAGE PARSING
        trading_logger.log_message(
            self.team,
            f"Starting order execution for {self.team}",
            "INFO"
        )
        print(f"OrderAgent: Starting order execution for {self.team}")
        
        order_data = self._parse_risk_message(state["messages"])
        if not order_data:
            trading_logger.log_message(
                self.team,
                f"No valid order data found for {self.team}",
                "WARNING"
            )
            print(f"OrderAgent: No valid order data found for {self.team}")
            # Mark as complete to avoid blocking workflow
            state["trading_teams"][self.team.lower()]["order_complete"] = True
            return state
        
        # STEP 2: ORDER VALIDATION
        validation_result = self._validate_order(order_data)
        
        if not validation_result["is_valid"]:
            trading_logger.log_message(
                self.team,
                f"Order validation failed: {validation_result['reasons']}",
                "WARNING"
            )
            print(f"OrderAgent: Order validation failed: {validation_result['reasons']}")
            
            # Send rejection message
            rejection_message = f"""TO: {self.team} Team Manager
MESSAGE: Order Execution - REJECTED
SIGNAL: {order_data.get('signal', 'N/A')}
PRICE: {order_data.get('price', 0)}
REASON: {', '.join(validation_result['reasons'])}
ACTION: Order rejected due to validation failure"""
            
            state["messages"].append(AIMessage(content=rejection_message))
            state["message_history"].append({
                "from": self.name,
                "content": f"Order REJECTED: {', '.join(validation_result['reasons'])}",
                "timestamp": datetime.now().isoformat()
            })
            
            # Mark as complete
            state["trading_teams"][self.team.lower()]["order_complete"] = True
            return state
        
        # Log warnings if any
        if validation_result["warnings"]:
            trading_logger.log_message(
                self.team,
                f"Order validation warnings: {validation_result['warnings']}",
                "WARNING"
            )
            print(f"OrderAgent: Validation warnings: {validation_result['warnings']}")
        
        # STEP 3: ORDER EXECUTION (if validation passes)
        signal = order_data.get('signal', '')
        price = order_data.get('price', 0)
        position_size = order_data.get('position_size', 0)
        
        execution_result = self._execute_market_order(signal, price, position_size)
        
        if not execution_result.get("success", False):
            trading_logger.log_message(
                self.team,
                f"Order execution failed: {execution_result.get('error', 'Unknown error')}",
                "ERROR"
            )
            print(f"OrderAgent: Order execution failed: {execution_result.get('error', 'Unknown error')}")
            
            # Send failure message
            failure_message = f"""TO: {self.team} Team Manager
MESSAGE: Order Execution - FAILED
SIGNAL: {signal}
PRICE: {price}
ERROR: {execution_result.get('error', 'Unknown error')}
ACTION: Order execution failed"""
            
            state["messages"].append(AIMessage(content=failure_message))
            state["message_history"].append({
                "from": self.name,
                "content": f"Order FAILED: {execution_result.get('error', 'Unknown error')}",
                "timestamp": datetime.now().isoformat()
            })
            
            # Mark as complete
            state["trading_teams"][self.team.lower()]["order_complete"] = True
            return state
        
        # STEP 4: POSITION MANAGEMENT
        self._update_portfolio_positions(execution_result, signal)
        
        record_id = self._store_order_record(order_data, execution_result)
        
        # STEP 5: COMMUNICATION
        success_message = f"""TO: {self.team} Team Manager
MESSAGE: Order Execution - SUCCESS
ORDER_ID: {execution_result.get('order_id', 'N/A')}
SIGNAL: {signal}
EXECUTED_PRICE: {execution_result.get('executed_price', 0):.2f}
QUANTITY: {execution_result.get('quantity', 0):.4f}
FEES: {execution_result.get('fees', 0):.2f}
SLIPPAGE: {execution_result.get('slippage', 0):.4f}
RECORD_ID: {record_id}
ACTION: Order executed successfully"""
        
        state["messages"].append(AIMessage(content=success_message))
        state["message_history"].append({
            "from": self.name,
            "content": f"Order EXECUTED: {signal} {execution_result.get('quantity', 0):.4f} @ ${execution_result.get('executed_price', 0):.2f}",
            "timestamp": datetime.now().isoformat()
        })
        
        # STEP 6: STATE UPDATE
        state["trading_teams"][self.team.lower()]["order_complete"] = True
        
        trading_logger.log_message(
            self.team,
            f"Order execution completed for {self.team}",
            "INFO"
        )
        
        print(f"OrderAgent: Order execution completed - {signal} {execution_result.get('quantity', 0):.4f} @ ${execution_result.get('executed_price', 0):.2f}")
        
        return state

# Portfolio Management Agent
import copy

class PortfolioAgent(BaseAgent):
    def __init__(self, name: str, role: str):
        super().__init__(name=name, role=role)
        self.team_managers = {}
        print(f"PortfolioAgent initialized - Students can extend with portfolio management logic!")

    def inject_managers(self, managers: Dict[str, BaseAgent]):
        """Inject team managers for coordination"""
        self.team_managers = managers
        print(f"PortfolioAgent: Injected {len(managers)} team managers")

    async def process(self, state: AgentState) -> AgentState:
        """
        TODO: Students can implement advanced portfolio management here
        
        Advanced features students can add:
        1. Portfolio rebalancing logic
        2. Risk correlation analysis across teams
        3. Position size optimization
        4. Cross-team arbitrage opportunities
        5. Overall portfolio performance tracking
        6. Dynamic allocation based on market conditions
        """
        print(f"PortfolioAgent: Managing portfolio operations...")
        
        if not state["task_status"].get("portfolio_initialized", False):
            print(f"PortfolioAgent: Initializing trading operations for all teams...")

            # Add team manager nodes to run in parallel
            state["parallel_tasks"] = [f"{team['name'].lower()}_manager" for team in config["trading_teams"]]
            state["task_status"]["portfolio_initialized"] = True

            for team in config["trading_teams"]:
                print(f"PortfolioAgent: Starting {team['name']} trading operations")
                
                state["messages"].append(AIMessage(
                    content=f"""TO: {team['name']} Manager
MESSAGE: Initialize trading operations for {team['name']}
ACTION: Begin trading operations and report status"""
                ))
                state["message_history"].append({
                    "from": self.name,
                    "content": f"Initialized {team['name']} trading operations",
                    "timestamp": datetime.now().isoformat()
                })

        # Run all team managers concurrently with separate state copies
        pending_tasks = state["parallel_tasks"][:]
        results = await asyncio.gather(*[
            self.team_managers[task].process(copy.deepcopy(state))
            for task in pending_tasks
        ])

        # Merge updates from all parallel tasks
        for result in results:
            state["messages"].extend(result["messages"])
            state["message_history"].extend(result["message_history"])
            state["task_status"].update(result["task_status"])
            state["trading_teams"].update(result["trading_teams"])

        # Clear parallel tasks after completion
        state["parallel_tasks"].clear()

        print(f"PortfolioAgent: Completed portfolio processing cycle")

        return state


# Trading Team Manager Agent
class TeamManager(BaseAgent):
    def __init__(self, team_name: str, config: dict):
        super().__init__(
            name=f"{team_name} Team Manager",
            role=f"responsible for coordinating {team_name} trading team operations"
        )
        self.team_name = team_name
        self.agents = {}
        self._initialize_agents(config)
        print(f"TeamManager initialized for {team_name} - Students can extend with team coordination logic!")

    def _initialize_agents(self, config: dict):
        """Initialize the team's agents based on configuration"""
        team_config = next((team for team in config["trading_teams"] if team["name"] == self.team_name), None)
        if not team_config:
            raise ValueError(f"Team {self.team_name} not found in configuration")

        for agent_config in team_config["agents"]:
            agent = AgentFactory.create_agent(
                agent_type=agent_config["name"],
                team=self.team_name,
                role=agent_config["role"]
            )
            self.agents[agent_config["name"].lower()] = agent
        
        print(f"TeamManager: Initialized {len(self.agents)} agents for {self.team_name}")

    async def process(self, state: AgentState) -> AgentState:
        """
        TODO: Students can implement advanced team coordination here
        
        Advanced features students can add:
        1. Dynamic agent priority based on market conditions
        2. Agent failure recovery and fallback strategies
        3. Performance monitoring and optimization
        4. Custom workflow routing based on signals
        5. Team-specific configuration management
        """
        workflow_started_key = f"{self.team_name.lower()}_workflow_started"
        workflow_complete_key = f"{self.team_name.lower()}_complete"

        if not state["task_status"].get(workflow_started_key, False):
            state["task_status"][workflow_started_key] = True
            state["current_workflow"] = self.team_name.lower()
            
            print(f"TeamManager: Starting {self.team_name} trading workflow")
            
            state["messages"].append(AIMessage(
                content=f"""TO: Portfolio Manager
MESSAGE: Starting {self.team_name} trading workflow
ACTION: Acknowledge workflow start"""
            ))
            state["message_history"].append({
                "from": self.name,
                "content": f"Starting {self.team_name} trading workflow",
                "timestamp": datetime.now().isoformat()
            })
        
        if not state["task_status"].get(workflow_complete_key, False):
            await self.execute_trading_team_workflow(state)
            state["task_status"][workflow_complete_key] = True
            
            print(f"TeamManager: Completed {self.team_name} trading workflow")
            
            state["messages"].append(AIMessage(
                content=f"""TO: Portfolio Manager
MESSAGE: {self.team_name} trading workflow completed
ACTION: Review results"""
            ))
            state["message_history"].append({
                "from": self.name,
                "content": f"{self.team_name} trading workflow completed",
                "timestamp": datetime.now().isoformat()
            })
        
        if f"{self.team_name.lower()}_manager" in state["parallel_tasks"]:
            state["parallel_tasks"].remove(f"{self.team_name.lower()}_manager")
        
        return state

    async def execute_trading_team_workflow(self, state: AgentState):
        """
        Execute the trading workflow for this team
        
        TODO: Students can customize workflow execution here
        - Add conditional routing based on market conditions
        - Implement parallel agent execution for speed
        - Add retry logic for failed agents
        - Implement custom workflow patterns
        """
        print(f"TeamManager: Executing {self.team_name} agent workflow...")
        
        # Execute agents in the sequence defined in config
        for agent_type in config["workflow"]["agent_sequence"]:
            agent = self.agents[agent_type.lower()]
            state["current_agent"] = agent.name
            
            print(f"TeamManager: Running {agent_type} agent for {self.team_name}")
            
            state = await agent.process(state)
            
            print(f"TeamManager: Completed {agent_type} agent for {self.team_name}")

# =============================================================================
# DATA AGENT - IMPLEMENTATION TEMPLATE
# =============================================================================

class DataAgent(BaseAgent):
    """
    [STUDENT TODO] Implement Market Data Fetching Agent
    
    This agent should:
    1. Fetch real-time market data from Yahoo Finance API
    2. Handle timezone normalization and data validation
    3. Store price snapshots in MongoDB
    4. Manage data freshness and caching logic
    5. Provide error handling for API failures
    6. Communicate data to StrategyAgent
    """
    
    def __init__(self, team: str, role: str):
        """
        [TODO] Initialize Data Agent
        
        Students should implement:
        1. Call parent class __init__ with agent name and role
        2. Store team name for this agent
        3. Get symbol from config: config["data_fetching"]["symbols"][team]
        4. Get interval from config: config["data_fetching"]["interval"]
        5. Initialize last_fetch_time as None
        6. Initialize database service (MongoDBService)
        7. Initialize yfinance Ticker: yf.Ticker(self.symbol)
        8. Log successful initialization
        """
        # 1. Call parent class __init__ - REQUIRED for agent to work
        super().__init__(
            name=f"{team} Data Agent",
            role=role
        )
        
        # 2. Store team name for this agent
        self.team = team
        
        # 3. Get symbol from config: config["data_fetching"]["symbols"][team]
        self.symbol = config["data_fetching"]["symbols"][team]
        
        # 4. Get interval from config: config["data_fetching"]["interval"]
        self.interval = config["data_fetching"]["interval"]
        
        # 5. Initialize last_fetch_time as None
        self.last_fetch_time = None
        
        # 6. Initialize database service (MongoDBService)
        self.db_service = MongoDBService()
        
        # 7. Initialize yfinance Ticker: yf.Ticker(self.symbol)
        self.yf_token = yf.Ticker(self.symbol)
        
        # 8. Log successful initialization
        trading_logger.log_message(
            self.team,
            f"DataAgent initialized for {self.symbol} with interval {self.interval}",
            "INFO"
        )
        
        print(f"DataAgent initialized for {team} - Symbol: {self.symbol}, Interval: {self.interval}")

    async def fetch_market_data(self) -> Optional[PriceSnapshot]:
        """
        [TODO] Fetch Market Data from Yahoo Finance
        
        Students should implement:
        1. Use try-except for error handling
        2. Log attempt to fetch data
        3. Fetch 5 days of hourly data: self.yf_token.history(period="5d", interval="1h")
        4. Check if market_data is empty, return None if so
        5. Create price_data_points list
        6. Loop through market_data.iterrows():
           - Handle timezone: localize to UTC if no timezone, convert to UTC if exists
           - Create PriceDataPoint with: datetime, price (Close), open, high, low, volume, price_change
           - Append to price_data_points list
        7. Create PriceSnapshot object with:
           - token_symbol: self.team
           - source: "yfinance"
           - metadata: interval, symbol, period, data_points count, latest_price
           - price_data: price_data_points list
        8. Save to database: self.db_service.save_price_snapshot(snapshot)
        9. Log success and return snapshot
        10. Handle exceptions with error logging
        
        Returns:
            Optional[PriceSnapshot]: Price snapshot or None if failed
        """
        # TODO: Implement market data fetching
        print(f"TODO: {self.team} DataAgent.fetch_market_data() needs implementation!")
        return None

    async def process(self, state: AgentState) -> AgentState:
        """
        [TODO] Implement Data Fetching Workflow
        
        Students should implement:
        
        STEP 1: TIME CHECK
        - Get current time: datetime.now()
        - Check if it's time to fetch new data:
          * If last_fetch_time is None OR
          * If (current_time - last_fetch_time).total_seconds() >= self.interval
        
        STEP 2: FETCH NEW DATA (if time check passes)
        - Log that you're fetching real-time data
        - Call self.fetch_market_data()
        - If successful:
          * Update self.last_fetch_time = current_time
          * Create message for StrategyAgent with: symbol, data_points count, latest_price
          * Add to state["messages"] and state["message_history"]
          * Mark data_complete: state["trading_teams"][self.team.lower()]["data_complete"] = True
          * Log success
        - If failed:
          * Log error
          * Send error message to StrategyAgent
          * Still mark data_complete to avoid blocking workflow
        
        STEP 3: USE CACHED DATA (if time check fails)
        - Calculate time until next fetch
        - Try to get latest snapshot from database: self.db_service.get_latest_snapshot(self.team)
        - If snapshot exists:
          * Send message to StrategyAgent with cached data details
          * Add to state["messages"] and state["message_history"]
        - If no snapshot:
          * Send error message about no data available
        - Mark data_complete
        
        STEP 4: RETURN STATE
        - Return updated state
        
        Args:
            state: Current agent workflow state
            
        Returns:
            AgentState: Updated state with data fetching results
        """
        # TODO: Implement complete data fetching workflow
        print(f"TODO: {self.team} DataAgent.process() needs implementation!")
        print("HINT: Check time, fetch new data or use cached data, communicate with StrategyAgent")
        
        # Mark as complete to avoid blocking workflow during development
        state["trading_teams"][self.team.lower()]["data_complete"] = True
        return state

    def __del__(self):
        """
        [TODO] Implement Resource Cleanup
        
        Students should implement:
        1. Check if self.db_service exists using hasattr()
        2. Call self.db_service.close() to cleanup database connection
        3. This prevents resource leaks when agent is destroyed
        """
        # TODO: Implement resource cleanup
        pass

def create_hybrid_graph():
    # Initialize agents from config
    portfolio_agent = PortfolioAgent(
        name=config["portfolio_manager"]["name"],
        role=config["portfolio_manager"]["role"]
    )
    


# Inject managers into portfolio agent

    
    team_managers = {
        f"{team['name'].lower()}_manager": TeamManager(team["name"], config)
        for team in config["trading_teams"]
    }
    portfolio_agent.inject_managers(team_managers)

    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes for each agent
    workflow.add_node("portfolio", portfolio_agent.process)
    for manager_name, manager in team_managers.items():
        workflow.add_node(manager_name, manager.process)

    # Define the edges and conditions

    def should_continue(state: AgentState) -> str:
        if not state["task_status"].get("portfolio_initialized", False):
            return "portfolio"
        
        if state["parallel_tasks"]:
            return state["parallel_tasks"][0]
        
        all_complete = all(
            state["task_status"].get(f"{team['name'].lower()}_complete", False)
            for team in config["trading_teams"]
        )
        if all_complete:
            return END
        
        # Optional fallback
        return END


    # Add edges with conditional routing
    workflow.add_conditional_edges(
        "portfolio",
        should_continue,
        {
            **{manager_name: manager_name for manager_name in team_managers.keys()},
            END: END
        }
    )

    # Add edges for team managers
    for manager_name in team_managers.keys():
        workflow.add_conditional_edges(
            manager_name,
            should_continue,
            {
                "portfolio": "portfolio",
                **{name: name for name in team_managers.keys()},
                END: END
            }
        )

    # Set the entry point
    workflow.set_entry_point(config["workflow"]["entry_point"])

    # Compile the graph
    chain = workflow.compile()
    return chain

async def run_continuous_workflow(chain, initial_state):
    """Run the workflow continuously at the specified interval"""
    state = initial_state.copy()
    interval = config["data_fetching"]["interval"]
    
    while True:
        try:
            # Log the start of a new cycle
            trading_logger.log_message(
                "system",
                f"Starting new trading cycle at {datetime.now().isoformat()}",
                "INFO"
            )
            
            # Run the workflow
            result = await chain.ainvoke(state)
            
            # Update state for next cycle
            state = {
                "messages": result["messages"][-10:],  # Keep last 10 messages
                "current_agent": "portfolio",
                "message_history": result["message_history"][-50:],  # Keep last 50 history entries
                "task_status": {
                    "portfolio_initialized": False,  # Reset for new cycle
                    **{f"{team['name'].lower()}_workflow_started": False for team in config["trading_teams"]},
                    **{f"{team['name'].lower()}_complete": False for team in config["trading_teams"]}
                },
                "parallel_tasks": [],
                "current_workflow": "",
                "trading_teams": {
                    team["name"].lower(): {
                        f"{agent['name'].lower()}_complete": False
                        for agent in team["agents"]
                    }
                    for team in config["trading_teams"]
                }
            }
            
            # Print the results of this cycle
            print(f"\nTrading Cycle Results - {datetime.now()}")
            print("=" * 50)
            for message in result["message_history"][-10:]:  # Show last 10 messages
                print(f"\nFrom: {message['from']}")
                print(f"Time: {message['timestamp']}")
                print(f"Content:\n{message['content']}")
                print("-" * 50)
            
            # Wait for the next cycle
            await asyncio.sleep(interval)
            
        except Exception as e:
            trading_logger.log_message(
                "system",
                f"Error in trading cycle: {str(e)}",
                "ERROR"
            )
            # Wait a bit before retrying
            await asyncio.sleep(5)

async def main():
    # Create the graph
    chain = create_hybrid_graph()

    # Initialize the state
    initial_state = {
        "messages": [
            SystemMessage(content="Initialize the trading system and establish communication between agents."),
            HumanMessage(content="Coordinate trading operations between portfolio manager and trading teams.")
        ],
        "current_agent": "portfolio",
        "message_history": [],
        "task_status": {
            "portfolio_initialized": False,
            **{f"{team['name'].lower()}_workflow_started": False for team in config["trading_teams"]},
            **{f"{team['name'].lower()}_complete": False for team in config["trading_teams"]}
        },
        "parallel_tasks": [],
        "current_workflow": "",
        "trading_teams": {
            team["name"].lower(): {
                f"{agent['name'].lower()}_complete": False
                for agent in team["agents"]
            }
            for team in config["trading_teams"]
        }
    }

    # Run the continuous workflow
    await run_continuous_workflow(chain, initial_state)

# =============================================================================
# STUDENT IMPLEMENTATION GUIDE
# =============================================================================
"""
IMPLEMENTATION ROADMAP FOR :

PHASE 1 - STRATEGY AGENT (PRIORITY 1)
======================================
[ ] Focus on StrategyAgent class first - this is the core of the system
[ ] Implement all 4 methods: __init__, _load_optimized_parameters, _convert_price_data_to_dataframe, process
[ ] Use the provided functions from strategy.MA.ma_strategy module
[ ] Test with existing DataAgent to ensure data flow works

PHASE 2 - RISK AGENT (PRIORITY 2)  
==================================
[ ] Implement message parsing to extract StrategyAgent signals
[ ] Create basic risk assessment logic (position sizing, exposure limits)
[ ] Forward approved signals to OrderAgent

PHASE 3 - ORDER AGENT (PRIORITY 3)
===================================
[ ] Implement order validation and execution (simulation mode first)
[ ] Add portfolio position tracking
[ ] Store order records in database

PHASE 4 - TESTING & ENHANCEMENT
================================
[ ] Run complete workflow and verify agent communication
[ ] Add advanced risk features (VaR, correlation analysis)
[ ] Enhance portfolio management features
[ ] Add real exchange connectivity (advanced)

TESTING STRATEGY:
1. Start by running main() and checking console output
2. Verify DataAgent fetches market data successfully  
3. Implement StrategyAgent and check signal generation
4. Add RiskAgent to test risk assessment flow
5. Complete OrderAgent for full workflow
6. Monitor database for stored data and orders

DEBUGGING TIPS:
- Check logs/ directory for detailed logging
- Use trading_logger for all important events
- Add print statements for debugging workflow
- Monitor state["message_history"] for agent communication
- Verify database connections and data storage

SUCCESS CRITERIA:
[ ] All agents implement their TODO methods
[ ] Agents communicate successfully via messages
[ ] Market data flows from DataAgent to StrategyAgent
[ ] Signals flow from StrategyAgent to RiskAgent to OrderAgent
[ ] Orders are validated, executed, and stored
[ ] Portfolio positions are updated correctly
[ ] Email notifications work for trading signals
[ ] System runs continuously without errors

BONUS CHALLENGES:
- Add multiple trading strategies (Bollinger, RSI, MACD)
- Implement real exchange API integration  
- Add machine learning for signal optimization
- Create web dashboard for monitoring
- Add advanced portfolio rebalancing
- Implement cross-team arbitrage detection
"""

if __name__ == "__main__":
    """
    STUDENTS: This is where your implemented system will run!
    
    When you complete the agent implementations:
    1. The system will start with Portfolio Agent
    2. Portfolio Agent will initialize all Team Managers
    3. Each Team Manager will run: DataAgent -> StrategyAgent -> RiskAgent -> OrderAgent
    4. The cycle repeats continuously based on config interval
    5. Check console output and logs/ directory for results
    """
    try:
        print("Starting Multi-Agent Trading System...")
        print("STUDENT NOTE: Implement the TODO methods in agents to make this work!")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down trading system...")
        trading_logger.log_message("system", "Trading system shutdown initiated", "INFO")
    except Exception as e: 
        print(f"\nERROR in main execution: {str(e)}")
        print("HINT: Check if all agent TODO methods are implemented!")
        trading_logger.log_message("system", f"Error in main execution: {str(e)}", "ERROR") 