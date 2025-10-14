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
        
        # TODO: Implement remaining initialization (steps 2-7)
        # Hint: self.team = team
        # Hint: self.db_service = MongoDBService()
        # Hint: self.email_service = EmailService()
        # Hint: self.config = {"ma_fast": 10, "ma_slow": 20, ...}
        # Hint: self._load_optimized_parameters()
        
        print(f"TODO: {team} StrategyAgent needs implementation!")
        print(f"HINT: Complete steps 2-7 in __init__ method")

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
        # TODO: Implement optimized parameter loading
        print(f"TODO: {self.team} StrategyAgent._load_optimized_parameters() needs implementation!")
        pass


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
        # TODO: Implement price data to DataFrame conversion
        print(f"TODO: {self.team} StrategyAgent._convert_price_data_to_dataframe() needs implementation!")
        return pd.DataFrame()

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
        # TODO: Implement complete MA strategy workflow 
        print(f"TODO: {self.team} StrategyAgent.process() needs complete implementation!")
        print("HINT: Use functions from strategy.MA.ma_strategy module!")
        print("HINT: Follow the 8 steps outlined in the docstring above!")
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
        
        # TODO: Implement remaining initialization (steps 2-6)
        # Hint: self.team = team
        # Hint: self.db_service = MongoDBService()
        # Hint: Set up risk parameters in a dict or as attributes
        
        print(f"TODO: {team} RiskAgent needs implementation!")
        print(f"HINT: Complete steps 2-6 in __init__ method")

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
        # TODO: Implement message parsing
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
        # TODO: Implement position sizing logic
        return 0.05  # Placeholder: 5% position size

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
        # TODO: Implement risk assessment
        return {
            "risk_level": "MEDIUM",
            "risk_factors": ["TODO: Implement risk factor analysis"],
            "recommendation": "PROCEED",
            "adjusted_position_size": 0.05
        }

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
        # TODO: Implement complete risk assessment workflow
        print(f"TODO: {self.team} RiskAgent.process() needs implementation!")
        print("HINT: Start with _parse_strategy_message() to extract signal data")
        print("HINT: Use _assess_risk_level() to evaluate risks")
        print("HINT: Forward approved signals to Order Agent")
        
        # Mark as complete to avoid blocking workflow during development
        state["trading_teams"][self.team.lower()]["risk_complete"] = True
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
        
        # TODO: Implement remaining initialization (steps 2-7)
        # Hint: self.team = team
        # Hint: self.db_service = MongoDBService()
        # Hint: Set up order parameters as attributes or in a dict
        
        print(f"TODO: {team} OrderAgent needs implementation!")
        print(f"HINT: Complete steps 2-7 in __init__ method")

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
        # TODO: Implement risk message parsing
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
        # TODO: Implement order validation
        return {"is_valid": True, "reasons": []}

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
        # TODO: Implement market order execution
        return {
            "success": True,
            "order_id": "sim_" + str(datetime.now().timestamp()),
            "executed_price": price,
            "quantity": position_size,
            "fees": 0.001 * position_size  # 0.1% fee
        }

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
        # TODO: Implement portfolio position updates
        pass

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
        # TODO: Implement order record storage
        return "sim_record_" + str(datetime.now().timestamp())

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
        # TODO: Implement complete order execution workflow
        print(f"TODO: {self.team} OrderAgent.process() needs implementation!")
        print("HINT: Start with _parse_risk_message() to extract order data")
        print("HINT: Use _validate_order() before executing")
        print("HINT: Update portfolio positions after execution")
        
        # Mark as complete to avoid blocking workflow during development
        state["trading_teams"][self.team.lower()]["order_complete"] = True
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
        
        # TODO: Implement remaining initialization (steps 2-8)
        # Hint: self.team = team
        # Hint: self.symbol = config["data_fetching"]["symbols"][team]
        # Hint: self.interval = config["data_fetching"]["interval"]
        # Hint: self.last_fetch_time = None
        # Hint: self.db_service = MongoDBService()
        # Hint: self.yf_token = yf.Ticker(self.symbol)
        
        print(f"TODO: {team} DataAgent needs implementation!")
        print(f"HINT: Complete steps 2-8 in __init__ method")

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