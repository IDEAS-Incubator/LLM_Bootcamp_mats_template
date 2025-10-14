# Multi-Agent Trading System - Student Bootcamp Template

A hands-on learning project to build a multi-agent cryptocurrency trading system using LangChain, LangGraph, and multiple trading strategies.

**What you'll build**: Implement 4 core agents (19 total methods) to create a complete automated trading system that fetches real market data, generates trading signals, assesses risk, and executes orders.

## Quick Start

```bash
# 1. Setup environment
cd LLM_Bootcamp_mats_template
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate
pip install -r requirements.txt

# 2. Create .env file
cat > .env << EOF
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=trading_system
EMAIL_SENDER=your-email@gmail.com
EMAIL_PASSWORD=your-gmail-app-password
EOF

# 3. Initialize database and run
python load_portfolio.py
python load_historical_data.py
python main.py
```

## System Architecture

```
PortfolioAgent (System Coordinator) [PROVIDED]
    ├── BTC TeamManager [PROVIDED]
    │   ├── DataAgent [TODO] → StrategyAgent [TODO] → RiskAgent [TODO] → OrderAgent [TODO]
    ├── SOL TeamManager [PROVIDED]
    │   ├── DataAgent [TODO] → StrategyAgent [TODO] → RiskAgent [TODO] → OrderAgent [TODO]
    └── DOGE TeamManager [PROVIDED]
        ├── DataAgent [TODO] → StrategyAgent [TODO] → RiskAgent [TODO] → OrderAgent [TODO]
```

**Implementation Overview:**

| Agent | Status | Methods | What You'll Learn |
|-------|--------|---------|-------------------|
| **DataAgent** | TODO | 3 methods | Yahoo Finance API, data caching, timezone handling |
| **StrategyAgent** | TODO | 4 methods | Trading strategies (MA/EMA/MACD/RSI/VP), technical analysis |
| **RiskAgent** | TODO | 5 methods | Position sizing, risk assessment, portfolio limits |
| **OrderAgent** | TODO | 7 methods | Order execution, portfolio updates, trade records |
| **PortfolioAgent** | PROVIDED | - | System coordination (extensible) |
| **TeamManager** | PROVIDED | - | Team workflow management (extensible) |

**Total: 19 methods to implement across 4 core agents**

## Key Features

### Available Trading Strategies
You can implement any of these strategies (or create your own):

- **Moving Average (MA)**: Simple/Exponential moving average crossovers and trends
- **Exponential Moving Average (EMA)**: Faster-responding moving averages for trend detection
- **MACD**: Moving Average Convergence Divergence for momentum analysis
- **RSI**: Relative Strength Index for overbought/oversold conditions
- **Volume Profile (VP)**: Volume-based analysis and support/resistance levels

**Additional Technical Indicators Available**:
- Momentum indicators, volume analysis, Fibonacci levels, candlestick patterns

### Risk Management
- Maximum 10% position size per trade
- 80% maximum portfolio exposure
- Dynamic position sizing based on volatility
- Stop-loss and take-profit levels

### Order Execution
- Simulated market orders with realistic slippage
- Complete order history and portfolio tracking
- Real-time position and balance updates

## File Structure

```
src/
├── agents/multi_agent_system.py    # YOUR IMPLEMENTATION WORK HERE
├── strategy/                       # Multiple trading strategies (PROVIDED)
│   ├── MA/ma_strategy.py           # Moving Average strategy
│   ├── EMA/ema_strategy.py         # Exponential Moving Average strategy
│   ├── MACD/macd_strategy.py       # MACD strategy
│   ├── RSI/rsi_strategy.py         # RSI strategy
│   └── VP/vp_strategy.py           # Volume Profile strategy
├── services/                       # Database and email services (PROVIDED)
└── config/trading_config.yaml     # System configuration (PROVIDED)
```

## What You Need to Implement

### 1. DataAgent (3 methods)
**File**: `src/agents/multi_agent_system.py` - Lines 910-1021

**Methods to implement:**
- `__init__()`: Initialize Yahoo Finance ticker, database service, and fetch interval
- `fetch_market_data()`: Fetch 5 days of hourly OHLCV data from Yahoo Finance API
- `process()`: Time-based data fetching workflow with caching logic

**Key APIs you'll use:**
```python
import yfinance as yf

# Initialize ticker
self.yf_token = yf.Ticker(self.symbol)

# Fetch data
market_data = self.yf_token.history(period="5d", interval="1h")

# Create PriceDataPoint objects
from core.data_models import PriceSnapshot, PriceDataPoint
```

### 2. StrategyAgent (4 methods)
**File**: `src/agents/multi_agent_system.py` - Lines 174-302

**Methods to implement:**
- `__init__()`: Initialize database, email services, and strategy config
- `_load_optimized_parameters()`: Load strategy parameters from MongoDB
- `_convert_price_data_to_dataframe()`: Convert price data to pandas DataFrame
- `process()`: 8-step workflow to generate BUY/SELL/HOLD signals

**Choose any strategy to implement (examples):**

```python
# Moving Average Strategy
from strategy.MA.ma_strategy import (
    add_multi_timeframe_moving_averages,
    generate_ma_signal
)

# EMA Strategy  
from strategy.EMA.ema_strategy import (
    add_ema_indicators,
    generate_ema_signal
)

# MACD Strategy
from strategy.MACD.macd_strategy import (
    add_macd_indicators,
    generate_macd_signal
)

# RSI Strategy
from strategy.RSI.rsi_strategy import (
    add_rsi_indicators,
    generate_rsi_signal
)

# Volume Profile Strategy
from strategy.VP.vp_strategy import (
    add_volume_profile,
    generate_vp_signal
)
```

**You can implement any strategy or combine multiple strategies!**

### 3. RiskAgent (5 methods)
**File**: `src/agents/multi_agent_system.py` - Lines 357-497

**Methods to implement:**
- `__init__()`: Setup risk parameters (max position size, stop loss, etc.)
- `_parse_strategy_message()`: Extract signals from StrategyAgent messages
- `_calculate_position_size()`: Calculate safe position sizes based on risk
- `_assess_risk_level()`: Perform risk analysis and return recommendation
- `process()`: Parse signal, assess risk, forward to OrderAgent

### 4. OrderAgent (7 methods)
**File**: `src/agents/multi_agent_system.py` - Lines 517-716

**Methods to implement:**
- `__init__()`: Setup order execution parameters
- `_parse_risk_message()`: Extract risk assessment from RiskAgent
- `_validate_order()`: Validate order before execution
- `_execute_market_order()`: Execute simulated trade
- `_update_portfolio_positions()`: Update positions in database
- `_store_order_record()`: Store order history
- `process()`: Complete order execution workflow

## Provided Agents (Already Working)

### 5. PortfolioAgent [EXTENSIBLE]
**File**: `src/agents/multi_agent_system.py` - Lines 721-787

**What it does:**
- Coordinates all trading teams (BTC, SOL, DOGE)
- Runs team managers in parallel for efficiency
- Manages system-wide workflow state
- Provides extension points for advanced portfolio features

**Optional extensions you can add:**
- Portfolio rebalancing logic
- Risk correlation analysis across teams
- Cross-team arbitrage opportunities
- Dynamic allocation based on market conditions

### 6. TeamManager [EXTENSIBLE]
**File**: `src/agents/multi_agent_system.py` - Lines 790-892

**What it does:**
- Manages individual team operations (BTC team, SOL team, DOGE team)
- Coordinates agent execution sequence: Data → Strategy → Risk → Order
- Handles workflow state for each team
- Reports results back to PortfolioAgent

**Optional extensions you can add:**
- Advanced team coordination logic
- Agent failure recovery strategies
- Performance monitoring per team

## Implementation Steps

1. **Start with DataAgent**: Learn Yahoo Finance API - implement 3 methods to fetch market data
2. **Implement StrategyAgent**: Choose your trading strategy (MA/EMA/MACD/RSI/VP) - implement 4 methods
3. **Add RiskAgent**: Implement risk assessment and position sizing - 5 methods
4. **Complete OrderAgent**: Add order execution and portfolio updates - 7 methods  
5. **Test full workflow**: Verify all agents communicate properly and workflow completes

```bash
# Test your implementation
python main.py

# Check logs for detailed execution
tail -f logs/trading_system.log

# Monitor database with MongoDB Compass
```

## Learning Objectives

1. **API Integration**: Learn to fetch real-time market data from Yahoo Finance API (DataAgent)
2. **Technical Analysis**: Apply your chosen trading strategy (MA, EMA, MACD, RSI, VP) to real market data (StrategyAgent)
3. **Risk Management**: Implement position sizing and portfolio risk assessment (RiskAgent)
4. **Order Execution**: Handle trade execution and portfolio management (OrderAgent)
5. **Multi-Agent Systems**: Build coordinated agents that communicate via structured messages
6. **Database Integration**: Store and retrieve trading data with MongoDB
7. **Real-time Processing**: Create a continuous workflow that processes live market data
8. **Strategy Flexibility**: Learn to integrate different trading strategies and technical indicators

## Troubleshooting

**MongoDB Connection Issues**:
```bash
# macOS
brew services start mongodb-community

# Linux  
sudo systemctl start mongod

# Windows
net start MongoDB
```

**Yahoo Finance API Issues**:
- Check internet connection
- Verify symbols: BTC-USD, SOL-USD, DOGE-USD
- API rate limits: Wait 1-2 minutes between requests

**Common Python Errors**:
- **Import Errors**: Activate virtual environment and install requirements
- **Agent Errors**: Check `logs/trading_system.log` for detailed error messages
- **Database Errors**: Ensure MongoDB is running and accessible

## Key Resources

- **Implementation Template**: Detailed TODO instructions in `src/agents/multi_agent_system.py`
- **Strategy Functions**: Complete strategy implementations available:
  - `src/strategy/MA/ma_strategy.py` - Moving Average strategy
  - `src/strategy/EMA/ema_strategy.py` - Exponential Moving Average strategy
  - `src/strategy/MACD/macd_strategy.py` - MACD strategy
  - `src/strategy/RSI/rsi_strategy.py` - RSI strategy
  - `src/strategy/VP/vp_strategy.py` - Volume Profile strategy
- **Provided Infrastructure**: PortfolioAgent and TeamManager show multi-agent coordination patterns
- **Documentation**: LangGraph, pandas, MongoDB, yfinance APIs

## Success Criteria

When fully implemented, your system will:
1. Fetch real market data for BTC, SOL, and DOGE from Yahoo Finance API (DataAgent - you implement)
2. Generate BUY/SELL/HOLD signals using your chosen trading strategy (StrategyAgent - you implement)
3. Assess risk and calculate position sizes (RiskAgent - you implement)  
4. Execute simulated trades and update portfolios (OrderAgent - you implement)
5. Send email notifications for trading signals
6. Run continuously with comprehensive logging and data caching

**Strategy Flexibility**: You can implement MA, EMA, MACD, RSI, VP, or create your own custom strategy!

---

## Next Steps & Extensions

Once you complete the core implementation, consider these enhancements:

**Advanced Features**:
- Implement multiple trading strategies simultaneously
- Add real exchange API integration (Binance, Coinbase)
- Create a web dashboard for monitoring trades
- Add machine learning for signal optimization
- Implement backtesting with historical data

**Portfolio Enhancements**:
- Cross-team arbitrage detection
- Dynamic position rebalancing
- Risk correlation analysis across teams
- Performance analytics and reporting

**Production Ready**:
- Add comprehensive unit tests
- Implement proper error handling and retries
- Add configuration management for different environments
- Set up proper logging and monitoring

---

**Ready to start coding!** Open `src/agents/multi_agent_system.py` and begin with the DataAgent implementation!

**Need help?** Check the detailed TODO comments in the code - they provide step-by-step implementation guidance.