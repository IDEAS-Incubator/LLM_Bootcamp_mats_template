import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, AsyncMock, patch
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.data_models import PriceSnapshot, PriceDataPoint
from agents.multi_agent_system import (
    AgentState, BaseAgent, DataAgent, StrategyAgent, 
    RiskAgent, OrderAgent, PortfolioAgent, TeamManager
)

@pytest.fixture
def sample_price_data():
    """Sample price data for testing"""
    base_time = datetime.now()
    data_points = []
    
    for i in range(100):
        time = base_time + timedelta(hours=i)
        data_points.append(PriceDataPoint(
            datetime=time.strftime("%Y-%m-%d %H:%M:%S+00:00"),
            price=100.0 + i * 0.1,
            open=100.0 + i * 0.05,
            high=100.0 + i * 0.15,
            low=100.0 + i * 0.02,
            volume=1000.0 + i * 10,
            price_change=0.1
        ))
    
    return PriceSnapshot(
        token_symbol="BTC",
        source="yahoo_finance",
        price_data=data_points
    )

@pytest.fixture
def sample_dataframe():
    """Sample pandas DataFrame for testing"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
    data = {
        'Open': [100 + i * 0.05 for i in range(100)],
        'High': [100 + i * 0.15 for i in range(100)],
        'Low': [100 + i * 0.02 for i in range(100)],
        'Close': [100 + i * 0.1 for i in range(100)],
        'Volume': [1000 + i * 10 for i in range(100)]
    }
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def mock_db_service():
    """Mock database service"""
    mock_service = Mock()
    mock_service.db = Mock()
    mock_service.db.optimized_parameters = Mock()
    mock_service.db.optimized_parameters.find_one.return_value = {
        "parameters": {
            "strategy_type": "combined",
            "channel_strategy": {
                "channel_period": 20,
                "channel_deviation": 2.0
            },
            "dca_strategy": {
                "volatility_period": 20,
                "volatility_threshold": 1.0
            }
        }
    }
    mock_service.save_price_snapshot = AsyncMock()
    mock_service.get_portfolio = AsyncMock(return_value={
        "BTC": {"quantity": 1.0, "avg_price": 50000.0}
    })
    mock_service.update_portfolio = AsyncMock()
    mock_service.save_order = AsyncMock()
    
    # Create proper price data structure
    from datetime import datetime, timedelta
    price_data_points = []
    base_time = datetime.now()
    
    for i in range(10):
        time = base_time + timedelta(hours=i)
        mock_point = Mock(spec=PriceDataPoint)
        mock_point.datetime = time.strftime("%Y-%m-%d %H:%M:%S+00:00")
        mock_point.price = 100.0 + i * 0.1
        mock_point.open = 100.0 + i * 0.05
        mock_point.high = 100.0 + i * 0.15
        mock_point.low = 100.0 + i * 0.02
        mock_point.volume = 1000.0 + i * 10
        mock_point.price_change = 0.1
        price_data_points.append(mock_point)
    
    mock_snapshot = Mock()
    mock_snapshot.price_data = price_data_points
    mock_service.get_latest_snapshot = AsyncMock(return_value=mock_snapshot)
    
    return mock_service

@pytest.fixture
def mock_email_service():
    """Mock email service"""
    mock_service = Mock()
    mock_service.send_alert = AsyncMock()
    return mock_service

@pytest.fixture
def mock_config():
    """Mock configuration for all agents"""
    return {
        "data_fetching": {
            "symbols": {
                "BTC": "BTC-USD",
                "ETH": "ETH-USD",
                "SOL": "SOL-USD"
            },
            "interval": 60
        },
        "trading_teams": [
            {
                "name": "BTC",
                "status": True,
                "agents": [
                    {"name": "Data", "role": "data_collection"},
                    {"name": "Strategy", "role": "strategy_analysis"},
                    {"name": "Risk", "role": "risk_assessment"},
                    {"name": "Order", "role": "order_execution"}
                ]
            },
            {
                "name": "ETH",
                "status": True,
                "agents": [
                    {"name": "Data", "role": "data_collection"},
                    {"name": "Strategy", "role": "strategy_analysis"},
                    {"name": "Risk", "role": "risk_assessment"},
                    {"name": "Order", "role": "order_execution"}
                ]
            }
        ],
        "workflow": {
            "agent_sequence": ["Data", "Strategy", "Risk", "Order"]
        },
        "portfolio_manager": {
            "name": "Portfolio Manager",
            "role": "portfolio_management"
        },
        "strategy": {
            "bollinger_period": 20,
            "bollinger_std": 2.0,
            "dca_volatility_period": 20,
            "dca_volatility_threshold": 1.0
        },
        "risk": {
            "max_position_size": 0.1,
            "stop_loss_percentage": 0.05,
            "take_profit_percentage": 0.1
        },
        "order": {
            "max_order_size": 0.5,
            "min_order_size": 0.01
        }
    }

@pytest.fixture
def initial_agent_state():
    """Initial state for agent testing"""
    return {
        "messages": [],
        "current_agent": "",
        "message_history": [],
        "task_status": {},
        "parallel_tasks": [],
        "trading_teams": {"BTC": {"status": True}},
        "current_workflow": "data_collection"
    }

@pytest.fixture
def mock_ccxt_exchange():
    """Mock CCXT exchange"""
    mock_exchange = Mock()
    mock_exchange.fetch_ohlcv.return_value = [
        [1640995200000, 100.0, 105.0, 98.0, 102.0, 1000.0],
        [1640998800000, 102.0, 107.0, 100.0, 104.0, 1100.0],
        [1641002400000, 104.0, 109.0, 102.0, 106.0, 1200.0]
    ]
    return mock_exchange

@pytest.fixture
def mock_yfinance():
    """Mock yfinance"""
    mock_yf = Mock()
    mock_ticker = Mock()
    mock_ticker.history.return_value = pd.DataFrame({
        'Open': [100, 102, 104],
        'High': [105, 107, 109],
        'Low': [98, 100, 102],
        'Close': [102, 104, 106],
        'Volume': [1000, 1100, 1200]
    }, index=pd.date_range('2024-01-01', periods=3, freq='D'))
    mock_yf.Ticker.return_value = mock_ticker
    return mock_yf

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close() 

@pytest.fixture
def mock_services():
    """Mock database and email services"""
    mock_db = Mock()
    mock_email = Mock()
    
    # Mock database methods
    mock_db.save_price_snapshot.return_value = "mock_snapshot_id"
    mock_db.get_latest_snapshot.return_value = Mock(
        price_data=[
            Mock(
                datetime="2024-01-15 10:00:00+00:00",
                price=50000.0,
                open=49500.0,
                high=51000.0,
                low=49000.0,
                volume=1000.0,
                price_change=1.0
            )
        ]
    )
    mock_db.save_order.return_value = "mock_order_id"
    mock_db.update_portfolio.return_value = True
    mock_db.get_optimized_parameters.return_value = {
        "strategy_type": "combined",
        "channel_strategy": {"channel_period": 20, "channel_deviation": 2.0},
        "dca_strategy": {"volatility_period": 20, "volatility_threshold": 1.0}
    }
    mock_db.close.return_value = None
    
    # Mock email service methods
    mock_email.send_alert.return_value = True
    mock_email.send_report.return_value = True
    
    return mock_db, mock_email

@pytest.fixture
def mock_price_data():
    """Mock price data for testing"""
    return [
        Mock(
            datetime="2024-01-15 10:00:00+00:00",
            price=50000.0,
            open=49500.0,
            high=51000.0,
            low=49000.0,
            volume=1000.0,
            price_change=1.0
        ),
        Mock(
            datetime="2024-01-15 10:01:00+00:00",
            price=50100.0,
            open=50000.0,
            high=50200.0,
            low=49900.0,
            volume=1200.0,
            price_change=0.2
        )
    ]

@pytest.fixture
def mock_market_data():
    """Mock market data DataFrame"""
    return pd.DataFrame({
        'Open': [49500.0, 50000.0],
        'High': [51000.0, 50200.0],
        'Low': [49000.0, 49900.0],
        'Close': [50000.0, 50100.0],
        'Volume': [1000.0, 1200.0]
    }, index=pd.to_datetime(['2024-01-15 10:00:00', '2024-01-15 10:01:00']))

@pytest.fixture
def mock_agent_state():
    """Mock agent state for testing"""
    return {
        "messages": [],
        "current_agent": "Data",
        "message_history": [],
        "task_status": {},
        "parallel_tasks": [],
        "trading_teams": {"BTC": {"status": True}},
        "current_workflow": "data_collection"
    }

@pytest.fixture
def mock_yfinance():
    """Mock yfinance module"""
    mock_ticker = Mock()
    mock_ticker.history.return_value = pd.DataFrame({
        'Open': [49500.0],
        'High': [51000.0],
        'Low': [49000.0],
        'Close': [50000.0],
        'Volume': [1000.0]
    }, index=pd.to_datetime(['2024-01-15 10:00:00']))
    
    with patch('src.agents.multi_agent_system.yf') as mock_yf:
        mock_yf.Ticker.return_value = mock_ticker
        yield mock_yf

@pytest.fixture
def mock_ccxt():
    """Mock ccxt module"""
    mock_exchange = Mock()
    mock_exchange.fetch_ohlcv.return_value = [
        [1642233600000, 50000.0, 51000.0, 49000.0, 50000.0, 1000.0],
        [1642233660000, 50000.0, 50200.0, 49900.0, 50100.0, 1200.0]
    ]
    
    with patch('src.agents.multi_agent_system.ccxt') as mock_ccxt_module:
        mock_ccxt_module.binance.return_value = mock_exchange
        yield mock_ccxt_module

@pytest.fixture
def mock_event_loop():
    """Mock event loop for async tests"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close() 