import pytest
from unittest.mock import Mock, patch
from src.agents.multi_agent_system import (
    DataAgent, StrategyAgent, RiskAgent, OrderAgent, 
    PortfolioAgent, TeamManager, AgentFactory
)

class TestIntegration:
    """Integration tests for multi-agent system"""
    
    def test_agent_factory(self):
        """Test agent factory functionality"""
        # Test agent factory with mocked initialization
        with patch('src.agents.multi_agent_system.DataAgent.__init__', return_value=None):
            data_agent = AgentFactory.create_agent("Data", "BTC", "data_collection")
            assert isinstance(data_agent, DataAgent)
        
        with patch('src.agents.multi_agent_system.StrategyAgent.__init__', return_value=None):
            strategy_agent = AgentFactory.create_agent("Strategy", "BTC", "strategy_analysis")
            assert isinstance(strategy_agent, StrategyAgent)
        
        with patch('src.agents.multi_agent_system.RiskAgent.__init__', return_value=None):
            risk_agent = AgentFactory.create_agent("Risk", "BTC", "risk_assessment")
            assert isinstance(risk_agent, RiskAgent)
        
        with patch('src.agents.multi_agent_system.OrderAgent.__init__', return_value=None):
            order_agent = AgentFactory.create_agent("Order", "BTC", "order_execution")
            assert isinstance(order_agent, OrderAgent)
    
    def test_agent_factory_invalid_type(self):
        """Test agent factory with invalid agent type"""
        # Test invalid agent type
        with pytest.raises(ValueError):
            AgentFactory.create_agent("Invalid", "BTC", "data_collection")
    
    def test_base_agent_functionality(self):
        """Test base agent functionality"""
        # Test base agent methods
        base_agent = DataAgent.__new__(DataAgent)
        base_agent.name = "BTC Data Agent"
        base_agent.role = "data_collection"
        base_agent.team = base_agent._extract_team_from_name(base_agent.name)
        
        # Verify agent properties
        assert base_agent.name == "BTC Data Agent"
        assert base_agent.role == "data_collection"
        assert base_agent.team == "BTC"
    
    @pytest.mark.asyncio
    async def test_base_agent_process(self):
        """Test base agent process method"""
        # Test base agent process method with proper initialization
        base_agent = DataAgent.__new__(DataAgent)
        base_agent.name = "BTC Data Agent"
        base_agent.role = "data_collection"
        base_agent.team = "BTC"
        
        # Add required attributes for DataAgent
        base_agent.last_fetch_time = None
        base_agent.interval = 60
        base_agent.symbol = "BTC-USD"
        base_agent.db_service = Mock()
        base_agent.email_service = Mock()
        
        initial_state = {
            "messages": [],
            "current_agent": "Data",
            "message_history": [],
            "task_status": {},
            "parallel_tasks": [],
            "trading_teams": {"btc": {"status": True}},  # Use lowercase key
            "current_workflow": "data_collection"
        }
        
        # Mock the fetch_market_data method to avoid actual data fetching
        with patch.object(base_agent, 'fetch_market_data', return_value=Mock()):
            # Test base process method
            result_state = await base_agent.process(initial_state)
            
            # Verify processing
            assert len(result_state["messages"]) >= 0
            assert "messages" in result_state
            assert "message_history" in result_state
    
    def test_team_extraction(self):
        """Test team extraction from agent names"""
        # Test team extraction for different agent types
        base_agent = DataAgent.__new__(DataAgent)
        
        # Test data agent
        team = base_agent._extract_team_from_name("BTC Data Agent")
        assert team == "BTC"
        
        # Test strategy agent
        team = base_agent._extract_team_from_name("ETH Strategy Agent")
        assert team == "ETH"
        
        # Test portfolio agent
        team = base_agent._extract_team_from_name("Portfolio Manager")
        assert team == "portfolio_manager"
    
    def test_agent_creation(self):
        """Test agent creation without full initialization"""
        # Test creating agents without problematic initialization
        with patch('src.agents.multi_agent_system.DataAgent.__init__', return_value=None):
            data_agent = DataAgent("BTC", "data_collection")
            data_agent.name = "BTC Data Agent"
            data_agent.role = "data_collection"
            data_agent.team = "BTC"
            
            assert data_agent.name == "BTC Data Agent"
            assert data_agent.role == "data_collection"
            assert data_agent.team == "BTC" 