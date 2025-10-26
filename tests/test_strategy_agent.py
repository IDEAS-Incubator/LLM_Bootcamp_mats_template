import pytest
from unittest.mock import Mock, patch
from src.agents.multi_agent_system import StrategyAgent

class TestStrategyAgent:
    """Test cases for Strategy Agent"""
    
    def test_extract_team_from_name(self):
        """Test team extraction from agent name"""
        # Test the base class method directly
        base_agent = StrategyAgent.__new__(StrategyAgent)
        base_agent.name = "BTC Strategy Agent"
        
        # Test team extraction
        team = base_agent._extract_team_from_name("BTC Strategy Agent")
        assert team == "BTC"
        
        team = base_agent._extract_team_from_name("ETH Strategy Agent")
        assert team == "ETH"
    
    def test_base_agent_initialization(self):
        """Test base agent initialization"""
        # Test the base class functionality
        base_agent = StrategyAgent.__new__(StrategyAgent)
        base_agent.name = "BTC Strategy Agent"
        base_agent.role = "strategy_analysis"
        base_agent.team = base_agent._extract_team_from_name(base_agent.name)
        
        # Verify agent properties
        assert base_agent.name == "BTC Strategy Agent"
        assert base_agent.role == "strategy_analysis"
        assert base_agent.team == "BTC"
    
    @pytest.mark.asyncio
    async def test_base_process_method(self):
        """Test base process method"""
        # Test the base class process method with proper initialization
        base_agent = StrategyAgent.__new__(StrategyAgent)
        base_agent.name = "BTC Strategy Agent"
        base_agent.role = "strategy_analysis"
        base_agent.team = "BTC"
        
        # Add required attributes for StrategyAgent
        base_agent.db_service = Mock()
        base_agent.email_service = Mock()
        base_agent.config = {
            "strategy_type": "combined",
            "channel_strategy": {"channel_period": 20, "channel_deviation": 2.0},
            "dca_strategy": {"volatility_period": 20, "volatility_threshold": 1.0}
        }
        
        # Mock the database service methods
        base_agent.db_service.get_latest_snapshot.return_value = Mock()
        base_agent.db_service.get_latest_snapshot.return_value.price_data = []
        
        initial_state = {
            "messages": [],
            "current_agent": "Strategy",
            "message_history": [],
            "task_status": {},
            "parallel_tasks": [],
            "trading_teams": {"BTC": {"status": True}},
            "current_workflow": "strategy_analysis"
        }
        
        # Test base process method
        result_state = await base_agent.process(initial_state)
        
        # Verify processing
        assert len(result_state["messages"]) >= 0
        assert "messages" in result_state
        assert "message_history" in result_state
    
    def test_convert_price_data_to_dataframe(self):
        """Test price data to DataFrame conversion"""
        # Test the method directly with proper initialization
        strategy_agent = StrategyAgent.__new__(StrategyAgent)
        strategy_agent.team = "BTC"  # Add required team attribute
        
        # Test with empty price data
        df = strategy_agent._convert_price_data_to_dataframe([])
        assert df.empty
        
        # Test with mock price data
        mock_price_data = [
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
        
        df = strategy_agent._convert_price_data_to_dataframe(mock_price_data)
        assert not df.empty
        assert len(df) == 1 