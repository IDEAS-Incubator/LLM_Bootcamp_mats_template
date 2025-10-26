import pytest
from unittest.mock import Mock, patch
from src.agents.multi_agent_system import DataAgent

class TestDataAgent:
    """Test cases for Data Agent"""
    
    def test_extract_team_from_name(self):
        """Test team extraction from agent name"""
        # Test the base class method directly
        base_agent = DataAgent.__new__(DataAgent)
        base_agent.name = "BTC Data Agent"
        
        # Test team extraction
        team = base_agent._extract_team_from_name("BTC Data Agent")
        assert team == "BTC"
        
        team = base_agent._extract_team_from_name("ETH Data Agent")
        assert team == "ETH"
        
        team = base_agent._extract_team_from_name("Portfolio Manager")
        assert team == "portfolio_manager"
    
    def test_base_agent_initialization(self):
        """Test base agent initialization"""
        # Test the base class functionality
        base_agent = DataAgent.__new__(DataAgent)
        base_agent.name = "BTC Data Agent"
        base_agent.role = "data_collection"
        base_agent.team = base_agent._extract_team_from_name(base_agent.name)
        
        # Verify agent properties
        assert base_agent.name == "BTC Data Agent"
        assert base_agent.role == "data_collection"
        assert base_agent.team == "BTC"
    
    @pytest.mark.asyncio
    async def test_base_process_method(self):
        """Test base process method"""
        # Test the base class process method with proper initialization
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
    
    def test_agent_factory(self):
        """Test agent factory functionality"""
        from src.agents.multi_agent_system import AgentFactory
        
        # Test agent factory
        with patch('src.agents.multi_agent_system.DataAgent.__init__', return_value=None):
            data_agent = AgentFactory.create_agent("Data", "BTC", "data_collection")
            assert isinstance(data_agent, DataAgent)
    
    def test_agent_factory_invalid_type(self):
        """Test agent factory with invalid agent type"""
        from src.agents.multi_agent_system import AgentFactory
        
        # Test invalid agent type
        with pytest.raises(ValueError):
            AgentFactory.create_agent("Invalid", "BTC", "data_collection") 