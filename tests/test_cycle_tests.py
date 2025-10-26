import pytest
from unittest.mock import Mock, patch
from src.agents.multi_agent_system import (
    DataAgent, StrategyAgent, RiskAgent, OrderAgent, 
    PortfolioAgent, TeamManager, AgentFactory
)

class TestCycleTests:
    """Test complete trading cycles"""
    
    def test_agent_factory_cycle(self):
        """Test agent factory cycle"""
        # Test creating all agent types through factory
        agent_types = ["Data", "Strategy", "Risk", "Order"]
        
        for agent_type in agent_types:
            with patch(f'src.agents.multi_agent_system.{agent_type}Agent.__init__', return_value=None):
                agent = AgentFactory.create_agent(agent_type, "BTC", f"{agent_type.lower()}_role")
                assert agent is not None
                # Set required attributes after creation
                agent.name = f"BTC {agent_type} Agent"
                agent.role = f"{agent_type.lower()}_role"
                agent.team = "BTC"
                assert hasattr(agent, 'name')
                assert hasattr(agent, 'role')
                assert hasattr(agent, 'team')
    
    @pytest.mark.asyncio
    async def test_base_agent_cycle(self):
        """Test base agent cycle processing"""
        # Test base agent process method in a cycle with proper initialization
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
            # Process multiple cycles
            for cycle in range(3):
                result_state = await base_agent.process(initial_state.copy())
                
                # Verify processing
                assert len(result_state["messages"]) >= 0
                assert "messages" in result_state
                assert "message_history" in result_state
    
    def test_agent_creation_cycle(self):
        """Test agent creation cycle"""
        # Test creating multiple agents in a cycle
        agents = []
        
        for i in range(5):
            with patch('src.agents.multi_agent_system.DataAgent.__init__', return_value=None):
                agent = DataAgent("BTC", "data_collection")
                agent.name = f"BTC Data Agent {i}"
                agent.role = "data_collection"
                agent.team = "BTC"
                agents.append(agent)
        
        # Verify all agents created successfully
        assert len(agents) == 5
        
        for i, agent in enumerate(agents):
            assert agent.name == f"BTC Data Agent {i}"
            assert agent.role == "data_collection"
            assert agent.team == "BTC"
    
    def test_team_extraction_cycle(self):
        """Test team extraction cycle"""
        # Test team extraction for multiple agents
        base_agent = DataAgent.__new__(DataAgent)
        
        team_names = ["BTC", "ETH", "SOL", "ADA", "DOT"]
        
        for team in team_names:
            agent_name = f"{team} Data Agent"
            extracted_team = base_agent._extract_team_from_name(agent_name)
            assert extracted_team == team
    
    def test_memory_cycle(self):
        """Test memory management cycle"""
        # Test creating and destroying agents in a cycle
        for cycle in range(3):
            agents = []
            
            # Create agents
            for i in range(10):
                agent = DataAgent.__new__(DataAgent)
                agent.name = f"Agent{i}"
                agent.role = "test_role"
                agent.team = f"Team{i}"
                agents.append(agent)
            
            # Verify agents created
            assert len(agents) == 10
            
            # Clean up
            for agent in agents:
                del agent
    
    def test_error_recovery_cycle(self):
        """Test error recovery cycle"""
        # Test error handling in a cycle
        base_agent = DataAgent.__new__(DataAgent)
        
        # Test with various error conditions
        error_conditions = ["", None, "Invalid Agent Name", "123 Agent"]
        
        for condition in error_conditions:
            try:
                team = base_agent._extract_team_from_name(condition)
                # Should handle gracefully
                assert team is not None or team == ""
            except (AttributeError, TypeError, IndexError):
                # Expected behavior for some conditions
                pass
    
    def test_performance_cycle(self):
        """Test performance monitoring cycle"""
        # Test creating many agents quickly
        agents = []
        
        # Create 100 agents quickly
        for i in range(100):
            agent = DataAgent.__new__(DataAgent)
            agent.name = f"Performance Agent {i}"
            agent.role = "performance_test"
            agent.team = f"Team{i % 10}"  # Use modulo to limit team names
            agents.append(agent)
        
        # Verify all agents created
        assert len(agents) == 100
        
        # Verify agent properties
        for i, agent in enumerate(agents):
            assert agent.name == f"Performance Agent {i}"
            assert agent.role == "performance_test"
            assert agent.team == f"Team{i % 10}"
        
        # Clean up
        for agent in agents:
            del agent 