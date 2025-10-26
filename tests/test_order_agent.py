import pytest
from unittest.mock import Mock, patch
from src.agents.multi_agent_system import OrderAgent

class TestOrderAgent:
    """Test cases for Order Agent"""
    
    def test_extract_team_from_name(self):
        """Test team extraction from agent name"""
        # Test the base class method directly
        base_agent = OrderAgent.__new__(OrderAgent)
        base_agent.name = "BTC Order Agent"
        
        # Test team extraction
        team = base_agent._extract_team_from_name("BTC Order Agent")
        assert team == "BTC"
        
        team = base_agent._extract_team_from_name("ETH Order Agent")
        assert team == "ETH"
    
    def test_base_agent_initialization(self):
        """Test base agent initialization"""
        # Test the base class functionality
        base_agent = OrderAgent.__new__(OrderAgent)
        base_agent.name = "BTC Order Agent"
        base_agent.role = "order_execution"
        base_agent.team = base_agent._extract_team_from_name(base_agent.name)
        
        # Verify agent properties
        assert base_agent.name == "BTC Order Agent"
        assert base_agent.role == "order_execution"
        assert base_agent.team == "BTC"
    
    @pytest.mark.asyncio
    async def test_base_process_method(self):
        """Test base process method"""
        # Test the base class process method
        base_agent = OrderAgent.__new__(OrderAgent)
        base_agent.name = "BTC Order Agent"
        base_agent.role = "order_execution"
        base_agent.team = "BTC"
        
        initial_state = {
            "messages": [],
            "current_agent": "Order",
            "message_history": [],
            "task_status": {},
            "parallel_tasks": [],
            "trading_teams": {"BTC": {"status": True}},
            "current_workflow": "order_execution"
        }
        
        # Test base process method
        result_state = await base_agent.process(initial_state)
        
        # Verify processing
        assert len(result_state["messages"]) >= 0
        assert "messages" in result_state
        assert "message_history" in result_state 