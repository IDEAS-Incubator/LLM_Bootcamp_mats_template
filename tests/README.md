# Multi-Agent Crypto Trading System - Test Suite

This directory contains comprehensive tests for the Multi-Agent Crypto Trading System.

## Test Structure

### Unit Tests
- **`test_data_agent.py`** - Tests for DataAgent functionality
- **`test_strategy_agent.py`** - Tests for StrategyAgent functionality  
- **`test_risk_agent.py`** - Tests for RiskAgent functionality
- **`test_order_agent.py`** - Tests for OrderAgent functionality
- **`test_portfolio_agent.py`** - Tests for PortfolioAgent functionality

### Integration Tests
- **`test_integration.py`** - End-to-end workflow tests
- **`test_cycle_tests.py`** - Multi-cycle execution tests

### Edge Case Tests
- **`test_edge_cases.py`** - Error handling and edge cases

## Test Categories

### 1. Unit Tests
These verify individual components:

#### DataAgent Tests
- `test_fetch_market_data_success` - Mock Binance response and verify PriceSnapshot creation
- `test_fetch_market_data_empty` - Simulate empty CCXT data
- `test_fetch_market_data_exception` - Force exception (e.g., bad symbol)

#### StrategyAgent Tests
- `test_load_optimized_parameters_found` - DB returns config
- `test_load_optimized_parameters_not_found` - DB returns empty
- `test_combined_strategy_signal_generation` - Use mock dataframe
- `test_insufficient_data` - Short DF passed in

#### RiskAgent Tests
- `test_risk_assessment_hold` - Signal: HOLD
- `test_risk_assessment_buy` - Signal: BUY
- `test_no_strategy_message` - No messages in state

#### OrderAgent Tests
- `test_order_processing_buy` - Buy signal → verify update logic
- `test_order_processing_sell` - Sell signal → correct position reduced
- `test_order_exceeds_position` - Sell more than current_position
- `test_no_order_signal` - HOLD signal

#### PortfolioAgent Tests
- `test_initialize_portfolios` - Triggers portfolio setup for all teams
- `test_process_executed_order` - Message parsed and update verified
- `test_missing_order_message` - No "TO: Portfolio" message

### 2. Integration Tests
These check multi-agent flow for one team:

- `test_single_team_workflow` - End-to-end with mock BTC data
- `test_team_manager_sequence` - Strategy→Risk→Order
- `test_agent_failover` - Strategy fails → Risk skips

### 3. Cycle Tests
- `test_full_cycle_execution` - Run full async cycle
- `test_multiple_cycles` - Simulate 3 cycles

### 4. Edge Cases
- `test_missing_symbol` - Symbol not supported
- `test_db_unreachable` - MongoDB down
- `test_invalid_ccxt_response` - Malformed OHLCV
- `test_rate_limit_retry` - Simulate ccxt.NetworkError

## Running Tests

### Prerequisites
```bash
pip install pytest pytest-asyncio
```

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Run Specific Test Categories
```bash
# Unit tests only
python -m pytest tests/ -m unit -v

# Integration tests only
python -m pytest tests/ -m integration -v

# Edge case tests only
python -m pytest tests/ -m edge -v

# Cycle tests only
python -m pytest tests/ -m cycle -v
```

### Using the Test Runner
```bash
# Run all tests
python tests/run_tests.py

# Run specific test type
python tests/run_tests.py --type unit
python tests/run_tests.py --type integration
python tests/run_tests.py --type edge
python tests/run_tests.py --type cycle

# Verbose output
python tests/run_tests.py --verbose
```

## Test Configuration

### pytest.ini
- Configures pytest settings
- Sets up async test support
- Defines test markers

### conftest.py
- Contains shared fixtures
- Mock services setup
- Sample data generation

## Mock Services

The test suite uses comprehensive mocking:

### Database Service Mock
- `get_optimized_parameters()` - Returns strategy configuration
- `save_price_snapshot()` - Async mock for data persistence
- `get_portfolio()` - Returns mock portfolio data
- `update_portfolio()` - Async mock for portfolio updates
- `save_order()` - Async mock for order persistence

### Email Service Mock
- `send_alert()` - Async mock for email notifications

### Exchange Mock
- `fetch_ohlcv()` - Returns mock OHLCV data
- Handles rate limiting and errors

## Test Data

### Sample Price Data
- 100 data points with realistic price movements
- Includes OHLCV data for strategy testing
- Timestamped data for time-series analysis

### Sample DataFrames
- Pandas DataFrames for strategy testing
- Multiple time periods for different scenarios
- Realistic price patterns for signal generation

## Coverage Areas

### ✅ Covered
- Agent initialization and configuration
- Message passing between agents
- Database interactions
- Error handling and recovery
- Strategy signal generation
- Order processing logic
- Portfolio management
- Multi-cycle execution
- Concurrent agent execution

### 🔄 Partially Covered
- Real-time data streaming
- Advanced strategy algorithms
- Performance optimization
- Load testing

### ❌ Not Covered
- Live trading execution
- Real exchange integration
- Advanced risk management
- Machine learning components

## Best Practices

### Test Organization
- Each agent has its own test file
- Integration tests separate from unit tests
- Edge cases grouped together
- Clear test naming conventions

### Mock Strategy
- Comprehensive service mocking
- Realistic test data
- Error scenario simulation
- Async operation testing

### Assertion Patterns
- Verify state changes
- Check message content
- Validate database calls
- Confirm error handling

## Continuous Integration

### GitHub Actions (Recommended)
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-asyncio
    - name: Run tests
      run: python -m pytest tests/ -v
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure src directory is in Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

2. **Async Test Failures**
   ```bash
   # Install pytest-asyncio
   pip install pytest-asyncio
   ```

3. **Mock Issues**
   ```bash
   # Clear pytest cache
   python -m pytest --cache-clear
   ```

### Debug Mode
```bash
# Run with detailed output
python -m pytest tests/ -v -s --tb=long
```

