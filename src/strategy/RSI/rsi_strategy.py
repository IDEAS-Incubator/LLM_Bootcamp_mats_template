#!/usr/bin/env python3
"""
MULTI-TIMEFRAME TRADING STRATEGY BACKTEST
Tests the same strategy across different time periods and timeframes
Goal: Find optimal timeframe + period combination for 1% daily returns
"""

import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional
import matplotlib.pyplot as plt

# =============================================================================
# MULTI-TIMEFRAME CONFIGURATION
# =============================================================================

# Trading Parameters
SYMBOL = "BTC/USDT"
START_EQUITY = 10000
BASE_POSITION_ALLOCATION = 0.50
FEE_RATE = 0.001
SLIPPAGE = 0.0003

# Strategy Parameters (will be adapted for each timeframe)
RSI_PERIOD = 5
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65
RSI_ENTRY_THRESHOLD = 40

MOMENTUM_PERIOD = 3
MIN_MOMENTUM_STRENGTH = 0.0005
PRICE_CHANGE_THRESHOLD = 0.001

VOLUME_PERIOD = 8
MIN_VOLUME_MULTIPLIER = 1.1

# Timeframe-specific parameters
TIMEFRAME_CONFIGS = {
    "5m": {
        "tp_multiplier": 1.0,
        "sl_multiplier": 1.0,
        "holding_time_bars": 8,
        "max_trades_per_day": 12,
        "min_trade_interval": 1,
        "cooldown_after_loss": 1
    },
    "15m": {
        "tp_multiplier": 1.5,
        "sl_multiplier": 1.2,
        "holding_time_bars": 6,
        "max_trades_per_day": 8,
        "min_trade_interval": 2,
        "cooldown_after_loss": 2
    },
    "1h": {
        "tp_multiplier": 2.0,
        "sl_multiplier": 1.5,
        "holding_time_bars": 4,
        "max_trades_per_day": 6,
        "min_trade_interval": 3,
        "cooldown_after_loss": 3
    },
    "4h": {
        "tp_multiplier": 3.0,
        "sl_multiplier": 2.0,
        "holding_time_bars": 3,
        "max_trades_per_day": 4,
        "min_trade_interval": 4,
        "cooldown_after_loss": 4
    },
    "1d": {
        "tp_multiplier": 5.0,
        "sl_multiplier": 3.0,
        "holding_time_bars": 2,
        "max_trades_per_day": 2,
        "min_trade_interval": 5,
        "cooldown_after_loss": 5
    }
}

# Test periods
TEST_PERIODS = [30, 60, 90, 200]

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MultiTimeframeTrade:
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    side: str
    pnl: Optional[float]
    pnl_pct: Optional[float]
    bars_held: Optional[int]
    exit_reason: Optional[str]
    rsi_at_entry: float
    momentum_at_entry: float
    volume_at_entry: float
    trade_id: int
    timeframe: str
    period_days: int

# =============================================================================
# MULTI-TIMEFRAME INDICATORS
# =============================================================================

def add_multi_timeframe_rsi(df: pd.DataFrame, period: int = 5, 
                           oversold: int = RSI_OVERSOLD, 
                           overbought: int = RSI_OVERBOUGHT,
                           entry_threshold: int = RSI_ENTRY_THRESHOLD) -> pd.DataFrame:
    """RSI for multiple timeframes with customizable parameters"""
    df = df.copy()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Entry signals with custom parameters
    df['rsi_oversold'] = df['rsi'] < oversold
    df['rsi_overbought'] = df['rsi'] > overbought
    df['rsi_momentum'] = df['rsi'].diff(2)
    
    # RSI crossovers with custom threshold
    df['rsi_cross_up'] = (df['rsi'] > entry_threshold) & (df['rsi'].shift(1) <= entry_threshold)
    df['rsi_cross_down'] = (df['rsi'] < (100 - entry_threshold)) & (df['rsi'].shift(1) >= (100 - entry_threshold))
    
    return df

def add_multi_timeframe_momentum(df: pd.DataFrame, period: int = 3, 
                                min_strength: float = MIN_MOMENTUM_STRENGTH) -> pd.DataFrame:
    """Momentum indicators for multiple timeframes with customizable parameters"""
    df = df.copy()
    
    # Price momentum
    df['price_change'] = df['close'].pct_change(period)
    df['momentum_strength'] = df['price_change'].rolling(window=period).mean()
    df['momentum_confirmation'] = abs(df['momentum_strength']) > min_strength
    
    # Multiple momentum signals
    df['momentum_up'] = df['momentum_strength'] > min_strength
    df['momentum_down'] = df['momentum_strength'] < -min_strength
    
    return df

def add_multi_timeframe_volume(df: pd.DataFrame, period: int = 8, 
                              min_multiplier: float = MIN_VOLUME_MULTIPLIER) -> pd.DataFrame:
    """Volume analysis for multiple timeframes with customizable parameters"""
    df = df.copy()
    
    df['volume_sma'] = df['volume'].rolling(window=period).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    df['volume_confirmation'] = df['volume_ratio'] > min_multiplier
    
    return df

def add_multi_timeframe_support_resistance(df: pd.DataFrame, period: int = 15) -> pd.DataFrame:
    """Support/resistance for multiple timeframes"""
    df = df.copy()
    
    # Rolling highs and lows
    df['recent_high'] = df['high'].rolling(window=period).max()
    df['recent_low'] = df['low'].rolling(window=period).min()
    
    # Price position relative to range
    df['price_position'] = (df['close'] - df['recent_low']) / (df['recent_high'] - df['recent_low'])
    
    # Entry zones
    df['oversold_zone'] = df['price_position'] < 0.25
    df['overbought_zone'] = df['price_position'] > 0.75
    
    return df

def add_price_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Price pattern recognition"""
    df = df.copy()
    
    # Pin bar patterns
    df['pin_bar_up'] = (df['close'] > df['open']) & (df['low'] < df['close'] - 2 * (df['close'] - df['open']))
    df['pin_bar_down'] = (df['close'] < df['open']) & (df['high'] > df['close'] + 2 * (df['open'] - df['close']))
    
    # Engulfing patterns
    df['engulfing_bull'] = (df['close'] > df['open']) & (df['close'] > df['high'].shift(1)) & (df['open'] < df['low'].shift(1))
    df['engulfing_bear'] = (df['close'] < df['open']) & (df['close'] < df['low'].shift(1)) & (df['open'] > df['high'].shift(1))
    
    return df

# =============================================================================
# MULTI-TIMEFRAME STRATEGY
# =============================================================================

def backtest_multi_timeframe_strategy(df: pd.DataFrame, 
                                    timeframe: str,
                                    period_days: int,
                                    position_allocation: float = BASE_POSITION_ALLOCATION,
                                    custom_params: Dict = None) -> List[MultiTimeframeTrade]:
    """
    Multi-timeframe trading strategy
    
    Entry Logic:
    - RSI-based entries
    - Momentum confirmation
    - Volume confirmation
    - Pattern recognition
    
    Exit Logic:
    - Timeframe-adapted take profit/stop loss
    - Time-based exits
    - Trailing stops
    """
    
    # Get timeframe-specific configuration
    config = TIMEFRAME_CONFIGS[timeframe].copy()
    
    # Apply custom parameters if provided
    if custom_params:
        if 'base_tp_multiplier' in custom_params:
            config["tp_multiplier"] = custom_params['base_tp_multiplier']
        if 'base_sl_multiplier' in custom_params:
            config["sl_multiplier"] = custom_params['base_sl_multiplier']
        if 'max_trades_per_day' in custom_params:
            config["max_trades_per_day"] = custom_params['max_trades_per_day']
        if 'min_trade_interval' in custom_params:
            config["min_trade_interval"] = custom_params['min_trade_interval']
        if 'cooldown_after_loss' in custom_params:
            config["cooldown_after_loss"] = custom_params['cooldown_after_loss']
    
    # Calculate timeframe-adapted parameters
    base_tp = 0.015 * config["tp_multiplier"]
    base_sl = 0.008 * config["sl_multiplier"]
    max_holding_time = config["holding_time_bars"]
    max_trades_per_day = config["max_trades_per_day"]
    min_trade_interval = config["min_trade_interval"]
    cooldown_after_loss = config["cooldown_after_loss"]
    
    required_cols = ["rsi", "rsi_cross_up", "rsi_cross_down", "momentum_confirmation",
                     "volume_confirmation", "oversold_zone", "overbought_zone", "rsi_momentum",
                     "pin_bar_up", "pin_bar_down", "engulfing_bull", "engulfing_bear"]
    
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns: {required_cols}")
    
    trades = []
    active_trades = []
    trailing_stops = {}
    last_trade_time = None
    consecutive_losses = 0
    trade_counter = 0
    
    for idx, (timestamp, row) in enumerate(df.iterrows()):
        # Skip first few rows for indicator calculation
        if idx < 25:
            continue
            
        # Check for exit conditions on all active trades
        trades_to_remove = []
        for trade in active_trades:
            exit_reason = None
            exit_price = None
            
            # Calculate current PnL
            if trade.side == 'long':
                current_pnl = (row['close'] - trade.entry_price) / trade.entry_price
                trade.pnl_pct = current_pnl
                
                # Take profit
                if current_pnl >= base_tp:
                    exit_reason = "take_profit"
                    exit_price = trade.entry_price * (1 + base_tp)
                
                # Stop loss
                elif current_pnl <= -base_sl:
                    exit_reason = "stop_loss"
                    exit_price = trade.entry_price * (1 - base_sl)
                
                # Time-based exit
                elif trade.bars_held and trade.bars_held >= max_holding_time:
                    exit_reason = "time_exit"
                    exit_price = row['close']
                
                # RSI reversal exit
                elif row['rsi'] > RSI_OVERBOUGHT and current_pnl > 0.005:
                    exit_reason = "rsi_reversal"
                    exit_price = row['close']
            
            elif trade.side == 'short':
                current_pnl = (trade.entry_price - row['close']) / trade.entry_price
                trade.pnl_pct = current_pnl
                
                # Take profit
                if current_pnl >= base_tp:
                    exit_reason = "take_profit"
                    exit_price = trade.entry_price * (1 - base_tp)
                
                # Stop loss
                elif current_pnl <= -base_sl:
                    exit_reason = "stop_loss"
                    exit_price = trade.entry_price * (1 + base_sl)
                
                # Time-based exit
                elif trade.bars_held and trade.bars_held >= max_holding_time:
                    exit_reason = "time_exit"
                    exit_price = row['close']
                
                # RSI reversal exit
                elif row['rsi'] < RSI_OVERSOLD and current_pnl > 0.005:
                    exit_reason = "rsi_reversal"
                    exit_price = row['close']
            
            # Execute exit if conditions met
            if exit_reason:
                trade.exit_time = row.name
                trade.exit_price = exit_price
                trade.pnl = (exit_price - trade.entry_price) * trade.position_size
                if trade.side == 'short':
                    trade.pnl = -trade.pnl
                
                # Calculate bars held
                entry_idx = df.index.get_loc(trade.entry_time)
                trade.bars_held = idx - entry_idx
                trade.exit_reason = exit_reason
                
                trades.append(trade)
                trades_to_remove.append(trade)
                
                # Update consecutive losses
                if trade.pnl < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0
                
                # Clean up trailing stop
                if trade.trade_id in trailing_stops:
                    del trailing_stops[trade.trade_id]
        
        # Remove exited trades
        for trade in trades_to_remove:
            active_trades.remove(trade)
        
        # Check for new entry conditions
        if len(active_trades) < 2:  # Max 2 concurrent trades
            # Check trade interval and cooldown
            if last_trade_time is not None:
                # Calculate time since last trade based on timeframe
                if timeframe == "5m":
                    bars_since_last = (row.name - last_trade_time).total_seconds() / 300
                elif timeframe == "15m":
                    bars_since_last = (row.name - last_trade_time).total_seconds() / 900
                elif timeframe == "1h":
                    bars_since_last = (row.name - last_trade_time).total_seconds() / 3600
                elif timeframe == "4h":
                    bars_since_last = (row.name - last_trade_time).total_seconds() / 14400
                elif timeframe == "1d":
                    bars_since_last = (row.name - last_trade_time).total_seconds() / 86400
                
                if bars_since_last < min_trade_interval:
                    continue
                
                # Cooldown after consecutive losses
                if consecutive_losses >= 3 and bars_since_last < cooldown_after_loss:
                    continue
            
            # Multiple long entry conditions
            long_conditions = [
                row['rsi_cross_up'] and row['momentum_confirmation'] and row['volume_confirmation'],
                row['rsi_oversold'] and row['momentum_up'] and row['volume_confirmation'],
                row['pin_bar_up'] and row['volume_confirmation'],
                row['engulfing_bull'] and row['volume_confirmation'],
                row['oversold_zone'] and row['momentum_up'] and row['volume_confirmation']
            ]
            
            # Multiple short entry conditions
            short_conditions = [
                row['rsi_cross_down'] and row['momentum_confirmation'] and row['volume_confirmation'],
                row['rsi_overbought'] and row['momentum_down'] and row['volume_confirmation'],
                row['pin_bar_down'] and row['volume_confirmation'],
                row['engulfing_bear'] and row['volume_confirmation'],
                row['overbought_zone'] and row['momentum_down'] and row['volume_confirmation']
            ]
            
            # Check for long entry
            if any(long_conditions):
                trade_counter += 1
                new_trade = MultiTimeframeTrade(
                    entry_time=row.name,
                    exit_time=None,
                    entry_price=row['close'],
                    exit_price=None,
                    position_size=position_allocation * START_EQUITY / row['close'],
                    side='long',
                    pnl=None,
                    pnl_pct=None,
                    bars_held=None,
                    exit_reason=None,
                    rsi_at_entry=row['rsi'],
                    momentum_at_entry=row['momentum_strength'],
                    volume_at_entry=row['volume_ratio'],
                    trade_id=trade_counter,
                    timeframe=timeframe,
                    period_days=period_days
                )
                active_trades.append(new_trade)
                last_trade_time = row.name
            
            # Check for short entry
            elif any(short_conditions):
                trade_counter += 1
                new_trade = MultiTimeframeTrade(
                    entry_time=row.name,
                    exit_time=None,
                    entry_price=row['close'],
                    exit_price=None,
                    position_size=position_allocation * START_EQUITY / row['close'],
                    side='short',
                    pnl=None,
                    pnl_pct=None,
                    bars_held=None,
                    exit_reason=None,
                    rsi_at_entry=row['rsi'],
                    momentum_at_entry=row['momentum_strength'],
                    volume_at_entry=row['volume_ratio'],
                    trade_id=trade_counter,
                    timeframe=timeframe,
                    period_days=period_days
                )
                active_trades.append(new_trade)
                last_trade_time = row.name
    
    # Close any remaining open trades at the end
    for trade in active_trades:
        last_row = df.iloc[-1]
        trade.exit_time = last_row.name
        trade.exit_price = last_row['close']
        trade.pnl = (last_row['close'] - trade.entry_price) * trade.position_size
        if trade.side == 'short':
            trade.pnl = -trade.pnl
        
        entry_idx = df.index.get_loc(trade.entry_time)
        trade.bars_held = len(df) - entry_idx
        trade.exit_reason = "end_of_data"
        trades.append(trade)
    
    return trades

# =============================================================================
# MULTI-TIMEFRAME PERFORMANCE CALCULATION
# =============================================================================

def compute_multi_timeframe_metrics(trades: List[MultiTimeframeTrade], 
                                  start_equity: float = START_EQUITY,
                                  timeframe: str = "5m") -> Dict:
    """Compute performance metrics for multi-timeframe strategy"""
    if not trades:
        return {
            'Total_Return_pct': 0.0,
            'Trades': 0,
            'Win_Rate_pct': 0.0,
            'Avg_Trade_Ret_pct': 0.0,
            'Profit_Factor': 0.0,
            'Max_Drawdown_pct': 0.0,
            'Sharpe': 0.0,
            'Daily_Return_Estimate': 0.0,
            'Trades_Per_Day': 0.0,
            'Avg_Trade_Duration': 0.0,
            'Timeframe': timeframe
        }
    
    # Basic metrics
    n_trades = len(trades)
    total_pnl = sum(trade.pnl for trade in trades if trade.pnl is not None)
    total_return_pct = (total_pnl / start_equity) * 100
    
    # Win/loss analysis
    winning_trades = [t for t in trades if t.pnl is not None and t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl is not None and t.pnl < 0]
    
    win_rate = (len(winning_trades) / n_trades) * 100 if n_trades > 0 else 0
    
    # Safe calculation of average trade return
    valid_pnl_pcts = [t.pnl_pct for t in trades if t.pnl_pct is not None]
    avg_trade_ret = np.mean(valid_pnl_pcts) * 100 if valid_pnl_pcts else 0
    
    # Risk metrics
    valid_returns = [t.pnl_pct for t in trades if t.pnl_pct is not None]
    if len(valid_returns) > 1:
        # Adjust Sharpe calculation for different timeframes
        if timeframe == "5m":
            sharpe = np.mean(valid_returns) / np.std(valid_returns) * np.sqrt(252 * 288)
        elif timeframe == "15m":
            sharpe = np.mean(valid_returns) / np.std(valid_returns) * np.sqrt(252 * 96)
        elif timeframe == "1h":
            sharpe = np.mean(valid_returns) / np.std(valid_returns) * np.sqrt(252 * 24)
        elif timeframe == "4h":
            sharpe = np.mean(valid_returns) / np.std(valid_returns) * np.sqrt(252 * 6)
        elif timeframe == "1d":
            sharpe = np.mean(valid_returns) / np.std(valid_returns) * np.sqrt(252)
        else:
            sharpe = np.mean(valid_returns) / np.std(valid_returns) * np.sqrt(252)
    else:
        sharpe = 0
    
    # Drawdown calculation
    equity_curve = [start_equity]
    for trade in trades:
        if trade.pnl is not None:
            equity_curve.append(equity_curve[-1] + trade.pnl)
        else:
            equity_curve.append(equity_curve[-1])
    
    peak = start_equity
    max_drawdown = 0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak
        max_drawdown = max(max_drawdown, drawdown)
    
    max_drawdown_pct = max_drawdown * 100
    
    # Profit factor
    gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
    gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    # Trade frequency and duration
    valid_bars_held = [t.bars_held for t in trades if t.bars_held is not None]
    avg_bars_held = np.mean(valid_bars_held) if valid_bars_held else 0
    
    # Calculate timeframe-specific metrics
    if timeframe == "5m":
        bars_per_day = 288
        avg_trade_duration = avg_bars_held * 5  # minutes
    elif timeframe == "15m":
        bars_per_day = 96
        avg_trade_duration = avg_bars_held * 15  # minutes
    elif timeframe == "1h":
        bars_per_day = 24
        avg_trade_duration = avg_bars_held * 60  # minutes
    elif timeframe == "4h":
        bars_per_day = 6
        avg_trade_duration = avg_bars_held * 240  # minutes
    elif timeframe == "1d":
        bars_per_day = 1
        avg_trade_duration = avg_bars_held * 1440  # minutes
    else:
        bars_per_day = 288
        avg_trade_duration = avg_bars_held * 5
    
    # Estimate trades per day
    if len(trades) > 0:
        data_length_hours = (trades[-1].entry_time - trades[0].entry_time).total_seconds() / 3600
        trades_per_day = (n_trades / data_length_hours) * 24 if data_length_hours > 0 else 0
    else:
        trades_per_day = 0
    
    # Daily return estimate
    daily_return_estimate = (avg_trade_ret / 100) * trades_per_day * 100 if trades else 0
    
    return {
        'Total_Return_pct': total_return_pct,
        'Trades': n_trades,
        'Win_Rate_pct': win_rate,
        'Avg_Trade_Ret_pct': avg_trade_ret,
        'Profit_Factor': profit_factor,
        'Max_Drawdown_pct': max_drawdown_pct,
        'Sharpe': sharpe,
        'Daily_Return_Estimate': daily_return_estimate,
        'Trades_Per_Day': trades_per_day,
        'Avg_Trade_Duration': avg_trade_duration,
        'Timeframe': timeframe
    }

# =============================================================================
# MULTI-TIMEFRAME SCENARIO RUNNER
# =============================================================================

def run_multi_timeframe_scenario(timeframe: str, 
                               days: int, 
                               position_allocation: float = BASE_POSITION_ALLOCATION) -> Dict:
    """Run multi-timeframe backtest for a specific scenario"""
    
    # Fetch data
    exchange = ccxt.binance()
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    ohlcv = exchange.fetch_ohlcv(
        symbol=SYMBOL,
        timeframe=timeframe,
        since=int(start_time.timestamp() * 1000),
        limit=10000
    )
    
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Add all indicators
    df = add_multi_timeframe_rsi(df, RSI_PERIOD)
    df = add_multi_timeframe_momentum(df, MOMENTUM_PERIOD)
    df = add_multi_timeframe_volume(df, VOLUME_PERIOD)
    df = add_multi_timeframe_support_resistance(df, 15)
    df = add_price_patterns(df)
    
    # Run multi-timeframe strategy
    trades = backtest_multi_timeframe_strategy(df, timeframe, days, position_allocation)
    
    # Calculate metrics
    metrics = compute_multi_timeframe_metrics(trades, START_EQUITY, timeframe)
    
    return {
        'timeframe': timeframe,
        'days': days,
        'trades': trades,
        'metrics': metrics,
        'data': df
    }

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MULTI-TIMEFRAME TRADING STRATEGY BACKTEST")
    print("=" * 80)
    print(f"Symbol: {SYMBOL}")
    print(f"Start Equity: ${START_EQUITY:,.2f}")
    print()
    
    print("TESTING MATRIX:")
    print("Time Periods: 30, 60, 90, 200 days")
    print("Timeframes: 5m, 15m, 1h, 4h, 1d")
    print(f"Total Combinations: {len(TEST_PERIODS) * len(TIMEFRAME_CONFIGS)}")
    print()
    
    # Test all combinations
    all_results = []
    
    for timeframe in TIMEFRAME_CONFIGS.keys():
        print(f"Testing {timeframe} timeframe...")
        for days in TEST_PERIODS:
            print(f"  Testing {days} days...")
            try:
                result = run_multi_timeframe_scenario(timeframe, days)
                all_results.append(result)
                
                metrics = result['metrics']
                print(f"    {timeframe:>4} {days:>3}d: {metrics['Total_Return_pct']:>7.2f}% return, "
                      f"{metrics['Trades']:>3} trades, "
                      f"{metrics['Win_Rate_pct']:>5.1f}% win rate, "
                      f"{metrics['Daily_Return_Estimate']:>5.2f}% daily est.")
                
            except Exception as e:
                print(f"    Error testing {timeframe} {days} days: {e}")
                continue
    
    if all_results:
        print("\n" + "=" * 80)
        print("MULTI-TIMEFRAME PERFORMANCE SUMMARY")
        print("=" * 80)
        
        # Create comprehensive results DataFrame
        df_results = pd.DataFrame([
            {
                'Timeframe': r['timeframe'],
                'Days': r['days'],
                'Total_Return_pct': r['metrics']['Total_Return_pct'],
                'Trades': r['metrics']['Trades'],
                'Win_Rate_pct': r['metrics']['Win_Rate_pct'],
                'Avg_Trade_Ret_pct': r['metrics']['Avg_Trade_Ret_pct'],
                'Profit_Factor': r['metrics']['Profit_Factor'],
                'Max_Drawdown_pct': r['metrics']['Max_Drawdown_pct'],
                'Sharpe': r['metrics']['Sharpe'],
                'Daily_Return_Estimate': r['metrics']['Daily_Return_Estimate'],
                'Trades_Per_Day': r['metrics']['Trades_Per_Day'],
                'Avg_Trade_Duration': r['metrics']['Avg_Trade_Duration']
            }
            for r in all_results
        ])
        
        # Sort by daily return estimate (descending)
        df_results = df_results.sort_values('Daily_Return_Estimate', ascending=False)
        
        print(df_results.to_string(index=False))
        print()
        
        # Find best combinations
        print("=" * 80)
        print("TOP PERFORMING COMBINATIONS")
        print("=" * 80)
        
        # Top 5 by daily return estimate
        top_5 = df_results.head(5)
        for idx, row in top_5.iterrows():
            print(f"{idx+1}. {row['Timeframe']:>4} {row['Days']:>3}d: "
                  f"{row['Daily_Return_Estimate']:>6.2f}% daily, "
                  f"{row['Total_Return_pct']:>6.2f}% total, "
                  f"{row['Trades']:>2} trades, "
                  f"{row['Win_Rate_pct']:>5.1f}% win rate")
        
        print()
        
        # Summary statistics by timeframe
        print("=" * 80)
        print("PERFORMANCE BY TIMEFRAME")
        print("=" * 80)
        
        for timeframe in TIMEFRAME_CONFIGS.keys():
            tf_results = df_results[df_results['Timeframe'] == timeframe]
            if not tf_results.empty:
                avg_return = tf_results['Total_Return_pct'].mean()
                avg_daily = tf_results['Daily_Return_Estimate'].mean()
                total_trades = tf_results['Trades'].sum()
                avg_win_rate = tf_results['Win_Rate_pct'].mean()
                
                print(f"{timeframe:>4}: {avg_return:>7.2f}% avg return, "
                      f"{avg_daily:>6.2f}% avg daily, "
                      f"{total_trades:>3} total trades, "
                      f"{avg_win_rate:>5.1f}% avg win rate")
        
        print()
        
        # Target achievement analysis
        print("=" * 80)
        print("TARGET ACHIEVEMENT: 1% DAILY RETURNS")
        print("=" * 80)
        
        # Find combinations that achieve or come close to 1% daily
        target_achievers = df_results[df_results['Daily_Return_Estimate'] >= 0.5]
        close_to_target = df_results[(df_results['Daily_Return_Estimate'] >= 0.1) & 
                                   (df_results['Daily_Return_Estimate'] < 0.5)]
        
        if not target_achievers.empty:
            print(" COMBINATIONS ACHIEVING 0.5%+ DAILY RETURNS:")
            for idx, row in target_achievers.iterrows():
                print(f"   {row['Timeframe']:>4} {row['Days']:>3}d: {row['Daily_Return_Estimate']:>6.2f}% daily")
        else:
            print(" No combinations achieving 0.5%+ daily returns")
        
        if not close_to_target.empty:
            print("\nCOMBINATIONS CLOSE TO TARGET (0.1-0.5% daily):")
            for idx, row in close_to_target.iterrows():
                print(f"   {row['Timeframe']:>4} {row['Days']:>3}d: {row['Daily_Return_Estimate']:>6.2f}% daily")
        
        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"multi_timeframe_performance_report_{SYMBOL.replace('/','_')}_{timestamp}.csv"
        df_results.to_csv(filename, index=False)
        print(f"\nComprehensive performance report saved to: {filename}")
        
        # Best recommendation
        if not df_results.empty:
            best_combo = df_results.iloc[0]
            print(f"\n BEST RECOMMENDATION:")
            print(f"   Timeframe: {best_combo['Timeframe']}")
            print(f"   Period: {best_combo['Days']} days")
            print(f"   Expected Daily Return: {best_combo['Daily_Return_Estimate']:.2f}%")
            print(f"   Total Return: {best_combo['Total_Return_pct']:.2f}%")
            print(f"   Win Rate: {best_combo['Win_Rate_pct']:.1f}%")
            print(f"   Trades: {best_combo['Trades']}")
            
            if best_combo['Daily_Return_Estimate'] >= 1.0:
                print("   TARGET ACHIEVED: 1%+ daily returns!")
            elif best_combo['Daily_Return_Estimate'] >= 0.5:
                print("   Good performance: 0.5%+ daily returns")
            else:
                print("   Below target: <0.5% daily returns")
        
    else:
        print("No results to analyze.")
