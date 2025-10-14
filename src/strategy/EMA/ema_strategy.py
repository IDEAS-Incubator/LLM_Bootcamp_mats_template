#!/usr/bin/env python3
"""
Exponential Moving Average Trading Strategy
Contains technical indicators, trading logic, and strategy execution
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
EMA_FAST = 8
EMA_SLOW = 21
EMA_TREND = 55
EMA_CROSS_THRESHOLD = 0.0001

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
    """Represents a single trade in the multi-timeframe EMA strategy"""
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    side: str  # 'long' or 'short'
    pnl: Optional[float]
    pnl_pct: Optional[float]
    bars_held: Optional[int]
    exit_reason: Optional[str]
    ema_fast_at_entry: float
    ema_slow_at_entry: float
    ema_trend_at_entry: float
    momentum_at_entry: float
    volume_at_entry: float
    trade_id: int
    timeframe: str
    period_days: int

# =============================================================================
# MULTI-TIMEFRAME INDICATORS
# =============================================================================

def add_multi_timeframe_ema(df: pd.DataFrame, fast: int = 8, slow: int = 21, trend: int = 55) -> pd.DataFrame:
    """Exponential Moving Average indicators for multiple timeframes"""
    df = df.copy()
    
    # Calculate different EMAs
    df['ema_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
    df['ema_trend'] = df['close'].ewm(span=trend, adjust=False).mean()
    
    # EMA crossovers
    df['ema_cross_up'] = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
    df['ema_cross_down'] = (df['ema_fast'] < df['ema_slow']) & (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1))
    
    # Price position relative to EMAs
    df['price_above_ema_fast'] = df['close'] > df['ema_fast']
    df['price_above_ema_slow'] = df['close'] > df['ema_slow']
    df['price_above_ema_trend'] = df['close'] > df['ema_trend']
    
    # EMA alignment (golden cross, death cross)
    df['golden_cross'] = (df['ema_fast'] > df['ema_slow']) & (df['ema_slow'] > df['ema_trend'])
    df['death_cross'] = (df['ema_fast'] < df['ema_slow']) & (df['ema_slow'] < df['ema_trend'])
    
    # EMA slope (trend strength)
    df['ema_fast_slope'] = df['ema_fast'].diff(3)
    df['ema_slow_slope'] = df['ema_slow'].diff(5)
    df['ema_trend_slope'] = df['ema_trend'].diff(10)
    
    # EMA strength
    df['ema_strength'] = abs(df['ema_fast'] - df['ema_slow']) / df['ema_slow'] > EMA_CROSS_THRESHOLD
    
    # EMA ribbon (multiple EMAs)
    df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema_13'] = df['close'].ewm(span=13, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_34'] = df['close'].ewm(span=34, adjust=False).mean()
    df['ema_55'] = df['close'].ewm(span=55, adjust=False).mean()
    
    # EMA ribbon alignment
    df['ema_ribbon_bull'] = (df['ema_8'] > df['ema_13']) & (df['ema_13'] > df['ema_21']) & (df['ema_21'] > df['ema_34']) & (df['ema_34'] > df['ema_55'])
    df['ema_ribbon_bear'] = (df['ema_8'] < df['ema_13']) & (df['ema_13'] < df['ema_21']) & (df['ema_21'] < df['ema_34']) & (df['ema_34'] < df['ema_55'])
    
    # EMA momentum
    df['ema_momentum'] = df['ema_fast'].diff(2)
    df['ema_momentum_strong'] = abs(df['ema_momentum']) > EMA_CROSS_THRESHOLD
    
    return df

def add_multi_timeframe_momentum(df: pd.DataFrame, period: int = 3) -> pd.DataFrame:
    """Momentum indicators for multiple timeframes"""
    df = df.copy()
    
    # Price momentum
    df['price_change'] = df['close'].pct_change(period)
    df['momentum_strength'] = df['price_change'].rolling(window=period).mean()
    df['momentum_confirmation'] = abs(df['momentum_strength']) > MIN_MOMENTUM_STRENGTH
    
    # Multiple momentum signals
    df['momentum_up'] = df['momentum_strength'] > MIN_MOMENTUM_STRENGTH
    df['momentum_down'] = df['momentum_strength'] < -MIN_MOMENTUM_STRENGTH
    
    # RSI for additional confirmation
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # RSI extremes
    df['rsi_oversold'] = df['rsi'] < 30
    df['rsi_overbought'] = df['rsi'] > 70
    
    # Stochastic oscillator
    df['stoch_k'] = ((df['close'] - df['low'].rolling(window=14).min()) / 
                     (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min())) * 100
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    df['stoch_oversold'] = df['stoch_k'] < 20
    df['stoch_overbought'] = df['stoch_k'] > 80
    
    return df

def add_multi_timeframe_volume(df: pd.DataFrame, period: int = 8) -> pd.DataFrame:
    """Volume analysis for multiple timeframes"""
    df = df.copy()
    
    df['volume_sma'] = df['volume'].rolling(window=period).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    df['volume_confirmation'] = df['volume_ratio'] > MIN_VOLUME_MULTIPLIER
    
    # Volume trend
    df['volume_trend'] = df['volume'].rolling(window=period).mean().diff(period)
    df['volume_increasing'] = df['volume_trend'] > 0
    
    # Volume weighted average price (VWAP)
    df['vwap'] = (df['close'] * df['volume']).rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
    df['price_above_vwap'] = df['close'] > df['vwap']
    
    # On-Balance Volume (OBV)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_sma'] = df['obv'].rolling(window=period).mean()
    df['obv_trend'] = df['obv'] > df['obv_sma']
    
    # Money Flow Index (MFI)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
    
    mfi_ratio = positive_flow / negative_flow
    df['mfi'] = 100 - (100 / (1 + mfi_ratio))
    df['mfi_oversold'] = df['mfi'] < 20
    df['mfi_overbought'] = df['mfi'] > 80
    
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
    
    # Fibonacci retracement levels
    df['fib_38'] = df['recent_high'] - 0.382 * (df['recent_high'] - df['recent_low'])
    df['fib_50'] = df['recent_high'] - 0.500 * (df['recent_high'] - df['recent_low'])
    df['fib_61'] = df['recent_high'] - 0.618 * (df['recent_high'] - df['recent_low'])
    
    # Price near Fibonacci levels
    df['near_fib_38'] = abs(df['close'] - df['fib_38']) / df['close'] < 0.01
    df['near_fib_50'] = abs(df['close'] - df['fib_50']) / df['close'] < 0.01
    df['near_fib_61'] = abs(df['close'] - df['fib_61']) / df['close'] < 0.01
    
    # Pivot points
    df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
    df['r1'] = 2 * df['pivot'] - df['low']
    df['s1'] = 2 * df['pivot'] - df['high']
    df['r2'] = df['pivot'] + (df['high'] - df['low'])
    df['s2'] = df['pivot'] - (df['high'] - df['low'])
    
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
    
    # Doji patterns
    df['doji'] = abs(df['close'] - df['open']) <= (df['high'] - df['low']) * 0.1
    
    # Hammer and shooting star
    df['hammer'] = (df['close'] > df['open']) & (df['low'] < df['close'] - 3 * (df['close'] - df['open']))
    df['shooting_star'] = (df['close'] < df['open']) & (df['high'] > df['open'] + 3 * (df['open'] - df['close']))
    
    # Inside bar
    df['inside_bar'] = (df['high'] <= df['high'].shift(1)) & (df['low'] >= df['low'].shift(1))
    
    # Outside bar
    df['outside_bar'] = (df['high'] > df['high'].shift(1)) & (df['low'] < df['low'].shift(1))
    
    return df

# =============================================================================
# MULTI-TIMEFRAME STRATEGY
# =============================================================================

def backtest_multi_timeframe_strategy(df: pd.DataFrame, 
                                    timeframe: str,
                                    period_days: int,
                                    position_allocation: float = BASE_POSITION_ALLOCATION) -> List[MultiTimeframeTrade]:
    """
    Multi-timeframe EMA trading strategy
    
    Entry Logic:
    - EMA crossovers (fast above/below slow)
    - Golden cross and death cross signals
    - EMA ribbon alignment
    - Price position relative to EMAs
    - Momentum confirmation
    - Volume confirmation
    - Pattern recognition
    - Support/resistance levels
    
    Exit Logic:
    - Timeframe-adapted take profit/stop loss
    - Time-based exits
    - EMA reversal exits
    """
    
    # Get timeframe-specific configuration
    config = TIMEFRAME_CONFIGS[timeframe]
    
    # Calculate timeframe-adapted parameters
    base_tp = 0.015 * config["tp_multiplier"]
    base_sl = 0.008 * config["sl_multiplier"]
    max_holding_time = config["holding_time_bars"]
    max_trades_per_day = config["max_trades_per_day"]
    min_trade_interval = config["min_trade_interval"]
    cooldown_after_loss = config["cooldown_after_loss"]
    
    required_cols = ["ema_fast", "ema_slow", "ema_trend", "ema_cross_up", "ema_cross_down",
                     "golden_cross", "death_cross", "ema_ribbon_bull", "ema_ribbon_bear",
                     "price_above_ema_fast", "price_above_ema_slow", "price_above_ema_trend",
                     "ema_strength", "momentum_confirmation", "volume_confirmation"]
    
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns: {required_cols}")
    
    trades = []
    active_trades = []
    last_trade_time = None
    consecutive_losses = 0
    trade_counter = 0
    
    for idx, (timestamp, row) in enumerate(df.iterrows()):
        # Skip first few rows for indicator calculation
        if idx < 60:
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
                
                # EMA reversal exit
                elif row['ema_cross_down'] and current_pnl > 0.005:
                    exit_reason = "ema_reversal"
                    exit_price = row['close']
                
                # Price below fast EMA exit
                elif not row['price_above_ema_fast'] and current_pnl > 0.003:
                    exit_reason = "ema_breakdown"
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
                
                # EMA reversal exit
                elif row['ema_cross_up'] and current_pnl > 0.005:
                    exit_reason = "ema_reversal"
                    exit_price = row['close']
                
                # Price above fast EMA exit
                elif row['price_above_ema_fast'] and current_pnl > 0.003:
                    exit_reason = "ema_breakout"
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
                # Strong EMA crossover with confirmation
                row['ema_cross_up'] and row['ema_strength'] and row['momentum_confirmation'] and row['volume_confirmation'],
                
                # Golden cross formation
                row['golden_cross'] and row['momentum_confirmation'] and row['volume_confirmation'],
                
                # EMA ribbon bullish alignment
                row['ema_ribbon_bull'] and row['momentum_confirmation'] and row['volume_confirmation'],
                
                # Price above all EMAs with momentum
                row['price_above_ema_fast'] and row['price_above_ema_slow'] and row['price_above_ema_trend'] and row['momentum_up'],
                
                # Bounce from support with EMA alignment
                row['near_fib_38'] and row['price_above_ema_fast'] and row['volume_confirmation'],
                
                # Hammer pattern with EMA support
                row['hammer'] and row['price_above_ema_slow'] and row['volume_confirmation'],
                
                # Stochastic oversold with EMA support
                row['stoch_oversold'] and row['price_above_ema_fast'] and row['volume_confirmation']
            ]
            
            # Multiple short entry conditions
            short_conditions = [
                # Strong EMA crossover with confirmation
                row['ema_cross_down'] and row['ema_strength'] and row['momentum_confirmation'] and row['volume_confirmation'],
                
                # Death cross formation
                row['death_cross'] and row['momentum_confirmation'] and row['volume_confirmation'],
                
                # EMA ribbon bearish alignment
                row['ema_ribbon_bear'] and row['momentum_confirmation'] and row['volume_confirmation'],
                
                # Price below all EMAs with momentum
                not row['price_above_ema_fast'] and not row['price_above_ema_slow'] and not row['price_above_ema_trend'] and row['momentum_down'],
                
                # Rejection from resistance with EMA alignment
                row['near_fib_61'] and not row['price_above_ema_fast'] and row['volume_confirmation'],
                
                # Shooting star pattern with EMA resistance
                row['shooting_star'] and not row['price_above_ema_slow'] and row['volume_confirmation'],
                
                # Stochastic overbought with EMA resistance
                row['stoch_overbought'] and not row['price_above_ema_fast'] and row['volume_confirmation']
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
                    ema_fast_at_entry=row['ema_fast'],
                    ema_slow_at_entry=row['ema_slow'],
                    ema_trend_at_entry=row['ema_trend'],
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
                    ema_fast_at_entry=row['ema_fast'],
                    ema_slow_at_entry=row['ema_slow'],
                    ema_trend_at_entry=row['ema_trend'],
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
    """Compute performance metrics for multi-timeframe EMA strategy"""
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
    total_pnl = sum(trade.pnl for trade in trades)
    total_return_pct = (total_pnl / start_equity) * 100
    
    # Win/loss analysis
    winning_trades = [t for t in trades if t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl < 0]
    
    win_rate = (len(winning_trades) / n_trades) * 100 if n_trades > 0 else 0
    avg_trade_ret = np.mean([t.pnl_pct for t in trades]) * 100
    
    # Risk metrics
    returns = [t.pnl_pct for t in trades]
    if len(returns) > 1:
        # Adjust Sharpe calculation for different timeframes
        if timeframe == "5m":
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 288)
        elif timeframe == "15m":
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 96)
        elif timeframe == "1h":
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24)
        elif timeframe == "4h":
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 6)
        elif timeframe == "1d":
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    else:
        sharpe = 0
    
    # Drawdown calculation
    equity_curve = [start_equity]
    for trade in trades:
        equity_curve.append(equity_curve[-1] + trade.pnl)
    
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
    avg_bars_held = np.mean([t.bars_held for t in trades]) if trades else 0
    
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
    """Run multi-timeframe EMA backtest for a specific scenario"""
    
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
    df = add_multi_timeframe_ema(df, EMA_FAST, EMA_SLOW, EMA_TREND)
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
    print("MULTI-TIMEFRAME EMA TRADING STRATEGY BACKTEST")
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
        print("MULTI-TIMEFRAME EMA PERFORMANCE SUMMARY")
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
        print("TOP PERFORMING EMA COMBINATIONS")
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
        print("EMA PERFORMANCE BY TIMEFRAME")
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
            print("🎯 COMBINATIONS ACHIEVING 0.5%+ DAILY RETURNS:")
            for idx, row in target_achievers.iterrows():
                print(f"   {row['Timeframe']:>4} {row['Days']:>3}d: {row['Daily_Return_Estimate']:>6.2f}% daily")
        else:
            print("❌ No combinations achieving 0.5%+ daily returns")
        
        if not close_to_target.empty:
            print("\n✅ COMBINATIONS CLOSE TO TARGET (0.1-0.5% daily):")
            for idx, row in close_to_target.iterrows():
                print(f"   {row['Timeframe']:>4} {row['Days']:>3}d: {row['Daily_Return_Estimate']:>6.2f}% daily")
        
        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"ema_multi_timeframe_performance_report_{SYMBOL.replace('/','_')}_{timestamp}.csv"
        df_results.to_csv(filename, index=False)
        print(f"\nComprehensive EMA performance report saved to: {filename}")
        
        # Best recommendation
        if not df_results.empty:
            best_combo = df_results.iloc[0]
            print(f"\n🏆 BEST EMA RECOMMENDATION:")
            print(f"   Timeframe: {best_combo['Timeframe']}")
            print(f"   Period: {best_combo['Days']} days")
            print(f"   Expected Daily Return: {best_combo['Daily_Return_Estimate']:.2f}%")
            print(f"   Total Return: {best_combo['Total_Return_pct']:.2f}%")
            print(f"   Win Rate: {best_combo['Win_Rate_pct']:.1f}%")
            print(f"   Trades: {best_combo['Trades']}")
            
            if best_combo['Daily_Return_Estimate'] >= 1.0:
                print("   🎯 TARGET ACHIEVED: 1%+ daily returns!")
            elif best_combo['Daily_Return_Estimate'] >= 0.5:
                print("   ✅ Good performance: 0.5%+ daily returns")
            else:
                print("   ⚠️  Below target: <0.5% daily returns")
        
    else:
        print("No results to analyze.")
