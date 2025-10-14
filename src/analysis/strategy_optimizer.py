import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product
import argparse
import logging
import time
import random
from src.services.database_service import MongoDBService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class StrategyAgent:
    def __init__(self, config):
        self.config = config
        self.strategy_type = config.get("strategy_type", "combined")

    def process_channel_signals(self, df):
        if self.strategy_type == "bollinger":
            return self._process_bollinger_signals(df)
        elif self.strategy_type == "dca":
            return self._process_dca_signals(df)
        elif self.strategy_type == "combined":
            return self._process_combined_signals(df)
        else:
            raise ValueError(f"Unknown strategy type: {self.strategy_type}")

    def _process_bollinger_signals(self, df):
        config = self.config["channel_strategy"]
        signals = []

        # Calculate Bollinger Bands
        df["SMA"] = df["Close"].rolling(window=config["channel_period"]).mean()
        df["STD"] = df["Close"].rolling(window=config["channel_period"]).std()
        df["Upper"] = df["SMA"] + config["channel_deviation"] * df["STD"]
        df["Lower"] = df["SMA"] - config["channel_deviation"] * df["STD"]

        # Calculate local minima and maxima with smaller window
        window = 3  # Reduced window for more signals

        for i in range(window + config["channel_period"], len(df) - window):
            # Get scalar values for comparison
            price = df["Close"].iloc[i].item()
            upper = df["Upper"].iloc[i].item()
            lower = df["Lower"].iloc[i].item()
            time = df.index[i]

            # Get price window and calculate min/max
            price_window = df["Close"].iloc[i - window : i + window + 1]
            min_price = price_window.min().item()
            max_price = price_window.max().item()

            # More lenient conditions for signals
            # Buy when price is near or below lower band
            if price <= lower * 1.01 and price <= min_price * 1.01:
                signals.append(
                    {
                        "timestamp": time,
                        "signal": "buy",
                        "price": price,
                        "strategy": "bollinger",
                    }
                )

            # Sell when price is near or above upper band
            elif price >= upper * 0.99 and price >= max_price * 0.99:
                signals.append(
                    {
                        "timestamp": time,
                        "signal": "sell",
                        "price": price,
                        "strategy": "bollinger",
                    }
                )

        return signals

    def _process_dca_signals(self, df):
        config = self.config["dca_strategy"]
        signals = []

        # Calculate percentage distance from SMA
        df["SMA"] = df["Close"].rolling(window=config["volatility_period"]).mean()
        df["Distance"] = ((df["Close"] - df["SMA"]) / df["SMA"]) * 100

        # Calculate volatility with shorter window
        df["Daily_Return"] = df["Close"].pct_change()
        df["Volatility"] = (
            df["Daily_Return"].rolling(window=config["volatility_period"]).std()
        )
        avg_volatility = df["Volatility"].mean().item()

        for i in range(config["volatility_period"], len(df)):
            # Get scalar values for comparison
            time = df.index[i]
            price = df["Close"].iloc[i].item()
            distance = df["Distance"].iloc[i].item()
            volatility = df["Volatility"].iloc[i].item()

            # More lenient conditions for signals
            # Buy when price is below SMA with some threshold
            if (
                distance < -config["volatility_threshold"] * 0.8
                and volatility > avg_volatility * 0.8
            ):
                signals.append(
                    {
                        "timestamp": time,
                        "signal": "buy",
                        "price": price,
                        "strategy": "dca",
                    }
                )

            # Sell when price is above SMA with some threshold
            elif (
                distance > config["volatility_threshold"] * 0.8
                and volatility < avg_volatility * 1.2
            ):
                signals.append(
                    {
                        "timestamp": time,
                        "signal": "sell",
                        "price": price,
                        "strategy": "dca",
                    }
                )

        return signals

    def _process_combined_signals(self, df):
        # Get signals from both strategies
        bollinger_signals = self._process_bollinger_signals(df)
        dca_signals = self._process_dca_signals(df)

        # Combine all signals and sort by timestamp
        combined_signals = bollinger_signals + dca_signals
        combined_signals.sort(key=lambda x: x["timestamp"])

        return combined_signals


def backtest_trades(signals, trade_amount_usd=10):
    trades = []
    total_profit_pct = 0
    num_completed_trades = 0

    # Track positions for each strategy
    positions = {
        "bollinger": {"in_position": False, "entry_price": 0, "entry_time": None},
        "dca": {"in_position": False, "entry_price": 0, "entry_time": None},
    }

    for signal in signals:
        strategy = signal["strategy"]
        current_pos = positions[strategy]

        # Handle buy signals
        if signal["signal"] == "buy":
            if not current_pos["in_position"]:  # Only buy if not in position
                current_pos["in_position"] = True
                current_pos["entry_price"] = signal["price"]
                current_pos["entry_time"] = signal["timestamp"]
                trades.append(
                    {
                        "time": signal["timestamp"],
                        "price": signal["price"],
                        "signal": "buy",
                        "strategy": strategy,
                    }
                )

        # Handle sell signals
        elif signal["signal"] == "sell":
            if current_pos["in_position"]:  # Only sell if in position
                # Calculate profit percentage
                profit_pct = (
                    (signal["price"] - current_pos["entry_price"])
                    / current_pos["entry_price"]
                ) * 100
                total_profit_pct += profit_pct
                num_completed_trades += 1

                # Calculate holding period
                holding_period = signal["timestamp"] - current_pos["entry_time"]
                holding_hours = holding_period.total_seconds() / 3600

                trades.append(
                    {
                        "time": signal["timestamp"],
                        "price": signal["price"],
                        "signal": "sell",
                        "strategy": strategy,
                        "profit_pct": profit_pct,
                        "holding_hours": holding_hours,
                    }
                )

                # Mark the corresponding buy trade with the profit
                for t in reversed(trades):
                    if (
                        t["strategy"] == strategy
                        and t["signal"] == "buy"
                        and "profit_pct" not in t
                        and t["time"] == current_pos["entry_time"]
                    ):
                        t["profit_pct"] = profit_pct
                        t["holding_hours"] = holding_hours
                        break

                current_pos["in_position"] = False
                current_pos["entry_price"] = 0
                current_pos["entry_time"] = None

    # Calculate total signals
    total_signals = len(trades)
    buy_signals = len([t for t in trades if t["signal"] == "buy"])
    sell_signals = len([t for t in trades if t["signal"] == "sell"])

    # Calculate average profit per trade
    avg_profit_pct = (
        total_profit_pct / num_completed_trades if num_completed_trades > 0 else 0
    )

    return (
        trades,
        total_signals,
        buy_signals,
        sell_signals,
        total_profit_pct,
        avg_profit_pct,
        num_completed_trades,
    )


def fetch_data(symbol, days=20, max_retries=3):
    """Fetch data with rate limiting and retry logic using CCXT"""
    for attempt in range(max_retries):
        try:
            # Add random delay to avoid rate limiting
            time.sleep(random.uniform(1, 3))

            # Initialize Binance exchange
            # exchange = ccxt.binance({
            #     'enableRateLimit': True,
            #     'options': {
            #         'defaultType': 'spot'
            #     }
            # })
            exchange = ccxt.kucoin(
                {"enableRateLimit": True, "options": {"defaultType": "spot"}}
            )
            # Convert symbol to CCXT format
            ccxt_symbol = f"{symbol}/USDT"

            # Calculate timeframes
            end = datetime.now()
            start = end - timedelta(days=days)

            logger.info(
                f"Attempt {attempt + 1}/{max_retries}: Fetching {ccxt_symbol} data..."
            )

            # Fetch OHLCV data (1 hour intervals)
            ohlcv = exchange.fetch_ohlcv(
                symbol=ccxt_symbol,
                timeframe="1h",
                since=int(start.timestamp() * 1000),
                limit=1000,  # Get up to 1000 candles
            )

            if not ohlcv:
                logger.warning(f"No data received for {ccxt_symbol}, retrying...")
                time.sleep(random.uniform(2, 5))
                continue

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            # Rename columns to match expected format
            df.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                },
                inplace=True,
            )

            logger.info(f"Successfully fetched {len(df)} data points for {ccxt_symbol}")
            return df

        except Exception as e:
            error_msg = str(e)
            if (
                "rate limit" in error_msg.lower()
                or "too many requests" in error_msg.lower()
            ):
                wait_time = (attempt + 1) * 5  # Exponential backoff
                logger.warning(
                    f"Rate limited for {ccxt_symbol}. Waiting {wait_time} seconds before retry..."
                )
                time.sleep(wait_time)
            else:
                logger.error(f"Error fetching data for {ccxt_symbol}: {error_msg}")
                time.sleep(random.uniform(1, 3))

    logger.error(f"Failed to fetch data for {symbol} after {max_retries} attempts")
    return pd.DataFrame()  # Return empty DataFrame


def tune_parameters(df, strategy_type="combined", target_return=None):
    if strategy_type == "combined":
        # Define parameter ranges for both strategies
        bollinger_periods = [3, 5, 7, 10, 15, 20]  # Added more periods
        bollinger_deviations = [0.5, 0.7, 1.0, 1.2, 1.5, 2.0]  # Added more deviations
        dca_periods = [5, 10, 15, 20, 25, 30]  # Added more periods
        dca_thresholds = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]  # Added more thresholds

        best_profit = float("-inf")
        best_params = None
        results = []

        logger.info("Starting Strategy hyperparameter tuning...")
        total_combinations = (
            len(bollinger_periods)
            * len(bollinger_deviations)
            * len(dca_periods)
            * len(dca_thresholds)
        )
        current_combination = 0

        # Create separate results for each strategy
        bollinger_results = []
        dca_results = []

        for b_period, b_dev, d_period, d_thresh in product(
            bollinger_periods, bollinger_deviations, dca_periods, dca_thresholds
        ):
            current_combination += 1
            logger.info(
                f"Testing combination {current_combination}/{total_combinations}"
            )

            # Test Bollinger strategy
            config_bollinger = {
                "strategy_type": "bollinger",
                "channel_strategy": {
                    "channel_period": b_period,
                    "channel_deviation": b_dev,
                },
            }
            agent = StrategyAgent(config_bollinger)
            signals = agent.process_channel_signals(df)
            trades, _, _, _, total_profit_pct, avg_profit_pct, num_trades = (
                backtest_trades(signals)
            )

            bollinger_results.append(
                {
                    "period": b_period,
                    "deviation": b_dev,
                    "total_profit_pct": total_profit_pct,
                    "avg_profit_pct": avg_profit_pct,
                    "num_trades": num_trades,
                }
            )

            # Test DCA strategy
            config_dca = {
                "strategy_type": "dca",
                "dca_strategy": {
                    "volatility_period": d_period,
                    "volatility_threshold": d_thresh,
                },
            }
            agent = StrategyAgent(config_dca)
            signals = agent.process_channel_signals(df)
            trades, _, _, _, total_profit_pct, avg_profit_pct, num_trades = (
                backtest_trades(signals)
            )

            dca_results.append(
                {
                    "period": d_period,
                    "threshold": d_thresh,
                    "total_profit_pct": total_profit_pct,
                    "avg_profit_pct": avg_profit_pct,
                    "num_trades": num_trades,
                }
            )

            # Test combined strategy
            config_combined = {
                "strategy_type": "combined",
                "channel_strategy": {
                    "channel_period": b_period,
                    "channel_deviation": b_dev,
                },
                "dca_strategy": {
                    "volatility_period": d_period,
                    "volatility_threshold": d_thresh,
                },
            }

            agent = StrategyAgent(config_combined)
            signals = agent.process_channel_signals(df)
            (
                trades,
                total_signals,
                buy_signals,
                sell_signals,
                total_profit_pct,
                avg_profit_pct,
                num_completed_trades,
            ) = backtest_trades(signals)

            results.append(
                {
                    "bollinger_period": b_period,
                    "bollinger_deviation": b_dev,
                    "dca_period": d_period,
                    "dca_threshold": d_thresh,
                    "total_profit_pct": total_profit_pct,
                    "avg_profit_pct": avg_profit_pct,
                    "num_trades": len(trades),
                    "total_signals": total_signals,
                    "buy_signals": buy_signals,
                    "sell_signals": sell_signals,
                }
            )

            if total_profit_pct > best_profit:
                best_profit = total_profit_pct
                best_params = config_combined

        # Print best parameters for each strategy
        logger.info("Best Parameters Found:")

        # Best Bollinger
        best_bollinger = max(bollinger_results, key=lambda x: x["total_profit_pct"])
        logger.info(
            f"Bollinger Strategy - Period: {best_bollinger['period']}, Deviation: {best_bollinger['deviation']}, Return: {best_bollinger['total_profit_pct']:.2f}%, Trades: {best_bollinger['num_trades']}"
        )

        # Best DCA
        best_dca = max(dca_results, key=lambda x: x["total_profit_pct"])
        logger.info(
            f"DCA Strategy - Period: {best_dca['period']}, Threshold: {best_dca['threshold']}, Return: {best_dca['total_profit_pct']:.2f}%, Trades: {best_dca['num_trades']}"
        )

        # Best Combined
        best_combined = max(results, key=lambda x: x["total_profit_pct"])
        logger.info(
            f"Combined Strategy - Bollinger Period: {best_combined['bollinger_period']}, Deviation: {best_combined['bollinger_deviation']}, DCA Period: {best_combined['dca_period']}, Threshold: {best_combined['dca_threshold']}, Return: {best_combined['total_profit_pct']:.2f}%, Trades: {best_combined['num_trades']}"
        )

        return best_params, results

    return None, []


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Crypto Trading Strategy Parameter Optimizer"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="ALL",
        help="Trading symbol (default: ALL for BTC, SOL, DOGE)",
    )
    parser.add_argument(
        "--days", type=int, default=20, help="Number of days to backtest (default: 20)"
    )
    parser.add_argument(
        "--bollinger_period",
        type=int,
        nargs="+",
        default=[3, 5, 7, 10, 15, 20],
        help="Bollinger period values to test",
    )
    parser.add_argument(
        "--bollinger_dev",
        type=float,
        nargs="+",
        default=[0.5, 0.7, 1.0, 1.2, 1.5, 2.0],
        help="Bollinger deviation values to test",
    )
    parser.add_argument(
        "--dca_period",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20, 25, 30],
        help="DCA period values to test",
    )
    parser.add_argument(
        "--dca_threshold",
        type=float,
        nargs="+",
        default=[0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
        help="DCA threshold values to test",
    )

    # Parse arguments
    args = parser.parse_args()

    # Define symbols to process (CCXT format)
    symbols = (
        ["BTC", "SOL", "DOGE"]
        if args.symbol.upper() == "ALL"
        else [args.symbol.upper()]
    )

    # Override parameter ranges with command line arguments
    global bollinger_periods, bollinger_deviations, dca_periods, dca_thresholds
    bollinger_periods = args.bollinger_period
    bollinger_deviations = args.bollinger_dev
    dca_periods = args.dca_period
    dca_thresholds = args.dca_threshold

    # Process each symbol
    for symbol in symbols:
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting parameter tuning for {symbol}")
        logger.info(f"{'='*50}")

        # Add delay between symbols to avoid rate limiting
        if symbol != symbols[0]:
            delay = random.uniform(5, 10)
            logger.info(f"Waiting {delay:.1f} seconds before processing {symbol}...")
            time.sleep(delay)

        try:
            # Fetch data
            logger.info(
                f"Fetching {args.days} days of {symbol}/USDT data from Binance..."
            )
            df = fetch_data(symbol, days=args.days)

            if df.empty:
                logger.error(f"No data available for {symbol}")
                continue

            logger.info("Testing Combined Strategy...")
            logger.info(
                f"Parameters being tested - Bollinger Periods: {args.bollinger_period}, Deviations: {args.bollinger_dev}, DCA Periods: {args.dca_period}, Thresholds: {args.dca_threshold}"
            )

            # Run parameter tuning
            best_params, tuning_results = tune_parameters(df, strategy_type="combined")

            if best_params:
                logger.info(f"\nBest Parameters Found for {symbol}:")
                logger.info(
                    f"Bollinger Strategy - Channel Period: {best_params['channel_strategy']['channel_period']}, Deviation: {best_params['channel_strategy']['channel_deviation']}"
                )
                logger.info(
                    f"DCA Strategy - Volatility Period: {best_params['dca_strategy']['volatility_period']}, Threshold: {best_params['dca_strategy']['volatility_threshold']}"
                )

                logger.info(
                    f"\nRunning combined strategy with best parameters for {symbol}..."
                )
                agent = StrategyAgent(best_params)
                signals = agent.process_channel_signals(df)

                # Get all signals
                (
                    trades,
                    total_signals,
                    buy_signals,
                    sell_signals,
                    total_profit_pct,
                    avg_profit_pct,
                    num_completed_trades,
                ) = backtest_trades(signals)

                logger.info(
                    f"\nFinal Results for {symbol} ({args.days} days backtest):"
                )
                logger.info(f"Total Signals: {total_signals}")
                logger.info(f"Buy Signals: {buy_signals}")
                logger.info(f"Sell Signals: {sell_signals}")
                logger.info(f"Total Return: {total_profit_pct:.2f}%")
                logger.info(f"Average Return per Trade: {avg_profit_pct:.2f}%")
                logger.info(f"Number of Completed Trades: {num_completed_trades}")

                # Calculate win rate
                winning_trades = len(
                    [t for t in trades if t["signal"] == "sell" and t["profit_pct"] > 0]
                )
                win_rate = (
                    (winning_trades / num_completed_trades * 100)
                    if num_completed_trades > 0
                    else 0
                )
                logger.info(f"Win Rate: {win_rate:.1f}%")

                # Save results to database
                try:
                    db_service = MongoDBService()
                    db_service.db.optimized_parameters.update_one(
                        {"symbol": symbol},
                        {
                            "$set": {
                                "symbol": symbol,
                                "timestamp": datetime.utcnow().isoformat(),
                                "parameters": best_params,
                                "performance": {
                                    "total_return": total_profit_pct,
                                    "avg_return": avg_profit_pct,
                                    "num_trades": num_completed_trades,
                                    "win_rate": win_rate,
                                },
                            }
                        },
                        upsert=True,
                    )
                    logger.info(f"Saved optimized parameters for {symbol} to database")
                except Exception as e:
                    logger.error(f"Error saving results to database: {str(e)}")
            else:
                logger.error(
                    f"No suitable parameters found for {symbol}. Please try adjusting the strategy parameters."
                )

        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
            continue

    logger.info("\nParameter tuning completed for all symbols")


if __name__ == "__main__":
    main()
