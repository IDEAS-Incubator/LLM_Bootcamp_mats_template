import ccxt
import asyncio
import random
from datetime import datetime
from core.data_models import PriceSnapshot, PriceDataPoint
from services.database_service import MongoDBService
from utils.logging_utils import trading_logger


async def fetch_historical_data(
    symbol: str, team: str, interval: str = "1m", limit: int = 60
):
    """Fetch historical price data for a symbol using ccxt (Binance)"""
    try:
        trading_logger.log_message(
            team,
            f"Fetching historical data for {symbol} from Binance with interval {interval}",
            "INFO",
        )

        await asyncio.sleep(random.uniform(1, 3))  # avoid rate limiting

        # Initialize Binance exchange
        # exchange = ccxt.binance()
        exchange = ccxt.kucoin()

        # Fetch historical OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit)

        if not ohlcv:
            trading_logger.log_message(
                team, f"No historical data available for {symbol}", "ERROR"
            )
            return None

        # Convert to list of PriceDataPoint objects
        price_data = []
        for candle in ohlcv:
            ts, open_, high, low, close, volume = candle
            dt_str = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M:%S")

            price_data.append(
                PriceDataPoint(
                    datetime=dt_str,
                    price=close,
                    open=open_,
                    high=high,
                    low=low,
                    volume=volume,
                    price_change=((close - open_) / open_) * 100 if open_ else 0,
                )
            )

        # Create price snapshot
        snapshot = PriceSnapshot(
            token_symbol=team,
            source="ccxt-binance",
            metadata={"interval": interval, "symbol": symbol},
            price_data=price_data,
        )

        return snapshot

    except Exception as e:
        trading_logger.log_message(
            team, f"Error fetching historical data for {symbol}: {str(e)}", "ERROR"
        )
        return None


async def main():
    db_service = MongoDBService()

    # Symbols for Binance format
    symbols = {"BTC": "BTC/USDT", "SOL": "SOL/USDT", "DOGE": "DOGE/USDT"}

    try:
        for i, (team, symbol) in enumerate(symbols.items()):
            trading_logger.log_message(
                team, f"Starting historical data fetch for {team}", "INFO"
            )

            if i > 0:
                delay = random.uniform(3, 6)
                trading_logger.log_message(
                    "system",
                    f"Waiting {delay:.1f} seconds before processing {team}...",
                    "INFO",
                )
                await asyncio.sleep(delay)

            snapshot = await fetch_historical_data(symbol, team)

            if snapshot:
                doc_id = db_service.save_price_snapshot(snapshot)
                trading_logger.log_message(
                    team, f"Saved historical data to database with ID {doc_id}", "INFO"
                )
            else:
                trading_logger.log_message(
                    team, f"Failed to fetch historical data", "ERROR"
                )

        trading_logger.log_message(
            "system", "Completed historical data fetch for all symbols", "INFO"
        )

    except Exception as e:
        trading_logger.log_message(
            "system", f"Error in main execution: {str(e)}", "ERROR"
        )
    finally:
        db_service.close()


if __name__ == "__main__":
    asyncio.run(main())
