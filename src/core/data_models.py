from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime

class TimestampedModel(BaseModel):
    """Base model with timestamp fields"""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class PriceDataPoint(BaseModel):
    """Single price data point"""
    datetime: str = Field(..., description="Timestamp in YYYY-MM-DD HH:MM:SS+00:00 format")
    price: float = Field(..., description="Closing price")
    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    volume: float = Field(..., description="Trading volume")
    price_change: float = Field(..., description="Price change percentage")

class PriceSnapshot(TimestampedModel):
    """Price snapshot record - one entry per token"""
    token_symbol: str = Field(..., description="Token symbol")
    source: str = Field("yahoo_finance", description="Price data source")
    metadata: dict = Field(default_factory=dict, description="Additional price metadata")
    price_data: List[PriceDataPoint] = Field(default_factory=list, description="List of price data points") 