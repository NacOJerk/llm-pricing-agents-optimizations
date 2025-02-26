from dataclasses import dataclass
from typing import List

@dataclass
class PricedProduct:
    firm_id: int
    price: float
    quantity_sold: float

@dataclass
class MarketIteration:
    assigned_prices: List[PricedProduct]

@dataclass
class MarketHistory:
    past_iteration: List[MarketIteration]