import abc
from market_history import *

class PricingAgent(abc.ABC):
    def __init__(self, firm_id: int, price_per_unit: float):
        self.firm_id = firm_id
        self.price_per_unit = price_per_unit

    @abc.abstractmethod
    def generate_price(self, market_history: MarketHistory) -> float:
        pass

    def get_firm_id(self) -> int:
        return self.firm_id
    
    def get_price_per_unit(self) -> float:
        return self.price_per_unit