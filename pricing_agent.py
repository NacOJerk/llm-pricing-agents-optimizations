import abc
from market_history import *

class PricingAgent(abc.ABC):
    def __init__(self, firm_id: int):
        self.firm_id = firm_id

    @abc.abstractmethod
    def generate_price(self, market_history: MarketHistory) -> float:
        pass

    def get_firm_id(self) -> int:
        return self.firm_id