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
    
    def extract_my_product(self, market_iteration: MarketIteration) -> PricedProduct:
        options = list(filter(lambda priced_product: priced_product.firm_id == self.firm_id, market_iteration.priced_products))
        assert len(options) == 1, "Invalid priced products (%d)" % len(options)
        return options[0]
