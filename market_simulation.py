from dataclasses import dataclass
from math import exp
from typing import List, Dict

from market_history import *
from pricing_agent import PricingAgent

@dataclass
class ProductAndPricer:
    product_quality: float
    pricer: PricingAgent


class LogitPriceMarketSimulation:
    def __init__(self, quantity_scale: float, price_scale: float, horz_differn: float, outside_good: float):
        self.ququantity_scale = quantity_scale
        self.price_scale = price_scale
        self.horz_differn = horz_differn
        self.outoutside_good = outside_good
        self.products: Dict[int, ProductAndPricer] = {}
        self.market_iterations: List[MarketIteration] = []
    
    def add_firm(self, agent: PricingAgent, quality: float):
        self.products[agent.firm_id] = ProductAndPricer(product_quality=quality, pricer=agent)

    def _simulate_market(self) -> MarketIteration:
        firm_prices: Dict[int, float] = {}
        firm_logits: Dict[int, float] = {}
        logits_sum: float = 0
        market_history = MarketHistory(self.market_iterations)

        for firm_id, pricer in self.products.items():
            product_price = pricer.pricer.generate_price(market_history)
            firm_prices[firm_id] = product_price
            product_price_scaled = product_price / self.price_scale
            quality_minus_price = pricer.product_quality - product_price_scaled
            q_m_p_scaled = quality_minus_price / self.horz_differn
            final_logit = exp(q_m_p_scaled)
            firm_logits[firm_id] = final_logit
            logits_sum += final_logit
        
        outside_good = exp(self.outoutside_good) / self.horz_differn
        logits_scale = logits_sum + outside_good
        market_results: List[PricedProduct] = []
        for firm_id, firm_logit in firm_logits.items():
            quantity_sold = self.ququantity_scale * (firm_logit / logits_scale)
            market_results.append(PricedProduct(
                firm_id=firm_id,
                price=firm_prices[firm_id],
                quantity_sold=quantity_sold,
                profit=(firm_prices[firm_id] - self.products[firm_id].pricer.get_price_per_unit()) * quantity_sold
            ))
        
        market_iteration = MarketIteration(market_results)
        self.market_iterations.append(market_iteration)

        return market_iteration
        
    def simulate_market(self, count=1) -> List[MarketIteration]:
        market_iterations = []
        for i in range(count):
            market_iterations.append(self._simulate_market())
        
        return market_iterations