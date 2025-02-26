from random import gauss

from market_history import *
from market_simulation import LogitPriceMarketSimulation
from pricing_agent import PricingAgent

class NaivePricingAgent(PricingAgent):
    def __init__(self, firm_id: int, price_per_unit: float, decrease_multiplier: float, decay_factor: float):
        super().__init__(firm_id, price_per_unit)
        self.decrease_multiplier = decrease_multiplier
        self.decay_factor = decay_factor
        self.chosen_price = 0

    def generate_price(self, market_history: MarketHistory) -> float:
        average_iteration_prices = []

        if len(market_history.past_iteration) == 0:
            self.chosen_price = max(self.price_per_unit * 1.01, gauss(self.price_per_unit+5, 3))
            return self.chosen_price
        
        for market_iteration in market_history.past_iteration:
            current_iteration_sum = 0
            total_items_sold = 0
            for priced_product in market_iteration.priced_products:
                if priced_product.firm_id == self.get_firm_id():
                    continue
                if priced_product.quantity_sold == 0:
                    continue

                current_iteration_sum += priced_product.price * priced_product.quantity_sold
                total_items_sold += priced_product.quantity_sold
            
            if total_items_sold == 0:
                average_iteration_prices.append(self.chosen_price)
                continue

            current_iteration_average = current_iteration_sum / total_items_sold
            average_iteration_prices.append(current_iteration_average)
        
        weighted_sum = 0
        total_weight = 0
        current_weight = 1
        for price in reversed(average_iteration_prices):
            weighted_sum += price * current_weight
            total_weight += current_weight
            current_weight *= self.decay_factor
        
        return max((weighted_sum / total_weight) * self.decrease_multiplier, self.get_price_per_unit() * 1.01)

def main():
    naive_agent1: NaivePricingAgent = NaivePricingAgent(0, 1, 0.98, 0.1)
    naive_agent2: NaivePricingAgent = NaivePricingAgent(1, 1, 0.93, 0.5)
    naive_agent3: NaivePricingAgent = NaivePricingAgent(3, 1, 0.91, 0.9)

    simulation = LogitPriceMarketSimulation(
        quantity_scale=100,
        price_scale=10,
        horz_differn=0.25,
        outside_good=0
    )
    for firm in (naive_agent1,naive_agent2, naive_agent3):
        simulation.add_firm(firm, 1)
    
    for i, market_iteration in enumerate(simulation.simulate_market(count=5)):
        print(f"For iteration {i + 1}:")
        for priced_product in market_iteration.priced_products:
            print(f'\tFor firm {priced_product.firm_id}')
            print('\t\tChosen Price %.2f' % priced_product.price)
            print('\t\tQuantity sold %.2f' % priced_product.quantity_sold)
        print("\n")
    
if __name__ == "__main__":
    main()