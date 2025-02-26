from market_history import *
from pricing_agent import PricingAgent
from random import gauss

class NaivePricingAgent(PricingAgent):
    def __init__(self, firm_id: int, price_per_unit: float, decrease_multiplier: float, decay_factor: float):
        super().__init__(firm_id, price_per_unit)
        self.decrease_multiplier = decrease_multiplier
        self.decay_factor = decay_factor

    def generate_price(self, market_history: MarketHistory) -> float:
        average_iteration_prices = []

        if len(market_history.past_iteration) == 0:
            return max(self.price_per_unit * 1.01, gauss(self.price_per_unit+5, 3))
        
        for market_iteration in market_history.past_iteration:
            current_iteration_sum = 0
            total_items_sold = 0
            for priced_product in market_iteration.assigned_prices:
                if priced_product.firm_id == self.get_firm_id():
                    continue
                if priced_product.quantity_sold == 0:
                    continue
                
                current_iteration_sum += priced_product.price * priced_product.quantity_sold
                total_items_sold += priced_product.quantity_sold
            
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
    test: PricedProduct = PricedProduct(2, 0.2, 5)
    print(repr(test))

    test2: MarketIteration = MarketIteration([
        PricedProduct(1, 0.4,6),
        PricedProduct(5, 0.1, 2),
        PricedProduct(42, 0.32,2),
    ])
    print(test2)

    past_prices = []
    for i in range(5):
        single_price: MarketIteration = MarketIteration([
            PricedProduct(1, 0.4 * i,12),
            PricedProduct(5, 0.1 * i + 2,42),
            PricedProduct(42, 0.32 * i + 0.1,1),
        ])
        past_prices.append(single_price)
    test3: MarketHistory = MarketHistory(past_prices)
    print(test3)

    naive_agent: NaivePricingAgent = NaivePricingAgent(42, 1, 0.98, 0.5)
    chosen_price = naive_agent.generate_price(test3)
    print("%.2f" % chosen_price)

if __name__ == "__main__":
    main()