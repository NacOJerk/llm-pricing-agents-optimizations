from market_history import *
from pricing_agent import PricingAgent

class NaivePricingAgent(PricingAgent):
    def __init__(self, firm_id: int, decrease_multiplier: float, decay_factor: float):
        super().__init__(firm_id)
        self.decrease_multiplier = decrease_multiplier
        self.decay_factor = decay_factor

    def generate_price(self, market_history: MarketHistory) -> float:
        average_iteration_prices = []
        for market_iteration in market_history.past_iteration:
            current_iteration_sum = 0
            counted_prices = 0
            for priced_product in market_iteration.assigned_prices:
                if priced_product.firm_id == self.get_firm_id():
                    continue

                current_iteration_sum += priced_product.price
                counted_prices += 1
            
            current_iteration_average = current_iteration_sum / counted_prices
            average_iteration_prices.append(current_iteration_average)
        
        weighted_sum = 0
        total_weight = 0
        current_weight = 1
        for price in reversed(average_iteration_prices):
            weighted_sum += price * current_weight
            total_weight += current_weight
            current_weight *= self.decay_factor
        
        return (weighted_sum / total_weight) * self.decrease_multiplier

def main():
    test: PricedProduct = PricedProduct(2, 0.2)
    print(repr(test))

    test2: MarketIteration = MarketIteration([
        PricedProduct(1, 0.4),
        PricedProduct(5, 0.1),
        PricedProduct(42, 0.32),
    ])
    print(test2)

    past_prices = []
    for i in range(5):
        single_price: MarketIteration = MarketIteration([
            PricedProduct(1, 0.4 * i),
            PricedProduct(5, 0.1 * i + 2),
            PricedProduct(42, 0.32 * i + 0.1),
        ])
        past_prices.append(single_price)
    test3: MarketHistory = MarketHistory(past_prices)
    print(test3)

    naive_agent: NaivePricingAgent = NaivePricingAgent(42, 0.98, 0.5)
    chosen_price = naive_agent.generate_price(test3)
    print("%.2f" % chosen_price)

if __name__ == "__main__":
    main()