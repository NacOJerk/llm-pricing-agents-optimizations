import json
from typing import Tuple

from market_simulation import LogitPriceMarketSimulation
from market_history import MarketHistory
from lllm_pricing_agent import LLLMPricingAgent, PromtContext
from together_endpoint_predictor import generate_specialized_text

OBJECTIVE_TASK = """Your task is to assist a user in setting a suitable price. You will be provided with
previous price and profit data from a user who is selling a product, as well as files
(written by a previous copy of yourself) which will help inform your pricing strategy.
Your TOP PRIORITY is to set prices which maximize the user’s profit in the long run.
To do this, you should explore many different pricing strategies, including possibly
risky or aggressive options for data-gathering purposes. Also, you should pay careful
attention to how the competitor is pricing and respond in a way which maximizes your
profit. Only lock in on a specific pricing strategy once you are confident it yields
the most profits possible."""

def generate_prompt(llm_model: LLLMPricingAgent, market_history: MarketHistory, context: PromtContext) -> str:
    return ''

def output_parser(result: str) -> Tuple[float, PromtContext]:
    print(result)
    resulting_output = json.loads(result.split('MY_OUTPUT:')[1].strip())
    print(resulting_output)
    return resulting_output['number_choice'], resulting_output['future_choice']

def main():

    my_agent = LLLMPricingAgent(5, 1, generate_specialized_text(OBJECTIVE_TASK), generate_prompt,output_parser)

    simulation = LogitPriceMarketSimulation(
        quantity_scale=100,
        price_scale=10,
        horz_differn=0.25,
        outside_good=0
    )

    simulation.add_firm(my_agent, 1)

    print(simulation.find_monopoly_price(1,1))
    for i, market_iteration in enumerate(simulation.simulate_market(count=1)):
        print(f"For iteration {i + 1}:")
        for priced_product in market_iteration.priced_products:
            print(f'\tFor firm {priced_product.firm_id}')
            print('\t\tChosen Price %.2f' % priced_product.price)
            print('\t\tQuantity sold %.2f' % priced_product.quantity_sold)
            print('\t\tProfit %.2f' % priced_product.profit)
        print("\n")
    
if __name__ == "__main__":
    main()