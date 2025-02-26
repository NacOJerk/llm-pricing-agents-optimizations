import re
import json

from market_simulation import LogitPriceMarketSimulation
from google_endpoint_predictor import generate_specialized_text
from lllm_pricing_agent import LLLMPricingAgent

def generate_prompt(firm_id, market_history, context):
    if context is None:
        return "Please choose two numbers between 1 to 10, use the following json format and nothing else:\nMY_OUTPUT: {\"number_choice\": {NUMBER_RESULT}, \"future_choice\": {NUMBER_RESULT}}"
    return "Please choose two numbers between 1 to 10, your previous \"future_choice\" was: %d, please take that into account.\n Use the following json format and nothing else:\nMY_OUTPUT: {\"number_choice\": {NUMBER_RESULT}, \"future_choice\": {NUMBER_RESULT}}" % context

def output_parser(result):
    print(result)
    resulting_output = json.loads(result.split('MY_OUTPUT:')[1].strip())
    print(resulting_output)
    return resulting_output['number_choice'], resulting_output['future_choice']

def main():

    my_agent = LLLMPricingAgent(5, 1, generate_specialized_text("You are an obidient AI assistant. You respond with short answers following you directive, you add no uneeded text and follow the requested format."), generate_prompt,output_parser)

    simulation = LogitPriceMarketSimulation(
        quantity_scale=100,
        price_scale=10,
        horz_differn=0.25,
        outside_good=0
    )

    simulation.add_firm(my_agent, 1)
    
    for i, market_iteration in enumerate(simulation.simulate_market(count=30)):
        print(f"For iteration {i + 1}:")
        for priced_product in market_iteration.priced_products:
            print(f'\tFor firm {priced_product.firm_id}')
            print('\t\tChosen Price %.2f' % priced_product.price)
            print('\t\tQuantity sold %.2f' % priced_product.quantity_sold)
            print('\t\tProfit %.2f' % priced_product.profit)
        print("\n")
    
if __name__ == "__main__":
    main()