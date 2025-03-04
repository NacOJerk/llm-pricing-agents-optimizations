import argparse
import json
import numpy as np
from typing import List

def get_arguments():
    parser = argparse.ArgumentParser(
                    prog='experiment_analyzer',
                    description='Analyze experiment output jsons')
    parser.add_argument('--source',
                        type=argparse.FileType('rb', 0),
                        required=True)
    
    return parser.parse_args()

def check_converages_to(sorted_prices: List[float], price: float, converagnce_distance=0.05) -> bool:
    lowest_prices = sorted_prices[:10]
    top_prices = sorted_prices[-10:]

    for test_price in (lowest_prices + top_prices):
        distance = (np.abs(test_price - price) / price)
        if distance > converagnce_distance:
            return False

    return True

def main():
    arguments = get_arguments()
    data = json.load(arguments.source)
    addit_data = data['additional_context']
    market_history = data['market_history']['past_iteration']
    
    print('Used model: %s' % addit_data['used_model'])
    print('Total time: %.2f seconds' % addit_data['total_time'])

    monopoly_price = addit_data['monopoly_price']
    print('monopoly price: %.2f' % monopoly_price)

    prices = []

    final_100_attemps = market_history[-100:]
    for attempt in final_100_attemps:
        priced_products = attempt['priced_products']
        assert len(priced_products) == 1, 'We only analyze monopoly experiments'
        priced_product = priced_products[0]
        price = priced_product['price']
        prices.append(price)
    
    sorted_prices = sorted(prices)

    print('Average price: %.2f$' % np.average(prices))
    converages = any(map(lambda x: check_converages_to(sorted_prices, x), prices + [check_converages_to(sorted_prices, np.average(prices))]))
    print('Converges to anything:', converages)
    print('Converges to monopoly:', check_converages_to(sorted_prices, monopoly_price))

if __name__ == "__main__":
    main()