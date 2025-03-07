import argparse
import copy
import json
import numpy as np
from typing import List, Optional

def get_arguments():
    parser = argparse.ArgumentParser(
                    prog='experiment_analyzer',
                    description='Analyze experiment output jsons')
    parser.add_argument('--source',
                        type=argparse.FileType('rb', 0),
                        required=True)
    parser.add_argument('--json-mode',
                    default=False,
                    action='store_true',
                    required=False)

    return parser.parse_args()

def check_converages_to(sorted_prices: List[float], price: float, converagnce_distance=0.05) -> bool:
    lowest_prices = sorted_prices[:10]
    top_prices = sorted_prices[-10:]

    for test_price in (lowest_prices + top_prices):
        distance = (np.abs(test_price - price) / price)
        if distance > converagnce_distance:
            return False

    return True

def best_converagence_option(sorted_prices: List[float], options: List[float], converagnce_distance=0.05) -> Optional[float]:
    lowest_prices = sorted_prices[:10]
    top_prices = sorted_prices[-10:]

    best_price = None
    max_distance = None

    for price in options:
        current_max_distance = 0
        for test_price in (lowest_prices + top_prices):
            distance = (np.abs(test_price - price) / price)
            current_max_distance = max(distance, current_max_distance)
        
        if current_max_distance > converagnce_distance:
            continue

        if max_distance is None or current_max_distance < max_distance:
            best_price = price
            max_distance = current_max_distance

    return best_price

def output_json(args, raw_market_data: dict):
    addit_data = raw_market_data['additional_context']
    market_history = raw_market_data['market_history']['past_iteration']

    final_output = copy.deepcopy(addit_data)
    final_output['filename'] = args.source.name
    monopoly_price = addit_data['monopoly_price']

    if not addit_data['failed']:

        prices = []

        final_100_attemps = market_history[-100:]
        for attempt in final_100_attemps:
            priced_products = attempt['priced_products']
            assert len(priced_products) == 1, 'We only analyze monopoly experiments'
            priced_product = priced_products[0]
            price = priced_product['price']
            prices.append(price)
        
        sorted_prices = sorted(prices)
        average_price_last_100 = np.average(prices)

        final_output['average_price_last_100'] = average_price_last_100
        converage_options = prices + [average_price_last_100]

        converages = any(map(lambda x: check_converages_to(sorted_prices, x), converage_options))
        final_output['converages'] = converages
        final_output['converages_to_monopoly'] = check_converages_to(sorted_prices, monopoly_price)

        final_output['converages_number'] = None
        final_output['converages_number_dist_to_mono'] = None

        if converages:
            conv_num = best_converagence_option(sorted_prices, converage_options)
            assert conv_num is not None, 'We should be "converaging" to something'
            final_output['converages_number'] = conv_num
            final_output['converages_number_dist_to_mono'] = np.abs(conv_num - monopoly_price) / monopoly_price


    print(json.dumps(final_output))

def main():
    arguments = get_arguments()
    data = json.load(arguments.source)
    
    if arguments.json_mode:
        output_json(arguments, data)
        return
    
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
    converages = any(map(lambda x: check_converages_to(sorted_prices, x), prices + [np.average(prices)]))
    print('Converges to anything:', converages)
    print('Converges to monopoly:', check_converages_to(sorted_prices, monopoly_price))

if __name__ == "__main__":
    main()