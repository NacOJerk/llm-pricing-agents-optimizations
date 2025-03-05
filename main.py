import argparse
from dataclasses import asdict
from datetime import datetime
from enum import Enum, auto
import json
import numpy as np
from pathlib import Path
import time
from typing import Tuple, Dict

from json_prompt_setup import generate_prompt_for_json, output_json_parser, \
      has_examples as json_has_examples, set_add_example as json_set_add_example
from legacy_prompt_setup import generate_prompt, output_parser, has_examples, \
                                set_add_example
from llm_pricing_agent import LLMPricingAgent
from logger import init_logger, get_logger
from market_simulation import LogitPriceMarketSimulation
from market_history import MarketHistory
from prompt_commons import set_max_round_count, get_max_round_count
from simple_llm_context import LLMContext
from together_endpoint_predictor import generate_specialized_text, get_chosen_model, \
                                        get_available_models, set_chosen_model

MARKET_OUTSIDE_GOOD = 0
PRODUCT_QUALITIES = 2
HORZ_DIFFEREN = 0.25
QUANTITY_SCALE = 100
MARKET_ITERATIONS = 300

class PromptType(Enum):
    UNKNOWN = auto()
    LEGACY = auto()
    JSON = auto()

def simulate_full_experiment(price_scale: float, experiment_type: PromptType) -> Tuple[MarketHistory, Dict]:
    simulation = LogitPriceMarketSimulation(
        quantity_scale=QUANTITY_SCALE,
        price_scale=price_scale,
        horz_differn=HORZ_DIFFEREN,
        outside_good=MARKET_OUTSIDE_GOOD
    )

    if experiment_type == PromptType.LEGACY:
        prompt_pair = (generate_prompt, output_parser)
        has_example = has_examples()
    elif experiment_type == PromptType.JSON:
        prompt_pair = (generate_prompt_for_json, output_json_parser)
        has_example = json_has_examples()
    else:
        raise RuntimeError('Unsupported prompt type')

    get_logger().info('Simulation details:')
    get_logger().info(f'\tquantity_scale: {QUANTITY_SCALE}')
    get_logger().info(f'\tprice_scale: {price_scale}')
    get_logger().info(f'\thorz_differn: {HORZ_DIFFEREN}')
    get_logger().info(f'\toutside_good: {MARKET_OUTSIDE_GOOD}')
    get_logger().info(f'\tPrompt type: {experiment_type}')
    get_logger().info(f'\tModel: {get_chosen_model()}')
    get_logger().info(f'\tRound memory: {get_max_round_count()}')
    get_logger().info(f'\tHas example: {has_example}')


    AGENT_PRODUCT_QUALITY = 2
    AGENT_COST_TO_MAKE = 1
    AGENT_FIRM_ID = 1

    get_logger().info(f'AGENT_PRODUCT_QUALITY = {AGENT_PRODUCT_QUALITY}')
    get_logger().info(f'AGENT_COST_TO_MAKE = {AGENT_COST_TO_MAKE}')
    get_logger().info(f'AGENT_FIRM_ID = {AGENT_FIRM_ID}')

    monopoly_price_multiplier = np.random.uniform(1.5, 2.5)
    monopoly_price = simulation.find_monopoly_price(product_quality=AGENT_PRODUCT_QUALITY,
                                                    cost_to_make=AGENT_COST_TO_MAKE)
    
    get_logger().info('Chosen monopoly price multiplier: %.2f' % monopoly_price_multiplier)
    get_logger().info('Calculated optimal monopoly price: %.2f' % monopoly_price)

    initial_state = LLMContext(cost_per_unit=AGENT_COST_TO_MAKE,
                               max_client_price= monopoly_price * monopoly_price_multiplier,
                                plans = 'No known plans',
                                insights = 'No known insights')
    my_agent = LLMPricingAgent(AGENT_FIRM_ID,
                               initial_state.cost_per_unit,
                               generate_specialized_text('', max_toxens=None),
                               *prompt_pair,
                               initial_context=initial_state)

    simulation.add_firm(my_agent, AGENT_PRODUCT_QUALITY)

    failed = False
    last_iteration = 0
    get_logger().info('Starting simulation')
    start_time = time.time()
    try:
        for i, market_iteration in enumerate(simulation.simulate_market(count=MARKET_ITERATIONS)):
            get_logger().info(f"For iteration {i + 1}:")
            for priced_product in market_iteration.priced_products:
                get_logger().info(f'\tFor firm {priced_product.firm_id}')
                get_logger().info('\t\tChosen Price %.2f' % priced_product.price)
                get_logger().info('\t\tQuantity sold %.2f' % priced_product.quantity_sold)
                get_logger().info('\t\tProfit %.2f' % priced_product.profit)
            get_logger().info("\n")
            last_iteration = i + 1
    except Exception:
        get_logger().exception("Caught an exception:")
        failed = True
    finally:
        get_logger().info("Ran %d iterations" % (last_iteration))

    total_time = time.time() - start_time
    get_logger().info('Total running time %.2f seconds' % total_time)
    get_logger().info('Reminder the monopoly price is %.2f' % monopoly_price)
    additional_context = {'monopoly_price': monopoly_price,
                          'total_time': total_time,
                          'total_exceptions': my_agent.total_exceptions,
                          'monopoly_price_multiplier': monopoly_price_multiplier,
                          'failed': failed,
                          'used_model': get_chosen_model(),
                          'round_memory': get_max_round_count(),
                          'experiment_type': repr(experiment_type),
                          'has_example': has_example,
                          'total_iterations': len(simulation.market_iterations)
                        }

    return MarketHistory(simulation.market_iterations), additional_context

def get_args():
    parser = argparse.ArgumentParser(
                prog='llm_pricer',
                description='A simple llm pricing agent')
    parser.add_argument('--dest-dir',
                        help='Location to save all of our experiment data',
                        required=True)
    parser.add_argument('--prompt-type',
                    help='The type of prompt experiment to run',
                    choices=['legacy', 'json'],
                    required=True)
    parser.add_argument('--model',
                help='The type of prompt experiment to run',
                choices=get_available_models(),
                required=True)
    parser.add_argument('--round-memory',
                help='The max amount of past round present in the prompts',
                type=int,
                default=100,
                required=False)
    parser.add_argument('--add-example',
            help='The max amount of past round present in the prompts',
            default=False,
            action='store_true',
            required=False)

    return parser.parse_args()

def main():
    args = get_args()

    prompt_type: PromptType = PromptType.UNKNOWN
    if args.prompt_type == 'legacy':
        prompt_type = PromptType.LEGACY
    elif args.prompt_type == 'json':
        prompt_type = PromptType.JSON

    path = Path(args.dest_dir)
    init_logger(path)

    set_chosen_model(args.model)

    set_max_round_count(args.round_memory)
    set_add_example(args.add_example)
    json_set_add_example(args.add_example)

    market_history_template = datetime.now().strftime('market_history_%%.2f_%H_%M_%d_%m_%Y.json')

    for scale in [1, 3.2, 10]:
        market_history, addit_data = simulate_full_experiment(scale, prompt_type)
        market_history_transformed = asdict(market_history)
        final_state = {
            'additional_context': addit_data,
            'market_history': market_history_transformed
        }
        with open(path / (market_history_template % scale), 'w') as f:
            json.dump(final_state, f)

if __name__ == "__main__":
    main()