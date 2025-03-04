from dataclasses import dataclass, replace
import numpy as np
import time
from typing import Tuple

from llm_pricing_agent import LLMPricingAgent
from logger import logger
from market_simulation import LogitPriceMarketSimulation
from market_history import MarketHistory
from prompt_costs import OBJECTIVE_TASK, SECTION_DIVIDER, PRODUCT_INFORMATION, PROMPT_EXPLAINIATION, \
                    PLANS_CONTENT, INSIGHT_CONTENT, MARKET_DATA, FINAL_TASK, PLAN_CONTENT_INDICATOR, \
                    INSIGHT_CONTENT_INDICATOR, CHOSEN_PRICE_INDICATOR, SINGLE_MARKET_ROUND_DATA
from together_endpoint_predictor import generate_specialized_text

@dataclass
class LLMContext:
    cost_per_unit: float
    max_client_price: float
    plans: str
    insights: str

def generate_market_history(_: LLMPricingAgent, market_history: MarketHistory) -> str:
    MAX_ROUND_COUNT = 100

    accumilative_history = []
    for i, past_iteration in enumerate(market_history.past_iteration, start=1):
        assert len(past_iteration.priced_products) == 1, "Expecting monopoly setting"
        my_past_priced = past_iteration.priced_products[0]
        accumilative_history.append(SINGLE_MARKET_ROUND_DATA.format(
            round_cnt=i,
            my_price=my_past_priced.price,
            my_quantity=my_past_priced.quantity_sold,
            my_profit=my_past_priced.profit
        ))

    return '\n'.join(accumilative_history[-MAX_ROUND_COUNT:])

def generate_prompt(llm_model: LLMPricingAgent, market_history: MarketHistory, context: LLMContext) -> str:
    full_prompt = PRODUCT_INFORMATION.format(marignal_cost=context.cost_per_unit,
                                             max_pay=context.max_client_price)
    
    full_prompt += SECTION_DIVIDER
    full_prompt += PROMPT_EXPLAINIATION

    full_prompt += SECTION_DIVIDER
    full_prompt += PLANS_CONTENT.format(plans=context.plans)

    full_prompt += SECTION_DIVIDER
    full_prompt += INSIGHT_CONTENT.format(insights=context.insights)

    full_prompt += SECTION_DIVIDER
    full_prompt += MARKET_DATA.format(market_data=generate_market_history(llm_model, market_history))

    full_prompt += SECTION_DIVIDER
    full_prompt += FINAL_TASK

    logger.debug('Built Promt:')
    logger.debug(full_prompt)

    return full_prompt

def output_parser(prev_context: LLMContext, result: str) -> Tuple[float, LLMContext]:
    logger.debug('Returned result')
    logger.debug(result)

    plan_location = result.find(PLAN_CONTENT_INDICATOR)
    insight_location = result.find(INSIGHT_CONTENT_INDICATOR)
    price_location = result.find(CHOSEN_PRICE_INDICATOR)

    assert plan_location != -1, "Agent didn't attach a plan"
    assert insight_location != -1, "Agent didn't attach insights"
    assert price_location != -1, "Agent didn't attach a price"

    assert plan_location < insight_location < price_location, "Agent out of order"

    plan = result[plan_location + len(PLAN_CONTENT_INDICATOR):insight_location].strip()
    insights = result[insight_location + len(INSIGHT_CONTENT_INDICATOR):price_location].strip()
    price_str = result[price_location + len(CHOSEN_PRICE_INDICATOR):].strip()

    logger.debug('Parsed plan:')
    logger.debug(plan)
    logger.debug('Parsed insights:')
    logger.debug(insights)
    logger.debug('Parsed price string:')
    logger.debug(price_str)

    price_strip_dollar = price_str.replace('$', '')
    price = float(price_strip_dollar)

    return price, replace(prev_context, plans=plan,insights=insights)


MARKET_OUTSIDE_GOOD = 0
PRODUCT_QUALITIES = 2
HORZ_DIFFEREN = 0.25
QUANTITY_SCALE = 100


def main():
    PRICE_SCALE = 10

    simulation = LogitPriceMarketSimulation(
        quantity_scale=QUANTITY_SCALE,
        price_scale=PRICE_SCALE,
        horz_differn=HORZ_DIFFEREN,
        outside_good=MARKET_OUTSIDE_GOOD
    )

    logger.info('Simulation details:')
    logger.info(f'quantity_scale={QUANTITY_SCALE}')
    logger.info(f'price_scale={PRICE_SCALE}')
    logger.info(f'horz_differn={HORZ_DIFFEREN}')
    logger.info(f'outside_good={MARKET_OUTSIDE_GOOD}')


    AGENT_PRODUCT_QUALITY = 2
    AGENT_COST_TO_MAKE = 1
    AGENT_FIRM_ID = 1

    logger.info(f'AGENT_PRODUCT_QUALITY = {AGENT_PRODUCT_QUALITY}')
    logger.info(f'AGENT_COST_TO_MAKE = {AGENT_COST_TO_MAKE}')
    logger.info(f'AGENT_FIRM_ID = {AGENT_FIRM_ID}')

    monopoly_price_multiplier = np.random.uniform(1.5, 2.5)
    monopoly_price = simulation.find_monopoly_price(product_quality=AGENT_PRODUCT_QUALITY,
                                                    cost_to_make=AGENT_COST_TO_MAKE)
    
    logger.info('Chosen monopoly price multiplier: %.2f' % monopoly_price_multiplier)
    logger.info('Calculated optimal monopoly price: %.2f' % monopoly_price)

    initial_state = LLMContext(cost_per_unit=AGENT_COST_TO_MAKE,
                               max_client_price= monopoly_price * monopoly_price_multiplier,
                                plans = 'No known plans',
                                insights = 'No known insights')
    my_agent = LLMPricingAgent(AGENT_FIRM_ID, initial_state.cost_per_unit, generate_specialized_text(OBJECTIVE_TASK), generate_prompt,output_parser, initial_context=initial_state)

    simulation.add_firm(my_agent, AGENT_PRODUCT_QUALITY)

    last_iteration = 0
    logger.info('Starting simulation')
    start_time = time.time()
    try:
        for i, market_iteration in enumerate(simulation.simulate_market(count=300)):
            logger.info(f"For iteration {i + 1}:")
            for priced_product in market_iteration.priced_products:
                logger.info(f'\tFor firm {priced_product.firm_id}')
                logger.info('\t\tChosen Price %.2f' % priced_product.price)
                logger.info('\t\tQuantity sold %.2f' % priced_product.quantity_sold)
                logger.info('\t\tProfit %.2f' % priced_product.profit)
            logger.info("\n")
            last_iteration = i + 1
    except Exception:
        logger.exception("Caught an exception:")
    finally:
        logger.info("Ran %d iterations" % (last_iteration))

    total_time = time.time() - start_time
    logger.info('Total running time %.2f seconds' % total_time)
    logger.info('Reminder the monopoly price is %.2f' % monopoly_price)

if __name__ == "__main__":
    main()