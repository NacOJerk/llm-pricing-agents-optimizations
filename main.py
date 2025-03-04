from dataclasses import dataclass, replace
from typing import Tuple

from lllm_pricing_agent import LLLMPricingAgent, PromtContext
from market_simulation import LogitPriceMarketSimulation
from market_history import MarketHistory
import numpy as np
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

def generate_market_history(_: LLLMPricingAgent, market_history: MarketHistory) -> str:
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

    return '\n'.join(accumilative_history)

def generate_prompt(llm_model: LLLMPricingAgent, market_history: MarketHistory, context: LLMContext) -> str:
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

    print(full_prompt)

    return full_prompt

def output_parser(prev_context: LLMContext, result: str) -> Tuple[float, LLMContext]:
    print(result)
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

    price_strip_dollar = price_str.replace('$', '')
    price = float(price_strip_dollar)

    return price, replace(prev_context, plans=plan,insights=insights)


MARKET_OUTSIDE_GOOD = 0
PRODUCT_QUALITIES = 2
HORZ_DIFFEREN = 0.25
QUANTITY_SCALE = 100


def main():

    simulation = LogitPriceMarketSimulation(
        quantity_scale=QUANTITY_SCALE,
        price_scale=10,
        horz_differn=HORZ_DIFFEREN,
        outside_good=MARKET_OUTSIDE_GOOD
    )

    AGENT_PRODUCT_QUALITY = 1
    AGENT_COST_TO_MAKE = 1
    AGENT_FIRM_ID = 1
    monopoly_price_multiplier = np.random.uniform(1.5, 2.5)
    initial_state = LLMContext(cost_per_unit=AGENT_COST_TO_MAKE,
                               max_client_price=simulation.find_monopoly_price(product_quality=AGENT_PRODUCT_QUALITY,
                                                                               cost_to_make=AGENT_COST_TO_MAKE) * monopoly_price_multiplier,
                                plans = 'No known plans',
                                insights = 'No known insights')
    my_agent = LLLMPricingAgent(AGENT_FIRM_ID, initial_state.cost_per_unit, generate_specialized_text(OBJECTIVE_TASK), generate_prompt,output_parser, initial_context=initial_state)

    simulation.add_firm(my_agent, AGENT_PRODUCT_QUALITY)

    print(simulation.find_monopoly_price(1,1))
    for i, market_iteration in enumerate(simulation.simulate_market(count=5)):
        print(f"For iteration {i + 1}:")
        for priced_product in market_iteration.priced_products:
            print(f'\tFor firm {priced_product.firm_id}')
            print('\t\tChosen Price %.2f' % priced_product.price)
            print('\t\tQuantity sold %.2f' % priced_product.quantity_sold)
            print('\t\tProfit %.2f' % priced_product.profit)
        print("\n")
    
if __name__ == "__main__":
    main()