from dataclasses import replace
from typing import Tuple

from llm_pricing_agent import LLMPricingAgent
from logger import get_logger
from market_history import MarketHistory
from prompt_commons import generate_market_history
from prompt_costs import OBJECTIVE_TASK, SECTION_DIVIDER, PRODUCT_INFORMATION, PROMPT_EXPLAINIATION, \
                    PLANS_CONTENT, INSIGHT_CONTENT, MARKET_DATA, FINAL_TASK, PLAN_CONTENT_INDICATOR, \
                    INSIGHT_CONTENT_INDICATOR, CHOSEN_PRICE_INDICATOR
from simple_llm_context import LLMContext

def generate_prompt(llm_model: LLMPricingAgent, market_history: MarketHistory, context: LLMContext) -> str:
    full_prompt = OBJECTIVE_TASK

    full_prompt += SECTION_DIVIDER
    full_prompt += PRODUCT_INFORMATION.format(marignal_cost=context.cost_per_unit,
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

    get_logger().debug('Built Promt:')
    get_logger().debug(full_prompt)

    return full_prompt

def output_parser(prev_context: LLMContext, result: str) -> Tuple[float, LLMContext]:
    get_logger().debug('Returned result')
    get_logger().debug(result)

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

    get_logger().debug('Parsed plan:')
    get_logger().debug(plan)
    get_logger().debug('Parsed insights:')
    get_logger().debug(insights)
    get_logger().debug('Parsed price string:')
    get_logger().debug(price_str)

    price_strip_dollar = price_str.replace('$', '')
    price = float(price_strip_dollar)

    return price, replace(prev_context, plans=plan,insights=insights)