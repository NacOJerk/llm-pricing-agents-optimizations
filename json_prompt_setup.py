from dataclasses import replace
import json
from typing import Tuple
import regex

from llm_pricing_agent import LLMPricingAgent
from logger import get_logger
from market_history import MarketHistory
from prompt_commons import generate_market_history
from prompt_costs import OBJECTIVE_TASK, SECTION_DIVIDER, PRODUCT_INFORMATION, PROMPT_EXPLAINIATION, \
                    PLANS_CONTENT, INSIGHT_CONTENT, MARKET_DATA
from prompt_costs_jsons import FINAL_TASK as FINAL_TASK_JSON, FINAL_TASK_WITH_EXAMPLE as FINAL_TASK_JSON_WITH_EXAMPLE
from simple_llm_context import LLMContext

ADD_EXAMPLE = False
def set_add_example(should_add: bool):
    global ADD_EXAMPLE
    ADD_EXAMPLE = should_add

def has_examples() -> bool:
    return ADD_EXAMPLE

def generate_prompt_for_json(llm_model: LLMPricingAgent, market_history: MarketHistory, context: LLMContext) -> str:
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
    full_prompt += FINAL_TASK_JSON_WITH_EXAMPLE if ADD_EXAMPLE else FINAL_TASK_JSON

    get_logger().debug('Built Promt:')
    get_logger().debug(full_prompt)

    return full_prompt

json_regex_finder = regex.compile(r'\{(?:[^{}]|(?R))*\}')

def output_json_parser(prev_context: LLMContext, result: str) -> Tuple[float, LLMContext]:
    get_logger().debug('Returned result')
    get_logger().debug(result)

    optional_results = json_regex_finder.findall(result)
    assert len(optional_results) > 0, 'Agent didn\'t attach any json'

    for option in optional_results:
        get_logger().debug('Trying to analyaze:\n%s' % option)
        plans = None
        insights = None
        price = None
        option = option.strip()
        my_price_location = option.find('my_price')
        if my_price_location == -1:
            get_logger().debug('Couldn\'t find price')
            continue

        option = option[:my_price_location] + option[my_price_location:].replace('$','')

        try:
            parsed_result = json.loads(option)
            plans = parsed_result['plans.txt']
            insights = parsed_result['insights.txt']
            price = parsed_result['my_price']

            assert type(plans) == str
            assert type(insights) == str
            assert type(price) == int or type(price) == float

            break
        except Exception:
            plans = None
            insights = None
            price = None
            get_logger().debug('Failed parsing json')


    assert plans != None, "Couldn't parse any plans"
    assert insights != None, "Couldn't parse any insights"
    assert price != None, "Couldn't parse any price"

    get_logger().debug('Parsed plan:')
    get_logger().debug(plans)
    get_logger().debug('Parsed insights:')
    get_logger().debug(insights)
    get_logger().debug('Parsed price string:')
    get_logger().debug(price)

    return price, replace(prev_context, plans=plans,insights=insights)