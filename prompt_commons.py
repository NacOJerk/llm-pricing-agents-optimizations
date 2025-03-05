from llm_pricing_agent import LLMPricingAgent
from market_history import MarketHistory
from prompt_costs import SINGLE_MARKET_ROUND_DATA

MAX_ROUND_COUNT = 100

def set_max_round_count(count: int):
    assert count >= 0, 'Can\'t have a negative count'
    global MAX_ROUND_COUNT
    MAX_ROUND_COUNT = count

def get_max_round_count() -> int:
    return MAX_ROUND_COUNT

def generate_market_history(_: LLMPricingAgent, market_history: MarketHistory) -> str:

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