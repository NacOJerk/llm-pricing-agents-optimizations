from typing import Any, Callable, Tuple

from logger import get_logger
from market_history import MarketHistory
from pricing_agent import PricingAgent

PromtContext = Any
TextGenerator = Callable[[str], str]
PromtGenerator = Callable[['LLMPricingAgent', MarketHistory, PromtContext], str]
OutputParser = Callable[[PromtContext, str], Tuple[float, PromtContext]]
LLM_RETRY_COUNT = 10

class LLMPricingAgent(PricingAgent):
    def __init__(self, firm_id: int, price_per_unit: float,
                text_generator: TextGenerator,
                promt_generator: PromtGenerator,
                output_parser: OutputParser,
                add_tooling: bool,
                initial_context: PromtContext = None):
        super().__init__(firm_id, price_per_unit)
        self.text_generator: TextGenerator = text_generator
        self.promt_generator: PromtGenerator = promt_generator
        self.output_parser: OutputParser = output_parser
        self.context: PromtContext = initial_context
        self.total_exceptions = 0
        self.add_tooling = add_tooling
    
    def generate_price(self, market_history: MarketHistory) -> float:
        generated_promt = self.promt_generator(self, market_history, self.context)
        raw_data = [{'price': priced_products.priced_products[0].price,
                     'quanity_sold': priced_products.priced_products[0].quantity_sold,
                     'profit': priced_products.priced_products[0].profit} for priced_products in market_history.past_iteration]
        if self.add_tooling:
                generated_promt += """You have a list of dictionaries with the following struct:
                {'price': X, 'quanity_sold': X, 'profit': X}
                in a variable named market_history.
                """
        for i in range(LLM_RETRY_COUNT):
            try:
                addit_kwargs = {}
                if self.add_tooling:
                    addit_kwargs['local_varaibles'] = {'market_history': raw_data}
                llm_output = self.text_generator(generated_promt, **addit_kwargs)
                new_price, new_context = self.output_parser(self.context, llm_output)
                break
            except Exception:
                self.total_exceptions += 1
                get_logger().warning('Failed retrying (Current attempt: %d)' % (i+1))
                if i == (LLM_RETRY_COUNT - 1):
                    get_logger().error('To many failures, quiting experiment')
                    raise
                else:
                    get_logger().exception('Exception was:')
        self.context = new_context
        return new_price

