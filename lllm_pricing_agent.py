from typing import Any, Callable, Tuple
from market_history import MarketHistory
from pricing_agent import PricingAgent

PromtContext = Any
TextGenerator = Callable[[str], str]
PromtGenerator = Callable[['LLLMPricingAgent', MarketHistory, PromtContext], str]
OutputParser = Callable[[str], Tuple[float, PromtContext]]

class LLLMPricingAgent(PricingAgent):
    def __init__(self, firm_id: float, price_per_unit: float,
                text_generator: TextGenerator,
                promt_generator: PromtGenerator,
                output_parser: OutputParser,
                initial_context: PromtContext = None):
        super().__init__(firm_id, price_per_unit)
        self.text_generator: TextGenerator = text_generator
        self.promt_generator: PromtGenerator = promt_generator
        self.output_parser: OutputParser = output_parser
        self.context: PromtContext = initial_context
    
    def generate_price(self, market_history: MarketHistory) -> float:
        generated_promt = self.promt_generator(self, market_history, self.context)
        llm_output = self.text_generator(generated_promt)
        new_price, new_context = self.output_parser(llm_output) # TODO: Allow throwing promt generation exception
        self.context = new_context
        return new_price

