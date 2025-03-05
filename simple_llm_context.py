from dataclasses import dataclass

@dataclass
class LLMContext:
    cost_per_unit: float
    max_client_price: float
    plans: str
    insights: str