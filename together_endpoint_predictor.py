from together import Together
from typing import List

CHOSEN_MODEL = None

def get_available_models() -> List[str]:
    return ['meta-llama/Llama-2-7b-chat-hf', 'meta-llama/Meta-Llama-3-8B-Instruct-Turbo']

def set_chosen_model(model: str):
    assert model in get_available_models()

    global CHOSEN_MODEL
    CHOSEN_MODEL = model

def get_chosen_model() -> str:
    assert CHOSEN_MODEL != None
    return CHOSEN_MODEL

client = Together()

def genereate_text(message, system_prompt="", max_tokens=1500, temperature=0.7, top_p=0.7,top_k=50,repetition_penalty=1):
    messages = []
    if system_prompt:
        messages.append({
                    "role": "system",
                    "content": system_prompt
                })
    messages.append({
                    "role": "user",
                    "content": message
                })
    response = client.chat.completions.create(
        model=CHOSEN_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=1,
    )
    return response.choices[0].message.content

def generate_specialized_text(system_prompt, max_toxens=1500):
    def generate_text_spec(message):
        return genereate_text(message, system_prompt=system_prompt, max_tokens=max_toxens)
    return generate_text_spec