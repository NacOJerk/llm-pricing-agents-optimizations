import json
from together import Together
import traceback
from typing import List
import regex

from logger import get_logger

CHOSEN_MODEL = None

TOOLING_PROMPT = """You are an experienced market analyst. Whenever you wish to calculate something, you may output:
{"expr": "<math expr>"}
And you will wait to receive back the result in the following format:
{"result: "<math expr result>"}
The math syntax uses Python syntax.

For example:
{"expr": "5 + 2**4"}
<You wait>
{"result: "21"}

You will try to execute utilizing your tools whenever you need to.
You are not good with math, so please don't perform any calculations yourself or try to evaluate expressions by yourself.
The real user is unaware of this syntax, so your final answers can never contain the expr.
You may only submit one expr each time.
"""
USE_TOOLING = False
json_regex_finder = regex.compile(r'\{(?:[^{}]|(?R))*\}')

def set_using_tooling(use_tooling: bool):
    global USE_TOOLING
    USE_TOOLING = use_tooling

def get_available_models() -> List[str]:
    return ['meta-llama/Llama-2-7b-chat-hf', 'meta-llama/Meta-Llama-3-8B-Instruct-Turbo']

def set_chosen_model(model: str):
    assert model in get_available_models()

    global CHOSEN_MODEL
    CHOSEN_MODEL = model

def get_chosen_model() -> str:
    assert CHOSEN_MODEL != None
    return CHOSEN_MODEL

expr_hit_count = 0
invalid_expr_hit_count = 0

def reset_tooling_info():
    global expr_hit_count
    global invalid_expr_hit_count
    expr_hit_count = 0
    invalid_expr_hit_count = 0

def get_tooling_info_dict() -> dict:
    return {
        'expr_hit_count': expr_hit_count,
        'invalid_expr_hit_count': invalid_expr_hit_count,
    }

client = Together()

MAX_EXPRESION_RESPONSE_SIZE = 300

def genereate_text(message, 
                   max_tokens=1500, 
                   temperature=0.7, 
                   top_p=0.7,
                   top_k=50,
                   local_varaibles={}):
    global expr_hit_count, invalid_expr_hit_count
    messages = []
    if USE_TOOLING:
        messages.append({
                    "role": "system",
                    "content": TOOLING_PROMPT
                })
    messages.append({
                    "role": "user",
                    "content": message
                })
    while True:
        response = client.chat.completions.create(
            model=CHOSEN_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=1,
        )
        agent_response = response.choices[0].message.content
        if not USE_TOOLING:
            return agent_response
        possible_requests = json_regex_finder.findall(agent_response)
        used_tool = False
        for request in possible_requests[::-1]:
            try:
                if 'expr' not in request:
                    continue
                get_logger().debug('Detected possible expression:')
                get_logger().debug('\t%s' % request)
                request_dict = json.loads(request.strip())
                if 'expr' in request_dict:
                    expr = request_dict['expr']
                    message_to_assistant = None
                    expr_hit_count += 1
                    try:
                        result = eval(expr, {}, local_varaibles)
                        message_to_assistant = json.dumps({"result": str(result)})
                    except:
                        invalid_expr_hit_count += 1
                        message_to_assistant = json.dumps({"error": traceback.format_exc()})
                    if len(message_to_assistant) > MAX_EXPRESION_RESPONSE_SIZE:
                        message_to_assistant = json.dumps({"error": "Response too long"})
                    messages.append({
                                "role": "assistant",
                                "content": agent_response
                            })
                    messages.append({
                                    "role": "user",
                                    "content": message_to_assistant
                                })
                    get_logger().debug('Assistant requested:\n%s' % agent_response)
                    get_logger().debug('Got in response:\n%s' % message_to_assistant)
                    used_tool = True
                    break
            except:
                pass
        if not used_tool:
            break
    return agent_response

def generate_specialized_text(max_tokens=None, 
                              temperature=0.7, 
                              top_p=0.7,
                              top_k=50):
    def generate_text_spec(message, local_varaibles={}):
        return genereate_text(message,
                              max_tokens=max_tokens,
                              temperature=temperature,
                              top_p=top_p,
                              top_k=top_k,
                              local_varaibles=local_varaibles)
    return generate_text_spec