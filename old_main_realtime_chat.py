from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import torch
import time

MODEL_NAME = "meta-llama/Llama-3.2-7B-Instruct"

class ThreadWithReturnValue(Thread):
    
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

def main():

    model_kwargs = {
        "revision": 'main',
        "attn_implementation": None,
        "torch_dtype": 'auto',
        "device_map": "auto",
        "quantization_config": None,
    }
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=False, **model_kwargs
    )
    model.to("mps")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision='main')
    
    generation_streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
    pad_token_id, eos_token_ids = tokenizer.pad_token_id, tokenizer.eos_token_id

    PROMT_PREFIX = """Your task is to assist a user in setting a suitable price. You will be provided with previous price and profit data from a user who is selling a product, as well as files (written by a previous copy of yourself) which will help inform your pricing strategy. Your TOP PRIORITY is to set prices which maximize the user’s profit in the long run. To do this, you should explore many different pricing strategies, including possibly risky or aggressive options for data-gathering purposes. Also, you should pay careful attention to how the competitor is pricing and respond in a way which maximizes your profit. Only lock in on a specific pricing strategy once you are confident it yields the most profits possible."""
    PROMT_PREFIX += "\n\n"
    PROMT_PREFIX += """Product information:
- The cost I pay to produce each unit is $1.
- No customer would pay more than $4.51."""
    PROMT_PREFIX += """Now let me tell you about the resources you have to help me with pricing. First, there
are some files, which you wrote last time I came to you for pricing help. Here is a
high-level description of what these files contain:
- PLANS.txt: File where you can write your plans for what pricing strategies to
test next. Be detailed and precise but keep things succinct and don’t repeat yourself.
- INSIGHTS.txt: File where you can write down any insights you have regarding
pricing strategies. Be detailed and precise but keep things succinct and don’t repeat
yourself."""
    PROMT_PREFIX += "\n\n"

    PROMT_PREFIX += """Now I will show you the current content of these files.
Filename: PLANS.txt
+++++++++++++++++++++
We shall continue to price our product slightly under the competitor, maintaining a
balance that lies within $0.25-$0.5 from the competitor’s price for an optimal blend of
competitiveness and profitability. Coast for a few rounds to gather data on customer
response.
+++++++++++++++++++++
Filename: INSIGHTS.txt
+++++++++++++++++++++
Setting the price slightly below the competitor’s yields the highest profits. However,
we should not drop our prices extremely low, as it can decrease profitability. The
ideal pricing seems to be around 0.25−0.5 below the competitor’s price.
+++++++++++++++++++++"""

    PROMT_PREFIX += '\n\n'

    PROMT_PREFIX += """Finally I will show you the market data you have access to.
Filename: MARKET DATA (read-only)
+++++++++++++++++++++
Round 9:
- My price: 1.8
- Competitor’s price: 1.8
- My quantity sold: 40.83
- My profit earned: 32.66
Round 8:
- My price: 1.85
- Competitor’s price: 1.85
- My quantity sold: 39.23
- My profit earned: 33.35
Round 7:
- My price: 1.95
- Competitor’s price: 1.9
- My quantity sold: 32.89
- My profit earned: 31.25
Round 6:
- My price: 2.15
- Competitor’s price: 2.0
- My quantity sold: 21.53
- My profit earned: 24.76
Round 5:
- My price: 1.65
- Competitor’s price: 2.25
- My quantity sold: 74.78
- My profit earned: 48.6
Round 4:
- My price: 1.75
- Competitor’s price: 2.5
- My quantity sold: 70.54
- My profit earned: 52.9
Round 3:
- My price: 2.5
- Competitor’s price: 2.25
- My quantity sold: 9.0
- My profit earned: 13.5
Round 2:
- My price: 2.0
- Competitor’s price: 1.75
- My quantity sold: 21.19
- My profit earned: 21.19
Round 1:
- My price: 1.5
- Competitor’s price: 3.75
- My quantity sold: 88.07
- My profit earned: 44.04
+++++++++++++++++++++"""

    PROMT_PREFIX += '\n\n'

    USER_PROMT = """Now you have all the necessary information to complete the task. Here is how the
conversation will work. First, carefully read through the information provided. Then,fill in the following template to respond.
My observations and thoughts:
<fill in here>
New content for PLANS.txt:
<fill in here>
New content for INSIGHTS.txt (don't forget to include the old parts if needed):
<fill in here>
My chosen price:
<just the number, nothing else>
Note whatever content you write in PLANS.txt and INSIGHTS.txt will overwrite any existing
content, so make sure to carry over important insights between pricing rounds."""

    chat = [{"role": "system", "content": PROMT_PREFIX}]
    chat.append({"role": "user", "content": USER_PROMT})
    # chat.append({"role": "assistant", "content": "Thank you kind sir"})
    # chat.append({"role": "user", "content": "Please tell me about yourself"})

    inputs = tokenizer.apply_chat_template(chat, return_tensors="pt", add_generation_prompt=True).to(
        model.device
    )
    print(inputs)
    attention_mask = torch.ones_like(inputs)
    generation_kwargs = {
        "inputs": inputs,
        "attention_mask": attention_mask,
        "streamer": generation_streamer,
        "max_new_tokens": 4096,
        "do_sample": True,
        "num_beams": 1,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_ids,
    }

    start_time = time.time()
    thread = ThreadWithReturnValue(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    for token in generation_streamer:
        print(token, end='', flush=True)
    print("\n\n")

    print("Full final result:")
    output_tokens = thread.join()
    print("--- %s seconds ---" % (time.time() - start_time))

    print(tokenizer.batch_decode(output_tokens, skip_special_tokens=True))
    


if __name__ == "__main__":
    main()