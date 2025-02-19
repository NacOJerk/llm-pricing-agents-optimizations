from accelerate.test_utils.testing import get_backend
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.commands.chat import RichInterface
import torch
from threading import Thread

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

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

    interface = RichInterface(model_name=MODEL_NAME, user_name="hey")
    interface.clear()
    chat = []
    chat.append({"role": "user", "content": "Hello there handsom"})

    inputs = tokenizer.apply_chat_template(chat, return_tensors="pt", add_generation_prompt=True).to(
        model.device
    )
    attention_mask = torch.ones_like(inputs)
    generation_kwargs = {
        "inputs": inputs,
        "attention_mask": attention_mask,
        "streamer": generation_streamer,
        "max_new_tokens": 250,
        "do_sample": True,
        "num_beams": 1,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_ids,
    }
    # user_input = interface.input()
    # print(user_input)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    model_output = interface.stream_output(generation_streamer)
    thread.join()
    chat.append({"role": "assistant", "content": model_output})

if __name__ == "__main__":
    main()