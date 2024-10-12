from config import CFG
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

def load_model(model_name = CFG.model_name):
    if model_name == 'Qwen2.5':
        model_repo = 'Qwen/Qwen2.5-0.5B-Instruct'
        
        tokenizer = AutoTokenizer.from_pretrained(model_repo)    

        model = AutoModelForCausalLM.from_pretrained(
            model_repo,
            device_map = 'auto',
            low_cpu_mem_usage = True
        )
        
        max_len = 512

    else:
        raise NotImplementedError("Not implemented model (tokenizer and backbone)")

    return tokenizer, model, max_len
