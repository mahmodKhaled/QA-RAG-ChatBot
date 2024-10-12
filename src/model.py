from config import CFG
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from typing import Tuple
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

def load_model(
    model_name: str= CFG.model_name
) -> Tuple[AutoTokenizer, AutoModelForCausalLM, int]:
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

def create_llm(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_len: int,
    config: CFG
) -> HuggingFacePipeline:
    pipe = pipeline(
        task = "text-generation",
        model = model,
        tokenizer = tokenizer,
        pad_token_id = tokenizer.eos_token_id,
        max_new_tokens= max_len,
        do_sample = False,
        top_p = config.top_p,
        repetition_penalty = config.repetition_penalty
    )

    llm = HuggingFacePipeline(pipeline = pipe)

    return llm
