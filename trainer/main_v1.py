import transformers
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

model_id = "meta-llama/Meta-Llama-3-70B"
hg_token ='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'
cache_dir = "/data2/KJE/hg_weight"

tokenizer = AutoTokenizer.from_pretrained(model_id, token=hg_token,cache_dir=cache_dir,torch_dtype=torch.float16,)
model = AutoModelForCausalLM.from_pretrained(model_id,token=hg_token,cache_dir=cache_dir)

