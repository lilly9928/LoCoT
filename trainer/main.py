##llava -> rationle data 

import sys
sys.path.append('/home/user2/code/WIL_DeepLearningProject_2/NS2')



import os
import torch
from torch.utils.data import DataLoader, DistributedSampler
import json
from torchvision import transforms
import okvqa_data as data # 모듈 이름을 변경하여 전역 변수와 충돌 방지
from options import Options
from torch.nn.parallel import DistributedDataParallel as DDP
import transformers
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration,BitsAndBytesConfig,AutoTokenizer,AutoModelForCausalLM
import re
from tqdm import tqdm

def extract_question_answer_rationale(text):
    # Use regular expressions to extract the question, answer, and rationale
    question_match = re.search(r'question:\s*(.*)\s*\[\s*/INST\]', text)
    answer_match = re.search(r'Answer:\s*(.*)\s*,\s*Rationale:', text)
    rationale_match = re.search(r'Rationale:\s*(.*)\s*$', text)

    if question_match and answer_match and rationale_match:
        question = question_match.group(1)
        answer = answer_match.group(1)
        rationale = rationale_match.group(1)
        return question, answer, rationale
    elif question_match:
        question = question_match.group(1)
        return question, text, None
    else:
        return None, None, None

def calculate_accuracy(predictions, targets):
    # 비교를 위한 함수, 여기서는 단순하게 문자열 비교를 사용합니다.
    return predictions == targets



if __name__ == "__main__":

    transformers.logging.set_verbosity_error()
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()
    opt.cache_dir = "/data2/KJE/hg_weight"



    # 기본 설정
    image_path = '/data2/KJE/VQA/COCO/images'
    data_path = '/data2/KJE/ModelLogs/revive/processed_data/train.pkl'
    n_ex_context = 40
    batch_size = 1


    LLM_model = "mistralai/Mistral-7B-Instruct-v0.2"
    VLM_model = "llava-hf/llava-v1.6-mistral-7b-hf"


    train_examples = data.load_data(
        data_path,
    )

    dataset = data.FOL_2Dataset(
        data=train_examples,
        image_path=image_path,
        n_ex_context=n_ex_context,
    )

    # 데이터로더 초기화
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,collate_fn=data.collate_fn)

    # Model and processor initialization
    processor = LlavaNextProcessor.from_pretrained(VLM_model, cache_dir=opt.cache_dir)
    vlm = LlavaNextForConditionalGeneration.from_pretrained(VLM_model,
                                                              torch_dtype=torch.float16, low_cpu_mem_usage=True,
                                                             cache_dir=opt.cache_dir)
    vlm.to('cuda')

    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)

    llm = AutoModelForCausalLM.from_pretrained(
        LLM_model,
        quantization_config=bnb_config,
        device_map='auto',
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        cache_dir=opt.cache_dir 
    )

    tokenizer = AutoTokenizer.from_pretrained(LLM_model,cache_dir=opt.cache_dir ,token='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')
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    

    results_data = []
    error_data= []
    correct_count = 0
    total_count = 0

    # Process each batch
    for batch in tqdm(dataloader):
        inputs, image, question, targets = batch
        inputs.to('cuda')

        outputs = vlm.generate(**inputs, max_new_tokens=200)
        results = processor.decode(outputs[0], skip_special_tokens=True) 
        fol_output = results

        print(fol_output)


        llm_prompt = f""" 
        [INST]<image> Given information answer the question in short answer. \n\n {question[0]} \n\n information: {fol_output.split('[/INST]')[1]}\n .
        \n\nExamples:
        Answer: Apple 
        [/INST]
        """

        input_tokens = tokenizer(llm_prompt, return_tensors="pt")["input_ids"].to("cuda")

        with torch.cuda.amp.autocast():
            generation_output = llm.generate(
                input_ids=input_tokens,
                max_new_tokens=300,
                do_sample=True,
                top_k=10,
                top_p=0.9,
                temperature=0.3,
                repetition_penalty=1.15,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
            )
        op = tokenizer.decode(generation_output[0], skip_special_tokens=True)
        print(op)

        break



# save_path = '/data2/KJE/ModelLogs/Logic_LLama/llava-v1.6-mistral-7b_orignal'



# with open(os.path.join(save_path,f'results_acc_{accuracy}.json'), 'w', encoding='utf-8') as f:
#     json.dump(data, f, ensure_ascii=False, indent=4)

# with open(os.path.join(save_path,'error.json'), 'w', encoding='utf-8')as f:
#     json.dump(error_data, f, ensure_ascii=False, indent=4)