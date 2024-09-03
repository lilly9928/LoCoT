import sys
sys.path.append('/home/user2/code/WIL_DeepLearningProject_2/NS2')

import os
import torch
from torch.utils.data import DataLoader, DistributedSampler
import json
from torchvision import transforms
import okvqa_data as data_module  # 모듈 이름을 변경하여 전역 변수와 충돌 방지
from options import Options
from torch.nn.parallel import DistributedDataParallel as DDP
import transformers
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration,BitsAndBytesConfig,AutoTokenizer,AutoModelForCausalLM
import re
from tqdm import tqdm

def extract_question_answer(text):
    # question_match = re.search(r'question:\s*(.*)\s*', text)
    # answer_match = re.search(r'(?<=\[/INST\] answer: ).*', text)
    question_match = re.search(r'/^question:/g', text)
    answer_match = re.search(r'/^answer:/g', text)
    # return text,text,text

    if question_match and answer_match :
        question = question_match.group(1)
        answer = answer_match.group(1)
        return question, answer
    elif question_match:
        question = question_match.group(1)
        return question, text
    else:
        return None, None, None

def normalize_string(input_string: str) -> str:
    # 모든 대문자를 소문자로 변환
    input_string = input_string.lower()
    
    # 모든 빈칸 제거
    input_string = re.sub(r'\s+', '', input_string)
    
    return input_string

def calculate_accuracy(predictions, targets):
    targets = targets[0].split(' </s>')[0]
    targets= normalize_string(targets)
    predictions = normalize_string(predictions)
    return predictions == targets

def main():
    # Initialize the process group
    torch.distributed.init_process_group(backend='nccl')

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    transformers.logging.set_verbosity_error()
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()
    opt.cache_dir = "/data2/KJE/hg_weight"

    image_path = '/data2/KJE/VQA/COCO/images'
    data_path = '/data2/KJE/ModelLogs/revive/processed_data/train.pkl'
    n_ex_context = 40
    batch_size = 1

    LLM_model = "mistralai/Mistral-7B-Instruct-v0.2"
    VLM_model = "llava-hf/llava-v1.6-mistral-7b-hf"


    train_examples = data_module.load_data(data_path)
    dataset = data_module.FOL_2Dataset(data=train_examples, image_path=image_path, n_ex_context=n_ex_context)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler, collate_fn=data_module.collate_fn)

    processor = LlavaNextProcessor.from_pretrained(VLM_model, cache_dir=opt.cache_dir)
    vlm = LlavaNextForConditionalGeneration.from_pretrained(VLM_model,
                                                              torch_dtype=torch.float16, low_cpu_mem_usage=True,
                                                             cache_dir=opt.cache_dir)
    vlm = vlm.to(local_rank)
    vlm = DDP(vlm, device_ids=[local_rank])



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
    error_data = []
    correct_count = 0
    total_count = 0

    for batch in tqdm(dataloader):
        inputs, image, question, targets = batch
        inputs = {k: v.to(local_rank) for k, v in inputs.items()}

        outputs = vlm.module.generate(**inputs, max_new_tokens=200)
        results = [processor.decode(output, skip_special_tokens=True) for output in outputs]
        fol_output = results[0]

        # print(fol_output)


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

        answer = op.split('answer:')[-1]

        
        if answer: 
            calculate_accuracy(answer,targets)
            correct_count += 1
            results_data.append({
                "question":question[0],
                "answer": answer,
                "targets": targets
            })
             

        else:
            error_data.append({
                    "question":question[0],
                    "output": op,
                    "targets": targets,
                })
        total_count += 1

    accuracy = correct_count / total_count * 100 if total_count > 0 else 0


    save_path = '/data2/KJE/ModelLogs/Logic_LLama/llava-v1.6-mistral-7b_orignal'

    with open(os.path.join(save_path, f'okvqa_test_results_in_fol_acc_{accuracy}.json'), 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=4)

    with open(os.path.join(save_path, 'okvqa_test_in_fol_error.json'), 'w', encoding='utf-8') as f:
        json.dump(error_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
