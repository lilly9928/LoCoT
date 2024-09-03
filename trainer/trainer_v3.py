##llava -> rationle data

import sys
sys.path.append('/home/user2/code/WIL_DeepLearningProject_2/NS2')



import torch
from torch.utils.data import DataLoader
import json
from torchvision import transforms
import matplotlib.pyplot as plt
import okvqa_data as data
from options import Options
from torch.nn import DataParallel
import transformers
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import re
import os
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,LlavaNextProcessor, LlavaNextForConditionalGeneration,Blip2Processor,Blip2ForConditionalGeneration
import prompt

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


def normalize_string(input_string: str) -> str:
    # 모든 대문자를 소문자로 변환
    input_string = input_string.lower()

    # 모든 빈칸 제거
    input_string = re.sub(r'\s+', '', input_string)

    return input_string


def calculate_accuracy(predictions, targets):
    targets = targets.split(' </s>')[0]
    targets = normalize_string(targets)
    predictions = normalize_string(predictions)
    return predictions == targets


import re


def extract_last_category_list(text):
    # Use regex to extract the last occurrence of 'Categorys: ...'
    match = re.findall(r'Categorys:\s*\[.*?\]', text, re.DOTALL)

    # Get the last match and extract the list of words
    if match:
        last_category_str = match[-1]
        # Extract the list of words inside the brackets
        categories = re.findall(r'\[\'(.*?)\'\]', last_category_str)
        if categories:
            return categories[0].split("', '")
    return None


def extract_conclusion(text):
    # Use regex to extract the conclusion after 'Conclusion: '
    match = re.search(r'Conclusion:\s*(.*)#END#', text, re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        return None


def extract_answer(text):
    # Find the start index of "Answer:"
    start_index = text.find("Answer:")

    if start_index != -1:
        # Extract the part of the string after "Answer:"
        answer_part = text[start_index + len("Answer:"):].strip()

        # Extract the answer before any additional explanation
        end_index = answer_part.find("(")
        if end_index != -1:
            answer = answer_part[:end_index].strip()
        else:
            answer = answer_part

        return answer
    return None


if __name__ == "__main__":

    transformers.logging.set_verbosity_error()
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()

    opt.cache_dir = "/data2/KJE/hg_weight"


    # 기본 설정
    image_path = '/data2/KJE/VQA/COCO/images'
    data_path = '/data2/KJE/ModelLogs/revive/processed_data/test.pkl'
    object_path ='/data2/KJE/ModelLogs/Logic_LLama/data/preprocess_okvqa/okvqa_test_object.json'
    cache_dir = "/data2/KJE/hg_weight"

    vlm_name = "Salesforce/blip2-flan-t5-xxl"
    llm4logic_name = "mistralai/Mistral-7B-Instruct-v0.2"
    llm_name = "mistralai/Mistral-7B-Instruct-v0.2"
    peft_model_id = "/data2/KJE/ModelLogs/Logic_LLama/checkpoints/logic_distillation/lora_Mistral-7B-Instruct-v0.2_lr2e-4_epoch1_bs16_len512_wp0.05_lora16_8_quan_chat/checkpoint-500"

    hg_token = "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"
    batch_size = 1


    # 데이터셋 불러오기

    train_examples = data.load_data(
        data_path,
        # global_rank=opt.global_rank,
        # world_size=opt.world_size,
    )

    dataset = data.trainerV3_Dataset(
        data=train_examples,
        coco_path=image_path,
        object_path=object_path,
    )

    # 데이터로더 초기화
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,collate_fn=data.trainer_v3collate_fn)

    # Model and processor initialization
    processor = Blip2Processor.from_pretrained(vlm_name, cache_dir=cache_dir)
    vlm_model = Blip2ForConditionalGeneration.from_pretrained(vlm_name, device_map="cuda", cache_dir=cache_dir)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16)

    llm4logic_model = AutoModelForCausalLM.from_pretrained(
        llm4logic_name,
        quantization_config=bnb_config,
        device_map='cuda',
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir
    )

    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_name,
        quantization_config=bnb_config,
        device_map='cuda',
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir
    )

    logic_tokenizer = AutoTokenizer.from_pretrained(llm4logic_name, cache_dir=cache_dir,
                                              token=hg_token)

    logic_tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    logic_tokenizer.padding_side = "left"  # Allow batched inference


    llm_tokenizer = AutoTokenizer.from_pretrained(llm_name, cache_dir=cache_dir,
                                              token=hg_token)

    llm_tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    llm_tokenizer.padding_side = "left"  # Allow batched inference

    logic_model = PeftModel.from_pretrained(llm4logic_model, peft_model_id, torch_dtype=torch.float16)

    data = []
    error_data= []
    correct_count = 0
    total_count = 0
    max_i = 10
    last_answer = None
    answer = ' '
    # Process each batch
    for batch in tqdm(dataloader):
        images, question,caption, pred_classes, crop_images, target = batch
        object_list=[]

        iter_count = 0

        # while last_answer != answer:  # Ensure the while loop is properly nested
        vlm_inputs = processor(images, prompt.concept_prompt(pred_classes, question, object_list),
                               return_tensors="pt").to("cuda", torch.float16)
        out = vlm_model.generate(**vlm_inputs)
        pred_object = processor.decode(out[0], skip_special_tokens=True)
        object_list.append(pred_object)
        vlm_inputs = processor(images, prompt.concept_caption_prompt(pred_object), return_tensors="pt").to(
            "cuda", torch.float16)
        out = vlm_model.generate(**vlm_inputs)
        pred_object_caption = processor.decode(out[0], skip_special_tokens=True)

        topCat_tokens = logic_tokenizer(prompt.extract_topCat_prompt(pred_object), return_tensors="pt")[
            "input_ids"].to("cuda")
        with torch.cuda.amp.autocast():
            generation_output = llm_model.generate(
                input_ids=topCat_tokens,
                max_new_tokens=200,
                do_sample=True,
                top_k=10,
                top_p=0.9,
                temperature=0.3,
                repetition_penalty=1.15,
                num_return_sequences=1,
                eos_token_id=logic_tokenizer.eos_token_id,
            )
        topCat = logic_tokenizer.decode(generation_output[0], skip_special_tokens=True)
        topCat_clean = extract_last_category_list(topCat)

        input_tokens = logic_tokenizer(prompt.logic_prompt(topCat_clean), return_tensors="pt")["input_ids"].to(
            "cuda")
        with torch.cuda.amp.autocast():
            generation_output = logic_model.generate(
                input_ids=input_tokens,
                max_new_tokens=300,
                do_sample=True,
                top_k=10,
                top_p=0.9,
                temperature=0.3,
                repetition_penalty=1.15,
                num_return_sequences=1,
                eos_token_id=logic_tokenizer.eos_token_id,
            )
        logic_result = logic_tokenizer.decode(generation_output[0], skip_special_tokens=True)
        logic_result_clean = extract_conclusion(logic_result)

        input_tokens = llm_tokenizer(
            prompt.final_input_prompt(caption[0], pred_object_caption, logic_result_clean, question),
            return_tensors="pt")["input_ids"].to("cuda")
        # if iter_count == 0:
        #     input_tokens = llm_tokenizer(
        #         prompt.final_input_prompt(caption[0], pred_object_caption, logic_result_clean, question),
        #         return_tensors="pt")["input_ids"].to("cuda")
        # else:
        #     input_tokens = llm_tokenizer(
        #         prompt.final_input_prompt(caption[0], save_pred_object_caption + pred_object_caption,
        #                                   save_logic_result_clean + logic_result_clean, question),
        #         return_tensors="pt")["input_ids"].to("cuda")

        with torch.cuda.amp.autocast():
            generation_output = llm_model.generate(
                input_ids=input_tokens,
                max_new_tokens=300,
                do_sample=True,
                top_k=10,
                top_p=0.9,
                temperature=0.3,
                repetition_penalty=1.15,
                num_return_sequences=1,
                eos_token_id=llm_tokenizer.eos_token_id,
            )
        final = llm_tokenizer.decode(generation_output[0], skip_special_tokens=True)

        answer = extract_answer(final)

        save_pred_object_caption = pred_object_caption
        save_logic_result_clean = logic_result_clean
        last_answer = answer
        # iter_count += 1

        data.append({
            "question": question,
            "answer": answer,
            "target": target,
        })

save_path = '/data3/KJE/ModelLogs/Logic_LLama/'

with open(os.path.join(save_path,f'results_trainerv3.json'), 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
