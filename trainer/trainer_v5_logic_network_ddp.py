## gatum부분 추가

import sys
sys.path.append('/data3/KJE/user2/code/WIL_DeepLearningProject_2/NS2')

import torch
from torch.utils.data import DataLoader,DistributedSampler,Subset
import json
from torchvision import transforms
import matplotlib.pyplot as plt
import utils.okvqa_data as data
import utils.evaluation
from options import Options
from torch.nn.parallel import DistributedDataParallel as DDP
import transformers

import os
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,LlavaNextProcessor, LlavaNextForConditionalGeneration
from pathlib import Path
import prompt

import numpy as np
import logging
import utils.util as util

import warnings

# Suppress UserWarning
warnings.filterwarnings("ignore", category=UserWarning)


def main():

    torch.distributed.init_process_group(backend='nccl')


    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    transformers.logging.set_verbosity_error()
    options = Options()
    options.add_reader_options()
    options.add_model_options()
    opt = options.parse()
    opt.is_main = True
    opt.is_distributed = False
    logger = util.init_logger(opt.is_main, opt.is_distributed, Path(opt.logger_path) / opt.name / 'run.log')

    options.print_options(opt)

    exactmatch = []

    debug = False

    # 데이터셋 불러오기

    train_examples = data.load_data(
        opt.data_path,
        # global_rank=opt.global_rank,
        # world_size=opt.world_size,
    )

    dataset = data.trainerV3_Dataset(
        data=train_examples,
        coco_path=opt.image_path,
        object_path=opt.object_path,
    )


    # 데이터로더 초기화
    dataloader = DataLoader(dataset, batch_size=opt.per_gpu_batch_size, sampler=DistributedSampler(dataset), shuffle=False,
                            collate_fn=data.trainer_v3collate_fn)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    # Model and processor initialization
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", cache_dir=opt.cache_dir)
    vlm_model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",
                                                                  torch_dtype=torch.float16,
                                                                  use_flash_attention_2=True, cache_dir=opt.cache_dir)

    vlm_model = vlm_model.to(local_rank)
    vlm_model = DDP(vlm_model, device_ids=[local_rank])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16)

    llm_model = AutoModelForCausalLM.from_pretrained(
        opt.llm_name,
        quantization_config=bnb_config,
        device_map='auto',
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        cache_dir=opt.cache_dir
    )

    logic_tokenizer = AutoTokenizer.from_pretrained(opt.llm4logic_name, cache_dir=opt.cache_dir,
                                                    token=opt.hg_token)

    logic_tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    logic_tokenizer.padding_side = "left"  # Allow batched inference

    llm_tokenizer = AutoTokenizer.from_pretrained(opt.llm_name, cache_dir=opt.cache_dir,
                                                  token=opt.hg_token)

    llm_tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    llm_tokenizer.padding_side = "left"  # Allow batched inference

    # logic_model = PeftModel.from_pretrained(llm4logic_model, opt.peft_model_id, torch_dtype=torch.float16)

    all_predict_data = {}
    answer_rationle_data = {}
    correct_count = 0
    total_count = 0
    max_i = 10
    last_answer = None
    answer = ' '
    idx = 0

    ########################################################################

    # Process each batch
    for batch in tqdm(dataloader):
        ans_premises = []

        images,image_path, question, _,_,_,target = batch

        p1 = prompt.VLM_PROMPT_FORMAT_ANSWER.format(question=question[0])
        vlm_inputs = processor(p1, images[0], return_tensors="pt")
        vlm_inputs = {k: v.to(local_rank) for k, v in vlm_inputs.items()}

        output = vlm_model.module.generate(**vlm_inputs, max_new_tokens=100)
        answer_candidates = processor.decode(output[0], skip_special_tokens=True)
        answer_candidates = util.extract_answer_candidate(answer_candidates)
        answer_candidates = util.extract_elements(answer_candidates)
        answer_candidates = list(set(answer_candidates))

        p2 = prompt.VLM_PROMPT_FORMAT_CAPTION

        inputs = processor(p2, images[0], return_tensors="pt")
        inputs = {k: v.to(local_rank) for k, v in inputs.items()}
        output = vlm_model.module.generate(**inputs, max_new_tokens=250)
        output = processor.decode(output[0], skip_special_tokens=True)
        img_context = util.extract_answer_candidate(output)
        question = question[0]
        answer_score_dic = {}
        for ans in answer_candidates:
            answer_score_dic[ans] = 0
            answer = ans
            p2 = prompt.LLM_PROMPT_RULE_GENERATION.format(question=question, answer=answer)

            input_tokens = llm_tokenizer(
                p2,
                return_tensors="pt")["input_ids"].to("cuda")

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
            generated_text = llm_tokenizer.decode(generation_output[0], skip_special_tokens=True)
            generated_text = util.extract_elements(generated_text[len(p2):])
            ans_premises.append(generated_text)

            for premise in generated_text:
                p3 = prompt.VLM_PROMPT_FORMAT_SCORE.format(information=premise)
                inputs = processor(p3, images[0], return_tensors="pt")
                inputs = {k: v.to(local_rank) for k, v in inputs.items()}
                output = vlm_model.module.generate(**inputs, max_new_tokens=100)
                output = processor.decode(output[0], skip_special_tokens=True)
                out = util.extract_answer_candidate(output)
                answer_score_dic[ans] += util.get_score(out)

        try:
            final_answer = max(answer_score_dic, key=answer_score_dic.get)

        except:
            final_answer = 'none'

        score = util.ScoreEvaluation(final_answer, target[0], len(train_examples), idx,logger)  # answer,answer_list,dataset_length,index

        # if final_answer.lower() == target[0]:
        #     score = 1
        # else:
        #     score = 0
        exactmatch.append(score)

        idx += 1
        if idx % 10 == 0:
            logger.warning(
                f'Process rank: {0}, total: {idx} test examples | accuracy = {100 * np.mean(exactmatch):.2f}%')

        all_predict_data = {
            "imagepath": image_path,
            "question": question,
            "pre_answer": answer_candidates,
            "answer_premise": ans_premises,
            "score4premise": answer_score_dic,
            "gt": target,
            "final_answer": final_answer,
            "score4answer": score,

        }

        all_predict_data_path = os.path.join(opt.logger_path,opt.name, f'okvqa_predict_ddp_v2_.json')

        with open(all_predict_data_path, 'a', encoding='utf-8') as f:
            json.dump(all_predict_data, f, ensure_ascii=False, indent=4)
            f.write(',\n')




if __name__ == "__main__":
    main()