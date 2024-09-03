## gatum부분 추가

import sys

sys.path.append('/data3/KJE/user2/code/WIL_DeepLearningProject_2/NS2')
from openai import OpenAI
import torch
from torch.utils.data import DataLoader, DistributedSampler, Subset
import json
from torchvision import transforms
import matplotlib.pyplot as plt
import utils.okvqa_data as data
import utils.gqa_data as GQAdata
import utils.evaluation
from options import Options
from torch.nn.parallel import DistributedDataParallel as DDP
import transformers
from torch.nn.parallel import DistributedDataParallel as DDP

import os
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LlavaNextProcessor, \
    LlavaNextForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration
from pathlib import Path
import prompt2 as prompt
import datetime

import numpy as np
import logging
import utils.util as util

import warnings

# Suppress UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

from detectron2.utils.logger import setup_logger

setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


def main():
    torch.distributed.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=7200))

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

    torch.manual_seed(opt.seed)
    # 데이터셋 불러오기

    train_examples = data.load_data(
        opt.data_path,
    )

    if opt.data == 'aokvqa':
        dataset = data.AOKVQA_Dataset(
            data=train_examples,
            coco_path=opt.image_path,
        )

    elif opt.data == 'okvqa':
        dataset = data.trainerV3_Dataset(
            data=train_examples,
            coco_path=opt.image_path,
            object_path=opt.object_path,
        )

    elif opt.data == 'tdiuc':
        dataset = data.TDIUC_Dataset(
            data=train_examples,
            coco_path=opt.image_path,
        )

    elif opt.data == 'gqa':
        train_examples = GQAdata.loadFile(
            # '/data2/NS/GQA/preprocess/all_testdev_data.json'
            '/data2/NS/GQA/500_split_testdev_data.json'
        )
        dataset = GQAdata.trainerV3_Dataset(
            data=train_examples,
            image_dir='/data2/NS/GQA/images/images',
            object_path=opt.object_path,
        )

    if opt.debug:
        debug_num = 0

    # 데이터로더 초기화
    if opt.data == 'aokvqa':
        dataloader = DataLoader(dataset, batch_size=opt.per_gpu_batch_size, sampler=DistributedSampler(dataset),
                                shuffle=False,
                                collate_fn=data.aokvqa_collate_fn)
    elif opt.data == 'okvqa':
        dataloader = DataLoader(dataset, batch_size=opt.per_gpu_batch_size, sampler=DistributedSampler(dataset),
                                shuffle=False,
                                collate_fn=data.trainer_v3collate_fn)

    elif opt.data == 'tdiuc':
        dataloader = DataLoader(dataset, batch_size=opt.per_gpu_batch_size, sampler=DistributedSampler(dataset),
                                shuffle=False,
                                collate_fn=data.tdiuc_collate_fn)

    elif opt.data == 'gqa':
        dataloader = DataLoader(dataset, batch_size=opt.per_gpu_batch_size, sampler=DistributedSampler(dataset),
                                shuffle=False,
                                collate_fn=GQAdata.trainer_v3collate_fn)

    # Model and processor initialization
    if 'llava-hf' in opt.lvlm_name:
        processor = LlavaNextProcessor.from_pretrained(opt.lvlm_name, cache_dir=opt.cache_dir)
        vlm_model = LlavaNextForConditionalGeneration.from_pretrained(opt.lvlm_name,
                                                                      torch_dtype=torch.float16,
                                                                      use_flash_attention_2=True,
                                                                      cache_dir=opt.cache_dir)

    vlm_model = vlm_model.to(local_rank)
    vlm_model = DDP(vlm_model, device_ids=[local_rank])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16)

    if opt.llm_name == 'gpt-3.5-turbo':
        client = OpenAI(api_key=opt.openai_token)
    else:
        llm_model = AutoModelForCausalLM.from_pretrained(
            opt.llm_name,
            quantization_config=bnb_config,
            device_map='auto',
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            cache_dir=opt.cache_dir,
            token=opt.hg_token
        )

        llm_tokenizer = AutoTokenizer.from_pretrained(opt.llm_name, cache_dir=opt.cache_dir,
                                                      token=opt.hg_token)

        llm_tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        llm_tokenizer.padding_side = "left"  # Allow batched inference

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(opt.detection_config))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = opt.detection_weight
    predictor = DefaultPredictor(cfg)

    idx = 0

    all_predict_data = []
    predict_data = []
    ########################################################################

    # Process each batch
    for batch in tqdm(dataloader):
        if opt.data == 'aokvqa':
            images, image_path, question, choices, difficult_direct_answer, target, question_id = batch
        elif opt.data == 'okvqa':
            images, image_path, question, _, _, _, target = batch
        elif opt.data == 'gqa':
            images, image_path, question, target, type, group, fullanswer = batch
        elif opt.data == 'tdiuc':
            images, image_path, question, target, question_id, question_type = batch

        ##FastRCNN -> 이미지 속 객체 인식, 박스
        image_object, boxes = util.detect_image_ver1(image_path[0], predictor)
        boxes = boxes.tensor.cpu().tolist()

        ############################ conclsion 1 (Answer from LVLM) ##############################
        answer_candidate = choices[0]

        # c1_p = prompt.VLM_PROMPT_FORMAT_FINAL_ANSWER_onlyvlm.format(question=question[0])
        # vlm_inputs = processor(c1_p, images[0], return_tensors="pt")
        # vlm_inputs = {k: v.to(local_rank) for k, v in vlm_inputs.items()}
        # outputs = vlm_model.module.generate(**vlm_inputs, max_new_tokens=300,
        #                                     num_beams=opt.beam_num_vlm,
        #                                     num_return_sequences=opt.beam_num_vlm,
        #                                     early_stopping=True,
        #                                     no_repeat_ngram_size=opt.beam_num_vlm,
        #                                     top_p=0.99,
        #                                     temperature=0.5, )
        #
        # for output in outputs:
        #     c1 = processor.decode(output, skip_special_tokens=True)
        #     c1 = util.extract_answer_candidate(c1)
        #     answer_candidate.append(c1)

        # if opt.debug:
        #     print("############################ conclsion 1 (Answer from LVLM) ##############################\n")
        #     print(f"PROMPT: {c1_p}\n")
        #     print(f"ANSWER_CANDIDATE: {answer_candidate}\n")

        ############################ conclsion1 (Answer from LVLM) ##############################

        premise_for_answer = {}
        for answer in answer_candidate:
            premise_for_answer[answer] = []

        ########################### Object conclsion2 ################################################

        object_candidate = []
        c2_p = prompt.VLM_PROMPT_FORMAT_OBJECT_SELECT.format(object=image_object, question=question[0])
        vlm_inputs = processor(c2_p, images[0], return_tensors="pt")
        vlm_inputs = {k: v.to(local_rank) for k, v in vlm_inputs.items()}

        outputs = vlm_model.module.generate(**vlm_inputs, max_new_tokens=300,
                                            num_beams=opt.object_num,
                                            num_return_sequences=opt.object_num,
                                            early_stopping=True,
                                            no_repeat_ngram_size=opt.object_num,
                                            top_p=0.98,
                                            temperature=0.5)
        for output in outputs:
            c2 = processor.decode(output, skip_special_tokens=True)
            c2 = util.extract_answer_candidate(c2)
            object_candidate.append(c2)

        if opt.debug:
            print("########################### Object conclsion2 ################################################\n")
            print(f"PROMPT: {c2_p}\n")
            print(f"ANSWER_CANDIDATE: {object_candidate}\n")

        ########################### Object conclsion2 ################################################

        ########################### Object Premise1 ################################################
 
        crop_image = images[0]

        for answer in answer_candidate:
            p1_p = prompt.LLAVA_PROMPT_FORMAT_ANSWER_PREMISE.format(question=question[0], answer=answer,
                                                                    object=c2.lower())
            vlm_inputs = processor(p1_p, crop_image, return_tensors="pt")
            vlm_inputs = {k: v.to(local_rank) for k, v in vlm_inputs.items()}

            out = vlm_model.module.generate(**vlm_inputs, max_new_tokens=100)
            p1 = processor.decode(out[0], skip_special_tokens=True)
            p1 = util.extract_answer_candidate(p1)
            p1 = util.extract_elements(p1)

            try:
                premise_for_answer[answer] += p1

            except:
                premise_for_answer[answer] = p1

        if opt.debug:
            print(" ########################### Object Premise1 ################################################\n")
            print(f"PROMPT: {p1_p}\n")
            print(f"ANSWER_CANDIDATE: {premise_for_answer}\n")

        ########################### Object Premise1 ################################################

        ########################### LLM Premise2 ################################################
        answer_score_dic = {}
        for ans in answer_candidate:
            answer_score_dic[ans] = 0
            answer = ans
            p2 = prompt.LLM_PROMPT_RULE_GENERATION.format(question=question[0], answer=answer)

            if opt.llm_name == 'gpt-3.5-turbo':
                completion = client.chat.completions.create(
                    model=opt.llm_name,
                    messages=[
                        {"role": "user",
                         "content": p2}
                    ]
                )
                generated_text = completion.choices[0].message.content
                generated_text = util.extract_elements(generated_text)

            else:
                input_tokens = llm_tokenizer(
                    p2,
                    return_tensors="pt")["input_ids"].to("cuda")

                with torch.cuda.amp.autocast():
                    generation_output = llm_model.generate(
                        input_ids=input_tokens,
                        max_new_tokens=300,
                        do_sample=True,
                        top_k=1,
                        top_p=0.9,
                        temperature=0.3,
                        repetition_penalty=1.15,
                        num_return_sequences=1,
                        eos_token_id=llm_tokenizer.eos_token_id,
                    )
                generated_text = llm_tokenizer.decode(generation_output[0], skip_special_tokens=True)
                generated_text = util.extract_elements(generated_text[len(p2):])

            premise_for_answer[ans] += generated_text

        if opt.debug:
            print(" ########################### LLM Premise2 ################################################\n")
            print(f"PROMPT: {p2}\n")
            print(f"generated_text: {generated_text}\n")
            print(f"ANSWER_CANDIDATE: {premise_for_answer}\n")
        ########################### Scoring ################################################

        new_premise = {}

        for ans in answer_candidate:
            new_premise[ans] = []
            for premise in premise_for_answer[ans]:

                if opt.scoring_LLM:
                    p3_llm = prompt.LLM_PROMPT_FORMAT_SCORE.format(information=premise, conclusion=ans)

                    input_tokens = llm_tokenizer(p3_llm, return_tensors="pt")["input_ids"].to("cuda")

                    with torch.cuda.amp.autocast():
                        generation_output = llm_model.generate(
                            input_ids=input_tokens,
                            max_new_tokens=10,
                            do_sample=True,
                            top_k=10,
                            top_p=0.9,
                            temperature=0.3,
                            repetition_penalty=1.15,
                            num_return_sequences=1,
                            eos_token_id=llm_tokenizer.eos_token_id,
                        )
                    generated_text = llm_tokenizer.decode(generation_output[0], skip_special_tokens=True)
                    output = generated_text[len(p3_llm):]
                    score = util.get_score(output)
                    answer_score_dic[ans] += score

                    if opt.debug:
                        print(
                            "  ########################### Scoring Detail (LLM)################################################\n")
                        print(f"premise: {premise}\n")
                        print(f"score: {score}\n")

                    if score > opt.threshold:
                        new_premise[ans].append(premise)

                p3 = prompt.VLM_PROMPT_FORMAT_SCORE.format(information=premise)
                vlm_inputs = processor(p3, images[0], return_tensors="pt")
                vlm_inputs = {k: v.to(local_rank) for k, v in vlm_inputs.items()}
                output = vlm_model.module.generate(**vlm_inputs, max_new_tokens=100)
                output = processor.decode(output[0], skip_special_tokens=True)
                out = util.extract_answer_candidate(output)
                score = util.get_score(out)
                answer_score_dic[ans] += score

                if opt.debug:
                    print(
                        "  ########################### Scoring Detail ################################################\n")
                    print(f"premise: {premise}\n")
                    print(f"score: {score}\n")

                if score > opt.threshold:
                    new_premise[ans].append(premise)

        helpful_answer = max(answer_score_dic, key=answer_score_dic.get)

        ########################### INFERENCE ################################################
        if opt.inference_model == 'vlm':

            c3 = prompt.VLM_PROMPT_FORMAT_FINAL_ANSWER.format(question=question[0], premise=str(new_premise))

            vlm_inputs = processor(c3, images[0], return_tensors="pt")
            vlm_inputs = {k: v.to(local_rank) for k, v in vlm_inputs.items()}

            output = vlm_model.module.generate(**vlm_inputs, max_new_tokens=100)
            output = processor.decode(output[0], skip_special_tokens=True)
            answer = util.extract_answer_candidate(output)

        elif opt.inference_model == 'llm':

            vlm_inputs = processor("[INST] <image>\nWhat is shown in this image? [/INST]", images[0],
                                   return_tensors="pt")
            vlm_inputs = {k: v.to(local_rank) for k, v in vlm_inputs.items()}

            output = vlm_model.module.generate(**vlm_inputs, max_new_tokens=100)
            output = processor.decode(output[0], skip_special_tokens=True)
            caption = util.extract_answer_candidate(output)

            c3 = prompt.LLM_PROMPT_FORMAT_FINAL_ANSWER.format(question=question[0], caption=caption,
                                                              premise=str(new_premise))

            input_tokens = llm_tokenizer(c3, return_tensors="pt")["input_ids"].to("cuda")

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
            answer = generated_text[len(c3):]

        if opt.debug:
            print("  ########################### Scoring ################################################\n")
            print(f"PROMPT: {c3}\n")
            print(f"new_premise: {new_premise}\n")
            print(f"helpful_answer: {helpful_answer}\n")
            print(f"GT: {target[0]}\n")
            print(f"answer: {answer}\n")

        if target[0] != None:
            if opt.data == 'gqa' or opt.data == 'tdiuc':
                if answer.lower() == target[0]:
                    score = 1
                else:
                    score = 0
            else:
                score = util.ScoreEvaluation(answer, target[0], len(train_examples), idx,
                                             logger)  # answer,answer_list,dataset_length,index

            exactmatch.append(score)

        if opt.debug:
            print("  ########################### ACC ################################################\n")
            print(f"acc: {100 * np.mean(exactmatch)}\n")
            debug_num += 1
            if debug_num == 5:
                break

        idx += 1
        if target[0] != None:
            if idx % 10 == 0:
                logger.warning(
                    f'Process rank: {local_rank}, total: {idx} test examples | accuracy = {100 * np.mean(exactmatch):.2f}%')

        if opt.data == 'aokvqa':
            all_predict_data.append({
                "imagepath": image_path,
                "question": question,
                "premise_for_answer": premise_for_answer,
                "object_candidate": object_candidate,
                "new_premise": new_premise,
                "gt": target,
                "final_answer": answer,
                "difficult_direct_answer": difficult_direct_answer,
                "answer_score_dic":answer_score_dic
            })

            predict_data.append({
                question_id[0]:
                    {
                        'multiple_choice': None,
                        'direct_answer': answer
                    }
            })

        elif opt.data == 'okvqa':
            all_predict_data.append({
                "imagepath": image_path,
                "question": question,
                "pre_answer": answer_candidate,
                "premise": p1,
                "cs_premise": p3,
                "gt": target,
                "final_answer": answer,
            })

        elif opt.data == 'gqa':
            all_predict_data.append({
                "imagepath": image_path,
                "question": question,
                "gt": target,
                "answer_candidate": answer_candidate,
                "object_candidate": object_candidate,
                "premise": premise_for_answer,
                "new_premise": new_premise,
                "helpful_answer": helpful_answer,
                "score_for_GT": score,
                "type": type,
                "group": group,
                "final_answer": answer
            })

        elif opt.data == 'tdiuc':
            all_predict_data.append({
                "imagepath": image_path,
                "question": question,
                "pre_answer": answer_candidate,
                "premise": p1,
                "cs_premise": p3,
                "gt": target,
                "final_answer": answer,
                "question_type": question_type
            })

        torch.distributed.barrier()

    all_predict_data_path = os.path.join(opt.logger_path, opt.name, f'aokvqa_predict.json')
    temp_save_path = os.path.join(opt.logger_path, opt.name, f'temp_results_rank_{local_rank}.json')

    predict_data_path = os.path.join(opt.logger_path, opt.name, f'predictions_val.json')
    pre_temp_save_path = os.path.join(opt.logger_path, opt.name, f'predictions_val_rank_{local_rank}.json')

    with open(temp_save_path, 'w', encoding='utf-8') as f:
        json.dump(all_predict_data, f, ensure_ascii=False, indent=4)

    if opt.data == 'aokvqa':
        with open(pre_temp_save_path, 'w', encoding='utf-8') as f:
            json.dump(predict_data, f, ensure_ascii=False, indent=4)

    torch.distributed.barrier()

    # if local_rank == 0:
    #     # Merge results from all ranks


#     final_results = []
#     final_results2 = []
#
#     for rank in range(torch.distributed.get_world_size()):
#         temp_file = os.path.join(opt.logger_path,opt.name, f'temp_results_rank_{rank}.json')
#         with open(temp_file, 'r', encoding='utf-8') as f:
#             final_results.extend(json.load(f))
#         os.remove(temp_file)
#
#     with open(all_predict_data_path, 'w', encoding='utf-8') as f:
#         json.dump(final_results, f, ensure_ascii=False, indent=4)
#         f.write(',\n')
#
#     for rank in range(torch.distributed.get_world_size()):
#         temp_file = os.path.join(opt.logger_path, opt.name, f'predictions_val_rank_{rank}.json')
#         with open(temp_file, 'r', encoding='utf-8') as f:
#             final_results2.extend(json.load(f))
#         os.remove(temp_file)
#
#     if opt.data == 'aokvqa':
#         with open(predict_data_path, 'w', encoding='utf-8') as f:
#             json.dump(final_results2, f, ensure_ascii=False, indent=4)
#             f.write(',\n')


if __name__ == "__main__":
    main()