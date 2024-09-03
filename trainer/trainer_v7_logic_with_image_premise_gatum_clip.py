## gatum부분 추가

import sys
sys.path.append('/data3/KJE/user2/code/WIL_DeepLearningProject_2/NS2')

import torch
from torch.utils.data import DataLoader,DistributedSampler,Subset
import json
from torchvision import transforms
import matplotlib.pyplot as plt
import utils.okvqa_data as data
import utils.gqa_data as GQAdata
import utils.evaluation
from options import Options
from torch.nn.parallel import DistributedDataParallel as DDP
import transformers

import os
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,LlavaNextProcessor, LlavaNextForConditionalGeneration,Blip2Processor,Blip2ForConditionalGeneration
# add CLIP
from transformers import CLIPProcessor, CLIPModel

from pathlib import Path
import prompt2_gatum as prompt
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

    torch.distributed.init_process_group(backend='nccl',timeout=datetime.timedelta(seconds=7200))


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
    torch.manual_seed(opt.seed)
    # 데이터셋 불러오기

    train_examples = data.load_data(
        opt.data_path,
    )

    if opt.data =='aokvqa':
        dataset = data.AOKVQA_Dataset(
            data=train_examples,
            coco_path=opt.image_path,
        )

    elif opt.data=='okvqa':
        dataset = data.trainerV3_Dataset(
            data=train_examples,
            coco_path=opt.image_path,
            object_path=opt.object_path,
        )

    elif opt.data=='gqa':
        train_examples = GQAdata.loadFile(
            '/data2/NS/GQA/preprocess/all_testdev_data.json'
        )
        dataset = GQAdata.trainerV3_Dataset(
            data=train_examples['questions'],
            image_dir='/data2/NS/GQA/images/images',
            object_path=opt.object_path,
        )

    # 데이터로더 초기화
    if opt.data == 'aokvqa':
        dataloader = DataLoader(dataset, batch_size=opt.per_gpu_batch_size, sampler=DistributedSampler(dataset), shuffle=False,
                            collate_fn=data.aokvqa_collate_fn)
    elif opt.data=='okvqa':
        dataloader = DataLoader(dataset, batch_size=opt.per_gpu_batch_size, sampler=DistributedSampler(dataset),
                                shuffle=False,
                                collate_fn=data.trainer_v3collate_fn)
    elif opt.data == 'gqa':
        dataloader = DataLoader(dataset, batch_size=opt.per_gpu_batch_size, sampler=DistributedSampler(dataset),
                                shuffle=False,
                                collate_fn=GQAdata.trainer_v3collate_fn)

    # Model and processor initialization
    if 'llava-hf' in opt.lvlm_name:
        processor = LlavaNextProcessor.from_pretrained(opt.lvlm_name, cache_dir=opt.cache_dir)
        vlm_model = LlavaNextForConditionalGeneration.from_pretrained(opt.lvlm_name,
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

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(opt.detection_config))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = opt.detection_weight
    predictor = DefaultPredictor(cfg)

    idx = 0

    all_predict_data=[]
    predict_data=[]
    
    # CLIP instantiation
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14",
        device_map="auto",
        torch_dtype=torch.float16,
        cache_dir=opt.cache_dir,
    )
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    clip_model = clip_model.to(local_rank)
    clip_model = DDP(clip_model, device_ids=[local_rank])

    ########################################################################

    # Process each batch
    for batch in tqdm(dataloader):
        if opt.data == 'aokvqa':

            images, image_path, question, choices, difficult_direct_answer, target,question_id = batch
        elif opt.data == 'okvqa':
            images, image_path, question, _, _, _, target = batch
        elif opt.data == 'gqa':
            images, image_path, question, target, type, group, fullanswer = batch

        ##FastRCNN -> 이미지 속 객체 인식, 박스
        image_object, boxes = util.detect_image_ver1(image_path[0],predictor)
        boxes=boxes.tensor.cpu().tolist()

        ############################ conclsion 1 (Answer from LVLM) ##############################
        answer_candidate = []

        c1_p = prompt.VLM_PROMPT_FORMAT_ANSWER_onlyVLM.format(question=question[0])
        vlm_inputs = processor(c1_p, images[0], return_tensors="pt")
        vlm_inputs = {k: v.to(local_rank) for k, v in vlm_inputs.items()}
        outputs = vlm_model.module.generate(**vlm_inputs, max_new_tokens=300,
                                     num_beams=opt.beam_num_vlm,
                                     num_return_sequences=opt.beam_num_vlm,
                                     early_stopping=True,
                                     no_repeat_ngram_size=opt.beam_num_vlm,
                                     top_p=0.98,
                                     temperature=0.5, )

        for output in outputs:
            c1 = processor.decode(output, skip_special_tokens=True)
            c1 = util.extract_answer_candidate(c1)
            answer_candidate.append(c1)

         ############################ conclsion1 (Answer from LVLM) ##############################

        premise_for_answer = {}
        for answer in answer_candidate:
            premise_for_answer[answer] = []
            
        # print("=+=+=+=+=+=+= possible answers\n", answer_candidate)

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

        ########################### Object conclsion2 ################################################

        ########################### Object Premise1 ################################################
        LLAVA_PROMPT_FORMAT_ANSWER_PREMISE = prompt.prepare_object_premises_prompt(num_premises=opt.num_object_premises)
        for c2 in object_candidate:
            if c2.lower() in image_object:
                idx_pre_object = image_object.index(c2.lower())
                crop_image = util.ImageCrop(images[0], boxes[idx_pre_object])

            else:
                crop_image = images[0]

            for answer in answer_candidate:
                p1_p = LLAVA_PROMPT_FORMAT_ANSWER_PREMISE.format(question=question[0], answer=answer, object=c2.lower())
                # print('prompt object premise')
                # print(p1_p)
                vlm_inputs = processor(p1_p, crop_image, return_tensors="pt")
                vlm_inputs = {k: v.to(local_rank) for k, v in vlm_inputs.items()}

                out = vlm_model.module.generate(**vlm_inputs, max_new_tokens=100)
                p1 = processor.decode(out[0], skip_special_tokens=True)
                # print('================ result from vlm', p1)                    
                p1 = util.extract_answer_candidate(p1)
                if opt.num_object_premises > 1:
                    p1 = util.extract_elements(p1)
                else:
                    p1 = [p1]
                premise_for_answer[answer] = p1
        ########################### Object Premise1 ################################################

        ########################### LLM Premise2 ################################################
        answer_score_dic = {}
        LLM_PROMPT_RULE_GENERATION = prompt.prepare_llm_premises_prompt(opt.num_llm_premises)
        for ans in answer_candidate:
            answer_score_dic[ans] = 0
            answer = ans
            p2 = LLM_PROMPT_RULE_GENERATION.format(question=question[0], answer=answer)
            # print('prompt llm premise')
            # print(p2)

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
            # print('================ result from llm', generated_text)
            if (opt.num_llm_premises > 1):
                generated_text = util.extract_elements(generated_text[len(p2):])
            else:
                generated_text = [generated_text[len(p2):]]
            premise_for_answer[ans] += generated_text
        ########################### Scoring ################################################
        # print('========================== answer-premise')
        # print(premise_for_answer)
        # raise('dont worry :) this is not an error')

        new_premise = {}

        for ans in answer_candidate:
            new_premise[ans] = []
            for premise in premise_for_answer[ans]:
                # p3 = prompt.VLM_PROMPT_FORMAT_SCORE.format(information=premise)
                # vlm_inputs = processor(p3, images[0], return_tensors="pt")
                # vlm_inputs = {k: v.to(local_rank) for k, v in vlm_inputs.items()}
                # output = vlm_model.module.generate(**vlm_inputs, max_new_tokens=100)
                # output = processor.decode(output[0], skip_special_tokens=True)
                # out = util.extract_answer_candidate(output)
                # score = util.get_score(out)
                
                clip_inputs = clip_processor(text=[
                    f"The picture contains this information: {premise}", 
                    f"The picture doesn't contain this information: {premise}"], images=images[0], return_tensors="pt", padding=True)

                clip_outputs = clip_model(**clip_inputs)
                logits_per_image = clip_outputs.logits_per_image # this is the image-text similarity score
                probs = logits_per_image.softmax(dim=1)
                print("============ CLIP INPUTS", clip_inputs)
                print("============ CLIP PROBS", probs)
                raise('dont worry :) this is not an error')
                
                answer_score_dic[ans] += score
                if score > 0.5:
                    # This code pottentially returns empty answer because of the threshold.
                    new_premise[ans].append(premise)
                helpful_answer = max(answer_score_dic, key=answer_score_dic.get)

        if len(new_premise.values())==0:
            new_premise = ''
        else:
            new_premise= list(new_premise.values())
        
        try:
            c3 = prompt.VLM_PROMPT_FORMAT_FINAL_ANSWER.format(question=question[0], premise=str(new_premise))
        except:
            c3 = prompt.VLM_PROMPT_FORMAT_FINAL_ANSWER.format(question=question[0], premise="Clues are not provided for this question")

        vlm_inputs = processor(c3, images[0], return_tensors="pt")
        vlm_inputs = {k: v.to(local_rank) for k, v in vlm_inputs.items()}

        output = vlm_model.module.generate(**vlm_inputs, max_new_tokens=100)
        output = processor.decode(output[0], skip_special_tokens=True)
        answer = util.extract_answer_candidate(output)

        if target[0]!= None:
            if opt.data == 'gqa':
                if answer.lower() == target[0]:
                    score = 1
                else:
                    score = 0
            else:
                score = util.ScoreEvaluation(answer, target[0], len(train_examples), idx,
                                            logger)  # answer,answer_list,dataset_length,index

            exactmatch.append(score)

        if target[0]!= None:
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
            })

            predict_data.append({
                question_id[0]:
                    {
                        'multiple_choice': answer,
                        'direct_answer': answer
                    }
            })

        elif opt.data == 'okvqa':
            all_predict_data.append({
                "imagepath": image_path,
                "question": question,
                "pre_answer": c1,
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
                "final_answer": answer,
                "pre_answer": c1,
                "premise": p1,
                "cs_premise": p3,
                "group": group,
                "type": type
            })

        torch.distributed.barrier()
        # break

    all_predict_data_path = os.path.join(opt.logger_path,opt.name, f'aokvqa_predict.json')
    temp_save_path = os.path.join(opt.logger_path,opt.name, f'temp_results_rank_{local_rank}.json')

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