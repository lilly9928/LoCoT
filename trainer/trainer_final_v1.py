## a==ai-1 부분 수정 , answer prompt 수정

import sys
sys.path.append('/data3/KJE/user2/code/WIL_DeepLearningProject_2/NS2')

import torch
from torch.utils.data import DataLoader
import json
import utils.okvqa_data as data
import utils.util as util

from options import Options
from torch.nn import DataParallel
import transformers

import os
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,Blip2Model,AutoProcessor,Blip2Processor,Blip2ForConditionalGeneration,CLIPProcessor, CLIPModel,LlavaNextProcessor, LlavaNextForConditionalGeneration
from pathlib import Path
import prompt
import numpy as np
import logging

import warnings

# Suppress UserWarning
warnings.filterwarnings("ignore", category=UserWarning)


import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()


from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog






if __name__ == "__main__":

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
    dataloader = DataLoader(dataset, batch_size=opt.per_gpu_batch_size, shuffle=False,collate_fn=data.trainer_v3collate_fn)

    # Model and processor initialization
    processor = Blip2Processor.from_pretrained(opt.vlm_name, cache_dir=opt.cache_dir)
    vlm_model = Blip2ForConditionalGeneration.from_pretrained(opt.vlm_name, device_map="cuda", cache_dir=opt.cache_dir)


    llava_processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf" ,cache_dir=opt.cache_dir)
    llava = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",
                                                                  torch_dtype=torch.float16,
                                                                  use_flash_attention_2=True,cache_dir=opt.cache_dir)
    llava.to('cuda:0')

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16)



    llm_model = AutoModelForCausalLM.from_pretrained(
        opt.llm_name,
        quantization_config=bnb_config,
        device_map='cuda',
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        cache_dir=opt.cache_dir
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

    all_predict_data = {}
    answer_rationle_data= {}
    correct_count = 0
    total_count = 0
    max_i = 10
    last_answer = None
    answer = ' '
    idx = 0

    answer_model='llava'

    ########################################################################

    # Process each batch
    for batch in tqdm(dataloader):
        images,image4dectection, question,caption, pred_classes, crop_images, target = batch
        pred_pronouns = None

        if answer_model == 'blip':
            answer_prompt = prompt.BLIP_PROMPT_FORMAT_ANSWER.format(question=question[0])
            vlm_inputs = processor(images, question[0], return_tensors="pt").to("cuda")
            out = vlm_model.generate(**vlm_inputs,
            max_length=10,
            num_beams=5,
            num_return_sequences=5,
            early_stopping=True,
            no_repeat_ngram_size=2,
            top_p=0.95,
            temperature=0.5,
            do_sample=True)

            answer_list = []
            for output in out:
                answer_list.append(processor.decode(output, skip_special_tokens=True)) #c1

            answer_list = set(answer_list)

        elif answer_model == 'llava':
            answer_prompt = prompt.VLM_PROMPT_FORMAT_ANSWER.format(question=question[0])

            inputs = llava_processor(answer_prompt, images[0], return_tensors="pt").to('cuda')
            output = llava.generate(**inputs, max_new_tokens=100)
            output = llava_processor.decode(output[0], skip_special_tokens=True)
            answer_candidates = util.extract_answer_candidate(output)
            answer_candidates = util.extract_elements(answer_candidates)
            answer_list = list(set(answer_candidates))


        image_object, boxes = util.detect_image_ver1(image4dectection[0],predictor)
        boxes=boxes.tensor.cpu().tolist()

        object_prompt = prompt.BLIP_PROMPT_FORMAT_OBJECT.format(objects=image_object,question=question[0])
        vlm_inputs = processor(images, object_prompt,
                               return_tensors="pt").to("cuda")
        out = vlm_model.generate(**vlm_inputs)
        pred_object = processor.decode(out[0], skip_special_tokens=True) #Predict related object(before p1)


        Image_premise_prompt = prompt.BLIP_PROMPT_FORMAT_PREMISE.format(question=question[0])
        vlm_inputs = processor(images, Image_premise_prompt,
                                   return_tensors="pt").to("cuda")
        out = vlm_model.generate(**vlm_inputs)
        global_premise = processor.decode(out[0], skip_special_tokens=True) #global_premise(p1)

        object_premise_prompt = prompt.BLIP_PROMPT_FORMAT_OBJECT_PREMISE.format(object=pred_object)
        if pred_object in image_object:
            idx_pre_object = image_object.index(pred_object)
            crop_image = util.ImageCrop(images[0], boxes[idx_pre_object])
            vlm_inputs = processor(crop_image, object_premise_prompt,
                                   return_tensors="pt").to("cuda")
        else:
            vlm_inputs = processor(images, object_premise_prompt,
                                   return_tensors="pt").to("cuda")
        out = vlm_model.generate(**vlm_inputs)
        object_premise = processor.decode(out[0], skip_special_tokens=True)  # object_premise(p1)


        if debug:
            print('question ', question)
            print('predict answer ', answer_list)
            print('pred_object in image_object: ', pred_object in image_object)
            print('pred_object: ', pred_object)
            print('Predict object global premise: ', global_premise)
            print('Predict object premise: ', object_premise)


        ##logic

        cs_dir = {}
        for pre_answer in answer_list:
            commonsense_premise_prompt = prompt.LLM_PROMPT_FORMAT_COMMONSENSE_PREMISE.format(question=question[0],answer=pre_answer)
            input_tokens = llm_tokenizer(commonsense_premise_prompt, return_tensors="pt")["input_ids"].to("cuda")
            # input_tokens = {k: v.to(local_rank) for k, v in input_tokens.items()}
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
            cs_premise = llm_tokenizer.decode(generation_output[0], skip_special_tokens=True) #p2
            cs_premise = util.extract_elements(cs_premise[len(commonsense_premise_prompt):])

            cs_dir[pre_answer]=cs_premise

        commonsense_premise_prompt = prompt.LLM_PROMPT_FORMAT_COMMONSENSE_PREMISE.format(question=question[0],answer=pred_object)
        input_tokens = llm_tokenizer(commonsense_premise_prompt, return_tensors="pt")["input_ids"].to("cuda")
        # input_tokens = {k: v.to(local_rank) for k, v in input_tokens.items()}
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
        cs_premise = llm_tokenizer.decode(generation_output[0], skip_special_tokens=True)  # p2
        cs_premise = util.extract_elements(cs_premise[len(commonsense_premise_prompt):])

        cs_dir[pred_object] = cs_premise


        if debug:
            print('logic_result ', cs_dir)

        c2 = prompt.VLM_PROMPT_FORMAT_FINAL_ANSWER.format(question=question[0],answer=pre_answer,information=str(global_premise)+' '+str(object_premise), object_name=pre_answer, cs = str(cs_dir))


        inputs = llava_processor(c2, images[0], return_tensors="pt").to('cuda')
        output = llava.generate(**inputs, max_new_tokens=100)
        output = llava_processor.decode(output[0], skip_special_tokens=True)
        answer = util.extract_answer_candidate(output)

        if debug:
            print('output: ', answer)
            print('gt:',target)

        score = util.ScoreEvaluation(answer,target[0],len(train_examples),idx,logger)#answer,answer_list,dataset_length,index
        exactmatch.append(score)
        idx+=1
        if idx % 10 == 0:
            logging.warning(
                f'Process rank: {0}, total: {idx} test examples | accuracy = {100 * np.mean(exactmatch):.2f}%')
        if debug:
            logging.debug('final prompt ', output)
            logging.debug('answer ', answer)


        all_predict_data={
            "Predict answer with blip2": str(answer_list),
            "global premise": global_premise,
            "object premise": object_premise,
            "commonsense":str(cs_dir),
            "final prompt":output,
            "final answer":answer,
            "target":target,
        }



        answer_rationle_data={
            "answer":answer,
            "answer_list":target
        }

        if debug:
            logging.debug(all_predict_data)
            logging.debug(answer_rationle_data)

        answer_rationle_data_path = os.path.join(opt.logger_path,opt.name,f'new_version_blip2_Mistral_Llava_results_okvqa_test_0703.json')
        all_predict_data_path = os.path.join(opt.logger_path,opt.name,f'new_version_blip2_Mistral_Llava_all_predict_data_okvqa_test_0703.json')

        with open(answer_rationle_data_path, 'a', encoding='utf-8') as f:
            json.dump(answer_rationle_data, f, ensure_ascii=False, indent=4)
            f.write(',\n')

        with open(all_predict_data_path, 'a', encoding='utf-8') as f:
            json.dump(all_predict_data, f, ensure_ascii=False, indent=4)
            f.write(',\n')
