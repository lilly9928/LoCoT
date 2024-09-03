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
from torch.nn.parallel import DistributedDataParallel as DDP

import os
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,LlavaNextProcessor, LlavaNextForConditionalGeneration,Blip2Processor,Blip2ForConditionalGeneration
from pathlib import Path
import prompt

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

        start_index = int(opt.start_index)
        from_stopped = list(range(start_index, len(dataset) - 1))
        subset = Subset(dataset, from_stopped)


    # 데이터로더 초기화
    if opt.data == 'aokvqa':
        dataloader = DataLoader(dataset, batch_size=opt.per_gpu_batch_size, sampler=DistributedSampler(dataset), shuffle=False,
                            collate_fn=data.aokvqa_collate_fn)
    elif opt.data=='okvqa':
        dataloader = DataLoader(dataset, batch_size=opt.per_gpu_batch_size, sampler=DistributedSampler(dataset),
                                shuffle=False,
                                collate_fn=data.trainer_v3collate_fn)
    elif opt.data == 'gqa':
        dataloader = DataLoader(subset, batch_size=opt.per_gpu_batch_size, sampler=DistributedSampler(subset),
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

        ##c1 : lvlm 모델 정답 추론
        c1_p = prompt.VLM_PROMPT_FORMAT_ANSWER_onlyVLM.format(question=question[0])
        vlm_inputs = processor(c1_p, images[0], return_tensors="pt")
        vlm_inputs = {k: v.to(local_rank) for k, v in vlm_inputs.items()}

        output = vlm_model.module.generate(**vlm_inputs, max_new_tokens=100)
        c1 = processor.decode(output[0], skip_special_tokens=True)
        c1 = util.extract_answer_candidate(c1)

        ##p1 : 이미지 속 관련 물체 premise 생성
        p1_cp = prompt.LLAVA_PROMPT_FORMAT_OBJECT.format(objects=image_object, question=question[0])
        vlm_inputs = processor(p1_cp, images[0], return_tensors="pt")
        vlm_inputs = {k: v.to(local_rank) for k, v in vlm_inputs.items()}

        out = vlm_model.module.generate(**vlm_inputs, max_new_tokens=100)
        p1_c = processor.decode(out[0], skip_special_tokens=True)
        p1_c = util.extract_answer_candidate(p1_c)


        p1_p = prompt.LLAVA_PROMPT_FORMAT_OBJECT_PREMISE.format(object=p1_c)
        if p1_c in image_object:
            idx_pre_object = image_object.index(p1_c)
            crop_image = util.ImageCrop(images[0], boxes[idx_pre_object])
            vlm_inputs = processor(p1_p,crop_image,return_tensors="pt")
            vlm_inputs = {k: v.to(local_rank) for k, v in vlm_inputs.items()}
        else:
            vlm_inputs = processor(p1_p,images[0],return_tensors="pt")
            vlm_inputs = {k: v.to(local_rank) for k, v in vlm_inputs.items()}

        out = vlm_model.module.generate(**vlm_inputs, max_new_tokens=100)
        p1 = processor.decode(out[0], skip_special_tokens=True)
        p1 = util.extract_answer_candidate(p1)

        #p2 : 글로벌 premise 생성
        # p2_p = prompt.LLM_PROMPT_RULE_GENERATION_OBJECT.format(object=p1_c)
        # input_tokens = llm_tokenizer(p2_p,return_tensors="pt")["input_ids"].to("cuda")
        #
        # with torch.cuda.amp.autocast():
        #     generation_output = llm_model.generate(
        #         input_ids=input_tokens,
        #         max_new_tokens=300,
        #         do_sample=True,
        #         top_k=10,
        #         top_p=0.9,
        #         temperature=0.3,
        #         repetition_penalty=1.15,
        #         num_return_sequences=1,
        #         eos_token_id=llm_tokenizer.eos_token_id,
        #     )
        # p2 = llm_tokenizer.decode(generation_output[0], skip_special_tokens=True)
        # p2 = util.extract_elements(p2[len(p2_p):])



        # p2_p = prompt.VLM_PROMPT_RULE_GENERATION
        #
        # inputs = processor(p2_p, images[0], return_tensors="pt")
        # inputs = {k: v.to(local_rank) for k, v in inputs.items()}
        # output = vlm_model.module.generate(**inputs, max_new_tokens=250)
        # output = processor.decode(output[0], skip_special_tokens=True)
        # p2 = util.extract_answer_candidate(output)

        # p3 : 정답에 대한 premise 생성
        p3_p = prompt.LLM_PROMPT_RULE_GENERATION.format(question=question, answer=c1)

        input_tokens = llm_tokenizer(p3_p,return_tensors="pt")["input_ids"].to("cuda")

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
            p3 = llm_tokenizer.decode(generation_output[0], skip_special_tokens=True)
            p3 = util.extract_elements(p3[len(p3_p):])

        c2 = prompt.VLM_PROMPT_FORMAT_FINAL_ANSWER.format(question=question[0], answer=c1,context=str(p1),object=str(p1_c),
                                                          information=' ', cs=' '.join(p3))

        inputs = processor(c2, images[0], return_tensors="pt")
        inputs = {k: v.to(local_rank) for k, v in inputs.items()}
        output = vlm_model.module.generate(**inputs, max_new_tokens=100)
        output = processor.decode(output[0], skip_special_tokens=True)
        answer = util.extract_answer_candidate(output)

        if opt.data == 'gqa':
            if answer.lower() == target[0]:
                score = 1
            else:
                score = 0
        else:
            score = util.ScoreEvaluation(answer, target[0], len(train_examples), idx,
                                         logger)  # answer,answer_list,dataset_length,index

        exactmatch.append(score)

        idx += 1
        if idx % 10 == 0:
            logger.warning(
                f'Process rank: {local_rank}, total: {idx} test examples | accuracy = {100 * np.mean(exactmatch):.2f}%')

        if opt.data == 'aokvqa':
            all_predict_data.append({
                "imagepath": image_path,
                "question": question,
                "pre_answer": c1,
                "premise": p1,
                "cs_premise": p3,
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


    all_predict_data_path = os.path.join(opt.logger_path,opt.name, f'gqa_predict.json')
    temp_save_path = os.path.join(opt.logger_path,opt.name, f'temp_results_rank_{local_rank}.json')

    # predict_data_path = os.path.join(opt.logger_path, opt.name, f'predictions_val.json')
    # pre_temp_save_path = os.path.join(opt.logger_path, opt.name, f'predictions_val_rank_{local_rank}.json')

    with open(temp_save_path, 'w', encoding='utf-8') as f:
        json.dump(all_predict_data, f, ensure_ascii=False, indent=4)

    # with open(pre_temp_save_path, 'w', encoding='utf-8') as f:
    #     json.dump(predict_data, f, ensure_ascii=False, indent=4)

    torch.distributed.barrier()

    if local_rank == 0:
        # Merge results from all ranks
        final_results = []
        final_results2 = []

        for rank in range(torch.distributed.get_world_size()):
            temp_file = os.path.join(opt.logger_path,opt.name, f'temp_results_rank_{rank}.json')
            with open(temp_file, 'r', encoding='utf-8') as f:
                final_results.extend(json.load(f))
            os.remove(temp_file)

        with open(all_predict_data_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)
            f.write(',\n')

        for rank in range(torch.distributed.get_world_size()):
            temp_file = os.path.join(opt.logger_path, opt.name, f'predictions_val_rank_{rank}.json')
            with open(temp_file, 'r', encoding='utf-8') as f:
                final_results2.extend(json.load(f))
            os.remove(temp_file)

        # with open(predict_data_path, 'w', encoding='utf-8') as f:
        #     json.dump(final_results2, f, ensure_ascii=False, indent=4)
        #     f.write(',\n')


if __name__ == "__main__":
    main()