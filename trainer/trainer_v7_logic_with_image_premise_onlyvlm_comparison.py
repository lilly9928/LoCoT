## gatum부분 추가

import sys
sys.path.append('/data3/KJE/user2/code/WIL_DeepLearningProject_2/NS2')
from openai import OpenAI
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
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,LlavaNextProcessor, LlavaNextForConditionalGeneration,Blip2Processor,Blip2ForConditionalGeneration,AutoProcessor, Kosmos2ForConditionalGeneration,LlamaTokenizer,PaliGemmaForConditionalGeneration,AutoModel
from pathlib import Path
import prompt2 as prompt
import datetime

import numpy as np
import logging
import utils.util as util

import re
import warnings

# Suppress UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

#from detectron2.utils.logger import setup_logger
#setup_logger()


#from detectron2 import model_zoo
#from detectron2.engine import DefaultPredictor
#from detectron2.config import get_cfg




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

    elif opt.data == 'tdiuc':
        dataset = data.TDIUC_Dataset(
            data=train_examples,
            coco_path=opt.image_path,
        )

    elif opt.data=='gqa':
        train_examples = GQAdata.loadFile(
            '/data2/NS/GQA/preprocess/all_testdev_data.json'
            # '/data2/NS/GQA/500_split_testdev_data.json'
        )
        dataset = GQAdata.trainerV3_Dataset(
            data=train_examples['questions'],
            # data=train_examples,
            image_dir='/data2/NS/GQA/images/images',
            object_path=opt.object_path,
        )

    if opt.debug:
        debug_num = 0


    # 데이터로더 초기화
    if opt.data == 'aokvqa':
        dataloader = DataLoader(dataset, batch_size=opt.per_gpu_batch_size, sampler=DistributedSampler(dataset), shuffle=False,
                            collate_fn=data.aokvqa_collate_fn)
    elif opt.data=='okvqa':
        dataloader = DataLoader(dataset, batch_size=opt.per_gpu_batch_size, sampler=DistributedSampler(dataset),
                                shuffle=False,
                                collate_fn=data.trainer_v3collate_fn)

    elif opt.data=='tdiuc':
        dataloader = DataLoader(dataset, batch_size=opt.per_gpu_batch_size, sampler=DistributedSampler(dataset),
                                shuffle=False,
                                collate_fn=data.tdiuc_collate_fn)

    elif opt.data == 'gqa':
        dataloader = DataLoader(dataset, batch_size=opt.per_gpu_batch_size, sampler=DistributedSampler(dataset),
                                shuffle=False,
                                collate_fn=GQAdata.trainer_v3collate_fn)


    # Model and processor initialization
    # if 'llava' in opt.lvlm_name:
    # processor = LlavaNextProcessor.from_pretrained(opt.lvlm_name, cache_dir=opt.cache_dir)
    # vlm_model = LlavaNextForConditionalGeneration.from_pretrained(opt.lvlm_name,
    #                                                             torch_dtype=torch.float16,
    #                                                             use_flash_attention_2=True, cache_dir=opt.cache_dir)
    
    #processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    #vlm_model = Blip2ForConditionalGeneration.from_pretrained(
    #    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, cache_dir=opt.cache_dir
    #)  # doctest: +IGNORE_RESULT

    #processor = ChameleonProcessor.from_pretrained("facebook/chameleon-7b")
    #vlm_model = ChameleonForConditionalGeneration.from_pretrained("facebook/chameleon-7b", torch_dtype=torch.bfloat16,  cache_dir=opt.cache_dir)
    
    #vlm_model = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224", cache_dir=opt.cache_dir)
    #processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
    
    #processor = AutoProcessor.from_pretrained("google/paligemma-3b-mix-224")
    #vlm_model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-mix-224")

    processor = AutoTokenizer.from_pretrained("THUDM/visualglm-6b", trust_remote_code=True)
    vlm_model = AutoModel.from_pretrained("THUDM/visualglm-6b", trust_remote_code=True)

    vlm_model = vlm_model.to(local_rank)
    vlm_model = DDP(vlm_model, device_ids=[local_rank])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16)


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
            questionId,images, image_path, question, target, type, group, fullanswer = batch
        elif opt.data == 'tdiuc':
            images, image_path, question, target, question_id,question_type = batch



        c3 = prompt.VLM_PROMPT_FORMAT_FINAL_ANSWER_onlyvlm.format(question=question[0])
        print('======== images', images[0])
        print('======== images', images)
        vlm_prompt = f"Question: {question[0]} Answer:"
        tt  = f"Question: {c3} Answer:"
        #vlm_inputs = processor(images=images[0], text=vlm_prompt, return_tensors="pt")
        #vlm_inputs = processor(vlm_prompt, images[0], return_tensors="pt")
        vlm_inputs = processor(images=images[0], text=tt, return_tensors="pt")
        vlm_inputs = {k: v.to(local_rank) for k, v in vlm_inputs.items()}

        print('inputs finis')
        # output = vlm_model.module.generate(**vlm_inputs, max_new_tokens=300,
        #                              num_beams=opt.beam_num_vlm,
        #                              num_return_sequences=opt.beam_num_vlm,
        #                              early_stopping=True,
        #                              no_repeat_ngram_size=opt.beam_num_vlm,
        #                              top_p=0.99,
        #                              temperature=0.5, )
        # print('output finish')
        # output = processor.decode(output[0], skip_special_tokens=True)
        # print('============= output ', output)
        # answer = util.extract_answer_candidate(output)

        #generated_ids = vlm_model.module.generate(**vlm_inputs, max_length=100)
        #generated_ids = vlm_model.module.generate(pixel_values=vlm_inputs["pixel_values"],
        #                        input_ids=vlm_inputs["input_ids"],
        #                        attention_mask=vlm_inputs["attention_mask"],
        #                        image_embeds=None,
        #                        image_embeds_position_mask=vlm_inputs["image_embeds_position_mask"],
        #                        use_cache=True,
        #                        max_new_tokens=256,)
        print("strs")
        generated_ids = vlm_model.module.generate(**vlm_inputs, max_length=300)
        print(1)
        answer_raw = processor.batch_decode(generated_ids, skip_special_tokens=True)
        print(2)
        answer = answer_raw[0].strip()
        #match = re.search(r'Answer:\s*(.*)', answer1)
        #if match:
        #     answer = match.group(1)
        print(answer_raw)
        print(answer)
        



        if target[0]!= None:
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
            debug_num+=1
            print("c3 :"+ c3)
            print("output:"+ output)
            print("answer:"+ answer)
            if debug_num ==5:
                break

        idx += 1
        if target[0]!= None:
            if idx % 10 == 0:
                logger.warning(
                    f'Process rank: {local_rank}, total: {idx} test examples | accuracy = {100 * np.mean(exactmatch):.2f}%')

        if opt.data == 'aokvqa':
            all_predict_data.append({
                'questionid':question_id,
                "imagepath": image_path,
                "question": question,
                "gt": target,
                "final_answer": answer,
                "difficult_direct_answer": difficult_direct_answer,
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
                "gt": target,
                "final_answer": answer,
            })

        elif opt.data == 'gqa':
            all_predict_data.append({
                "questionId":questionId,
                "imagepath": image_path,
                "question": question,
                "gt": target,
                "final_answer": answer,
                "group": group,
                "type": type
            })

        elif opt.data == 'tdiuc':
            all_predict_data.append({
                "imagepath": image_path,
                "question": question,
                "gt": target,
                "final_answer": answer,
                "question_type":question_type
            })

        torch.distributed.barrier()

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