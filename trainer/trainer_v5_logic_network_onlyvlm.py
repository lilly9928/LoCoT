## gatum부분 추가

import sys
sys.path.append('/data3/KJE/user2/code/WIL_DeepLearningProject_2/NS2')

import torch
from torch.utils.data import DataLoader
import json
from torchvision import transforms
import matplotlib.pyplot as plt
import utils.okvqa_data as data
import utils.evaluation
from options import Options
from torch.nn import DataParallel
import transformers
import re
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,LlavaNextProcessor, LlavaNextForConditionalGeneration
from pathlib import Path
import prompt
import random
import numpy as np
import logging
import utils.util as util
from PIL import Image

import warnings

# Suppress UserWarning
warnings.filterwarnings("ignore", category=UserWarning)



if __name__ == "__main__":

    transformers.logging.set_verbosity_error()
    options = Options()
    options.add_reader_options()
    options.add_model_options()
    opt = options.parse()
    opt.is_main = True
    opt.is_distributed = False
    logger = util.init_logger(opt.is_main,opt.is_distributed, Path(opt.logger_path) / opt.name / 'run.log')

    options.print_options(opt)

    exactmatch = []

    debug=False

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
    dataloader = DataLoader(dataset, batch_size=opt.per_gpu_batch_size, shuffle=True,collate_fn=data.trainer_v3collate_fn)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    # Model and processor initialization
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf" ,cache_dir=opt.cache_dir)
    vlm_model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",cache_dir=opt.cache_dir)
    vlm_model.to('cuda:0')


    all_predict_data = {}
    answer_rationle_data= {}
    correct_count = 0
    total_count = 0
    max_i = 10
    last_answer = None
    answer = ' '
    idx = 0

    ########################################################################

    # Process each batch
    for batch in tqdm(dataloader):
        ans_premises=[]
        images,image_path, question,caption, pred_classes, crop_images, target = batch
        check_image=Image.open(image_path[0])
        p2 = prompt.VLM_PROMPT_FORMAT_ANSWER_onlyVLM.format(question=question[0])

        inputs = processor(p2, check_image, return_tensors="pt").to('cuda:0')
        output = vlm_model.generate(**inputs, max_new_tokens=100)
        output = processor.decode(output[0], skip_special_tokens=True)
        final_answer = util.extract_answer_candidate(output)


        score = util.ScoreEvaluation(final_answer, target[0], len(train_examples), idx,logger)  # answer,answer_list,dataset_length,index
        exactmatch.append(score)

        idx += 1
        if idx % 10 == 0:
            logger.warning(
                f'Process rank: {0}, total: {idx} test examples | accuracy = {100 * np.mean(exactmatch):.2f}%')


        all_predict_data={
                "imagepath": image_path,
                "question": question,
                "gt":target,
                "final_answer":final_answer
            }

        all_predict_data_path = os.path.join(opt.logger_path, f'okvqa_predict_llava-v1.6-mistral-7b-hf-v2_no_random.json')

        with open(all_predict_data_path, 'a', encoding='utf-8') as f:
            json.dump(all_predict_data, f, ensure_ascii=False, indent=4)
            f.write(',\n')