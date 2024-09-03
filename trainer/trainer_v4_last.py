## a==ai-1 부분 수정 , answer prompt 수정

import sys
sys.path.append('/data3/KJE/user2/code/WIL_DeepLearningProject_2/NS2')

import torch
from torch.utils.data import DataLoader
import json
from torchvision import transforms
import matplotlib.pyplot as plt
import utils.okvqa_data as data
import utils.evaluation as evaluation
from options import Options
from torch.nn import DataParallel
import transformers
import re
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,Blip2Processor,Blip2ForConditionalGeneration,CLIPProcessor, CLIPModel,LlavaNextProcessor, LlavaNextForConditionalGeneration

import prompt
import random
import numpy as np
import logging
import cv2

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

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_random_seed(seed=0)


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
    result = text.split("====\n")[1].split("Answer:")[1].strip()
    result = re.sub(r'( +)', ' ', re.sub(r'\n', ' ', result))
    if "Context" in result:
        return result.split('Context')[0]
    else:
        return result


def extract_ans_rat(text):
    keyword = "answer is"

    text = text.lower()
    res = text.split('.')

    answer = res[0]
    keyword_position = answer.find(keyword)
    if keyword_position != -1:
        after_keyword = answer[keyword_position + len(keyword):].strip()
        try:
            answer = after_keyword.split()[0]

        except:
            answer = res[0]

        try:
            rat = res[1]
        except:
            rat= text

    return answer, rat


def check_pronouns(text):
    pronouns = [
        "I", "you", "he", "she", "it", "we", "they",
        "me", "you", "him", "her", "it", "us", "them",
        "my", "your", "his", "her", "its", "our", "their",
        "mine", "yours", "his", "hers", "ours", "theirs",
        "myself", "yourself", "himself", "herself", "itself",
        "ourselves", "yourselves", "themselves",
        "this", "that", "these", "those"
    ]
    if debug:
        print(text)
        print(any(pronoun in text for pronoun in pronouns))
    if any(pronoun in text for pronoun in pronouns):
        return True

    else:
        return False


def detect_image(image):
    image_object = []
    img=cv2.imread(image)
    outputs = predictor(img)
    classids = outputs["instances"].pred_classes
    for id in classids:
        image_object.append(coco_categories[id])

    pred_boxes = outputs["instances"].pred_boxes

    return image_object,pred_boxes


def ImageCrop(image, box_coordinates):
    try:
        # box_coordinates=box_coordinates.cpu().numpy().astype(int).tolist()[0]
        cropped_image = image.crop(box_coordinates)

        return cropped_image
    except Exception as e:
        print(f"Error: {e}")
        return None

def final_prompt(caption, pred_pronouns_caption, pred_object_caption, logic, question):
    sentence = f"""
    Context : {caption}{pred_pronouns_caption}. {pred_object_caption}.{logic}\n
    Question: {question}\n"""
    return sentence

def ScoreEvaluation(answer,answer_list,dataset_length,index):
    score = evaluation.cems( answer, answer_list, index, dataset_length,logging)
    return score



if __name__ == "__main__":

    logging.basicConfig(filename='/data3/KJE/modelLog/Logic_LLama/run_trainer_v4_debug.log', level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

    transformers.logging.set_verbosity_error()
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()

    exactmatch = []

    opt.cache_dir = "/data3/hg_weight/hg_weight"


    # 기본 설정
    image_path = '/data2/KJE/VQA/COCO/images'
    data_path = '/data2/KJE/ModelLogs/revive/processed_data/test.pkl'
    object_path ='/data2/KJE/ModelLogs/Logic_LLama/data/preprocess_okvqa/okvqa_test_object.json'
    cache_dir = "/data3/hg_weight/hg_weight"

    vlm_name = "Salesforce/blip2-flan-t5-xxl"
    # vlm_name = "llava-hf/llava-v1.6-mistral-7b-hf"
    llm4logic_name = "mistralai/Mistral-7B-Instruct-v0.2"
    llm_name = "mistralai/Mistral-7B-Instruct-v0.2"
    peft_model_id = "lilly9928/LogicLLM"
    clip_name = "openai/clip-vit-base-patch32"

    detection_config = '../configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'
    detection_weight='/data2/KJE/weights/model_final_68b088.pkl'

    hg_token = "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"
    batch_size = 1

    debug=False

    coco_categories = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                       'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                       'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                       'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                       'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                       'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                       'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                       'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                       'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                       'teddy bear', 'hair drier', 'toothbrush']

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
    # processor = LlavaNextProcessor.from_pretrained(vlm_name,cache_dir=cache_dir)
    # vlm_model = LlavaNextForConditionalGeneration.from_pretrained(vlm_name,
    #                                                           torch_dtype=torch.float16, low_cpu_mem_usage=True,cache_dir=cache_dir)


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




    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(detection_config))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = detection_weight
    predictor = DefaultPredictor(cfg)


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
        images,image4dectection, question,caption, pred_classes, crop_images, target = batch
        ### 추후 trainable 변경
        image_object, boxes = detect_image(image4dectection[0]) # 이미지 물체 디텍션 -> 모든 물체, 박스 저장

        boxes=boxes.tensor.cpu().tolist()

        proNouns=check_pronouns(question[0]) # pronoun있는지 체크

        if proNouns: # pronoun 있으면 pronoun 으로 가장 관련있는 단어 리턴 (오브젝트 리스트 중에 선택) : llm 예외 존재
            vlm_inputs = processor(images, prompt.pronoun_prompt(question[0],image_object),
                                   return_tensors="pt").to("cuda")
            out = vlm_model.generate(**vlm_inputs)
            pred_pronouns = processor.decode(out[0], skip_special_tokens=True) # pronoun으로 추측되는 단어 반환
            if pred_pronouns in image_object: # 오브젝트 리스트에서 추출했을 경우
                idx_pre_object =image_object.index(pred_pronouns) #이미지 박스에서 캡션 생성
                crop_image = ImageCrop(images[0],boxes[idx_pre_object])
                vlm_inputs = processor(crop_image, prompt.concept_caption_prompt(question,pred_pronouns,idx_pre_object), return_tensors="pt").to("cuda")

            else:
                idx_pre_object = None #전체 이미지에서 캡션 생성
                vlm_inputs = processor(images, prompt.concept_caption_prompt(question,pred_pronouns,idx_pre_object), return_tensors="pt").to("cuda")

            out = vlm_model.generate(**vlm_inputs)
            pred_pronouns_caption = processor.decode(out[0], skip_special_tokens=True)

            if debug:
                print('proNouns: ',proNouns)
                print('Predict ProNouns: ',pred_pronouns)
                print('Predict ProNouns Caption: ',pred_pronouns_caption)


        else:
            pred_pronouns = None

        vlm_inputs = processor(images, prompt.concept_prompt(pred_pronouns, question),
                               return_tensors="pt").to("cuda")
        out = vlm_model.generate(**vlm_inputs)
        pred_object = processor.decode(out[0], skip_special_tokens=True)

        if pred_object in image_object:
            idx_pre_object = image_object.index(pred_object)
            crop_image = ImageCrop(images[0], boxes[idx_pre_object])
            vlm_inputs = processor(crop_image, prompt.concept_caption_prompt(question, pred_object,idx_pre_object),
                                   return_tensors="pt").to("cuda")

        else:
            idx_pre_object = None  # 전체 이미지에서 캡션 생성
            vlm_inputs = processor(images, prompt.concept_caption_prompt(question,pred_object, idx_pre_object),
                                   return_tensors="pt").to("cuda")

        out = vlm_model.generate(**vlm_inputs)
        pred_object_caption = processor.decode(out[0], skip_special_tokens=True)

        if debug:
            print('pred_object in image_object: ', pred_object in image_object)
            print('pred_object: ', pred_object)
            print('Predict object Caption: ', pred_object_caption)


        ##logic
        input_tokens = logic_tokenizer(prompt.logic_prompt(pred_object), return_tensors="pt")["input_ids"].to("cuda")
        # input_tokens = {k: v.to(local_rank) for k, v in input_tokens.items()}
        with torch.cuda.amp.autocast():
            generation_output = logic_model.generate(
                input_ids=input_tokens,
                max_new_tokens=30,
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

        if debug:
            print('logic_result ', logic_result)
            print('logic_result_clean ', logic_result_clean)


        input_tokens = llm_tokenizer(
            prompt.final_input_prompt(caption[0], pred_object_caption,pred_pronouns_caption, logic_result_clean, question[0]),
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
        final = llm_tokenizer.decode(generation_output[0], skip_special_tokens=True)

        answer = extract_answer(final)

        ans,rat=extract_ans_rat(answer)

        fprompt=final_prompt(caption[0], pred_object_caption, pred_pronouns_caption, logic_result_clean, question[0])

        score = ScoreEvaluation(ans,target[0],len(train_examples),idx)#answer,answer_list,dataset_length,index
        exactmatch.append(score)
        idx+=1
        if idx % 10 == 0:
            logging.warning(
                f'Process rank: {0}, total: {idx} test examples | accuracy = {100 * np.mean(exactmatch):.2f}%')
        if debug:
            print('final prompt ', fprompt)
            print('answer ', answer)


        all_predict_data={
            "proNouns_TF": proNouns,
            "pred_ProNouns": pred_pronouns,
            "pred_ProNouns_caption": pred_pronouns_caption,
            "pred_object":pred_object,
            "pred_object_caption":pred_object_caption,
            "logic_result":logic_result,
            "logic_result_clean":logic_result_clean,
            "final_prompt":fprompt,
            "answer":answer
        }



        answer_rationle_data={
            "full_answer":answer,
            "answer":ans,
            "rationle":rat,
            "answer_list":target
        }

        if debug:
            print(all_predict_data)
            print(answer_rationle_data)

        save_path = '/data3/KJE/modelLog/Logic_LLama/'
        answer_rationle_data_path = os.path.join(save_path,f'without_iter_results_okvqa_test_0827.json')
        all_predict_data_path = os.path.join(save_path,f'without_iter_all_predict_data_okvqa_test_0827.json')

        with open(answer_rationle_data_path, 'a', encoding='utf-8') as f:
            json.dump(answer_rationle_data, f, ensure_ascii=False, indent=4)
            f.write(',\n')

        with open(all_predict_data_path, 'a', encoding='utf-8') as f:
            json.dump(all_predict_data, f, ensure_ascii=False, indent=4)
            f.write(',\n')
