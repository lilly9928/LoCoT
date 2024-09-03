import sys
sys.path.append('/home/user2/code/WIL_DeepLearningProject_2/NS2')

import torch
import json
from torchvision import transforms
import matplotlib.pyplot as plt
import utils.evaluation as evaluation
from torch.nn import DataParallel
import transformers
import re
import os
import logging
import random
import numpy as np
import cv2

logger = logging.getLogger(__name__)

def init_logger(is_main=True, is_distributed=False, filename=None):
    if is_distributed:
        torch.distributed.barrier()
    handlers = [logging.StreamHandler(sys.stdout)]
    if filename is not None:
        handlers.append(logging.FileHandler(filename=filename))
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main else logging.WARN,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )
    logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)
    logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)
    return logger
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


def extract_answer_candidate(out):
    token = '[/INST]'
    position = out.find(token)

    # Extract the characters after the token
    if position != -1:
        result = out[position + len(token):].strip()
    elif 'ASSISTANT' in out:
        result = out.split('ASSISTANT:')[1]
    else:
        result = out

    return result


def extract_elements(out):
    # Define the regex pattern to match the elements after the numbering
    pattern = r'\d+\.\s([^0-9]+?)(?=\s\d+\.|$)'

    # Find all matches using the regex pattern
    matches = re.findall(pattern, out)

    # Strip any leading/trailing whitespace from each match
    matches = [match.strip() for match in matches]

    return matches


def extract_elements_string(out):
    # Define the regex pattern to match the elements after the numbering
    pattern = r'\d+\.\s([^0-9]+?)(?=\s\d+\.|$)'

    # Find all matches using the regex pattern
    matches = re.findall(pattern, out)

    # Strip any leading/trailing whitespace from each match
    matches = [match.strip() for match in matches]

    return matches


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


def check_pronouns(text,debug=None):
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

def detect_image(image,predictor):
    image_object = []
    outputs = predictor(image)
    classids = outputs["instances"].pred_classes
    for id in classids:
        image_object.append(coco_categories[id])

    pred_boxes = outputs["instances"].pred_boxes

    return image_object,pred_boxes



def detect_image_ver1(image,predictor):
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

def ScoreEvaluation(answer,answer_list,dataset_length,index,logging):
    score = evaluation.cems( answer, answer_list, index, dataset_length,logging)
    return score


def get_score(string):
    score = {
        'a': 0.1,
        'b': 0.5,
        'c': 0.7,
        'd': 0.9,
        'e': 0.95,
        'f': 0,
    }
    # string = '(f) unknown'
    pattern = r'\((\w)\)'
    match = re.search(pattern, string)
    if match:
        letter = match.group(1)
        return score[letter]
    else:
        return 0