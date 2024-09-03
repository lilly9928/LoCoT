import pdb
import torch
import random
import json
import numpy as np
import os 
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import cv2
import glob


class trainerV3_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 image_dir,
                 object_path):
        
        self.data = data
        self.image_dir = image_dir
        self.object_path = object_path
        self.question_prefix = 'question:'
        

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            return target
        elif 'answers_list' in example:
            return example['answers_list']
        else:
            return None
        

    def get_sample_object(self,path,image_name,image):
        with open(path, 'r') as fin:
            data = json.load(fin)
        for i in range(len(data)):
            if image_name == data[i]['image_name']:
                found = data[i]
                pred_classes = found['pred_classes']
                pred_boxes = found['pred_boxes']
                crop_images = []
                for box in pred_boxes:
                    crop_images.append(image.crop(box))
                
                return pred_classes, crop_images

    def __getitem__(self, index):
        example = self.data[index]

        questionId = example['questionId']
        image_name = example['imageId']+'.jpg'
        image_path = os.path.join(self.image_dir,image_name)
        image = Image.open(image_path)

        question =  example['question']
    
        target = example['answer']
        fullanswer = example['fullAnswer']

        type = example['type']
        group = example['group']

        return {
            'index': index,
            'questionId':questionId,
            'image_path':image_path,
            'image':image,
            'question':question,
            'fullanswer': fullanswer,
            'type': type,
            'group': group,
            'target':target
        }


    def get_example(self, index):
        return self.data[index]

    def extract_im_element(self, im_context):
        im_context_new, entities = [], []
        for i in im_context:
            key = i['title']
            if key not in entities:
                entities.append(key)
                im_context_new.append(i)

        return im_context_new





def load_data(data_path=None, global_rank=-1, world_size=-1):
    assert data_path
    if data_path.endswith('.jsonl'):
        data = open(data_path, 'r')
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    else:
        data = np.load(data_path, allow_pickle=True)
    examples = []
    for k, example in enumerate(data):
        examples.append(example)

    if data_path is not None and data_path.endswith('.jsonl'):
        data.close()

    return examples


def loadFile(name):
    # load standard json file
    if os.path.isfile(name):
        with open(name) as file:
            data = json.load(file)
    # load file chunks if too big
    elif os.path.isdir(name.split(".")[0]):
        data = {}
        chunks = glob.glob('{dir}/{dir}_*.{ext}'.format(dir = name.split(".")[0], ext = name.split(".")[1]))
        for chunk in chunks:
            with open(chunk) as file:
                data.update(json.load(file))
    else:
        raise Exception("Can't find {}".format(name))
    return data


def trainer_v3collate_fn(batch):
    # 'image_path': image_path,
    # 'image': image,
    # 'question': question,
    # 'fullanswer': fullanswer,
    # 'type': type,
    # 'group': group,


    images = [item['image'] for item in batch]
    image_path = [item['image_path'] for item in batch]
    question = [item['question'] for item in batch]
    questionId = [item['questionId'] for item in batch]
    target = [item['target'] for item in batch]
    type = [item['type'] for item in batch]
    group = [item['group'] for item in batch]
    fullanswer = [item['fullanswer'] for item in batch]


    return questionId,images,image_path,question,target,type, group,fullanswer


