import pdb
import torch
import random
import json
import numpy as np
import os 
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import cv2



class SCIENCEQA_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,image_dir,split):

        self.data = data
        self.image_dir = image_dir
        with open('/data5/solomon/ScienceQA/data/scienceqa/problems.json', 'r') as fin:
            self.question_data = json.load(fin) 


    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        
        if 'answer' in example:
            answer_index = int(example['answer'])
            final_answer = example['choices'][answer_index]
            return final_answer
        else:
            return None


    def __getitem__(self, index):

        question_id = self.data[index]

        example = self.question_data[question_id]
    

        try: 
            image_path = self.image_dir + str(question_id) + '/'+'image.png'
            image = Image.open(image_path)
        except:
            image_path = 'no'
            image = Image.new('RGB', (448,448), (255, 255, 255))
            pass

        question = example['question']

        value = example['lecture']

        choices = example['choices']

        target = self.get_target(example)

        return {
            'question_id': question_id,
            'image_path': image_path,
            'image': image,
            'question': question,
            'target_str': example['answer'],
            'value': value,
            'choices' :choices
        }

    def get_example(self, index):
        # q, lecture, solu, choices, answer, image_id
        return self.data[index] 

    def extract_im_element(self, im_context):
        im_context_new, entities = [], []
        for i in im_context:
            key = i['title']
            if key not in entities:
                entities.append(key)
                im_context_new.append(i)

        return im_context_new
    


class TDIUC_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,coco_path,split='val'):

        self.data = data
        self.coco_dir = coco_path
        self.split = split

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            return target
        elif 'direct_answers' in example:
            return example['direct_answers']

        elif 'answers' in example:
            return example['answers'][0]['answer']

        else:
            return None


    def __getitem__(self, index):
        example = self.data[index]

        image_id = example['image_id']
        split = self.split

        image_path=get_coco14_path(split,image_id,self.coco_dir)

        image = Image.open(image_path)

        question = example['question']

        question_id = example['question_id']

        question_type = example['question_type']

        target = self.get_target(example)

        return {
            'index': index,
            'image_path': image_path,
            'image': image,
            'question': question,
            'target': target,
            'question_id': question_id,
            'question_type' :question_type
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
class AOKVQA_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,coco_path):

        self.data = data
        self.coco_dir = coco_path

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            return target
        elif 'direct_answers' in example:
            return example['direct_answers']
        else:
            return None


    def __getitem__(self, index):
        example = self.data[index]

        image_id = example['image_id']
        
        split = example['split']

        image_path=get_coco_path(split,image_id,self.coco_dir)

        image = Image.open(image_path)

        question = example['question']

        choices = example['choices']

        difficult_direct_answer = example['difficult_direct_answer']

        question_id = example['question_id']

        target = self.get_target(example)

        return {
            'index': index,
            'image_path': image_path,
            'image': image,
            'question': question,
            'choices': choices,
            'difficult_direct_answer': difficult_direct_answer,
            'target': target,
            'question_id': question_id,
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

class trainerV3_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 coco_path,
                 object_path):
        
        self.data = data
        self.coco_path = coco_path
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
   
        image_name = example['image_name']
        image_dir = image_name.split('_')[1]
        image_path = os.path.join(self.coco_path,image_dir,image_name)
        image = Image.open(image_path)
        caption = example['caption']

        question =  example['question']
    
        target = self.get_target(example)

        pred_classes, crop_images = self.get_sample_object(self.object_path, image_name, image)




        return {
            'index': index,
        
            'image_path':image_path,
            'image':image,
            'question':question,
            'caption':caption,
            'pred_classes':pred_classes,
            'crop_images':crop_images,
            'target': target,
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

class trainer_premise_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 coco_path):
        
        self.data = data
        self.coco_path = coco_path
        self.question_prefix = 'question:'
        

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'gt' in example:
            target = example['gt']
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
   
        question_id = example['question_id'][0]
        image_path = example['imagepath'][0]
        image = Image.open(image_path)
        question =  example['question'][0]
        all_premise =  example['all_premise']
        
    
        target = self.get_target(example)

        # pred_classes, crop_images = self.get_sample_object(self.object_path, image_name, image)




        return {
            'question_id': question_id,
            'image_path':image_path,
            'image':image,
            'question':question,
            'target': target,
            'all_premise':all_premise
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



class FOL_2Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 image_path,
                 n_ex_context,
                 context_prefix='context:',
                 question_prefix='question:',
                 em_title_prefix='entity:',
                 em_passage_prefix='description:',):
        
        self.data = data
        self.image_path = image_path
        self.context_prefix = context_prefix
        self.n_ex_context = n_ex_context
        self.question_prefix = question_prefix
        self.em_title_prefix = em_title_prefix
        self.em_passage_prefix = em_passage_prefix
        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            return target + ' </s>'
        elif 'answers_list' in example:
            return random.choice(example['answers_list']) + ' </s>'
        else:
            return None

    def __getitem__(self, index):
        example = self.data[index]

        image = example['image_name']
        image_dir = image.split('_')[1]
        image_path = os.path.join(self.image_path,image_dir,image)
        image = Image.open(image_path)


        question = self.question_prefix + " " + example['question']
    
        target = self.get_target(example)

        prompt = f""" 
        [INST]<image> "Detect two objects that help you infer the correct answer from question. According to knowledge in reality, please list 5 predicates between two objects to describe the object interaction.\n\n 
             Respond image captioning if it's uncertain. 
             {question}\n\n 
             Examples:\n" +\
            "Object: Animal, Vehicle \n" +\
            "Predicate: CanCatchUp(Animal X, Vehicle Y) \n" +\
            "Object: Substance, Substance \n" +\
            "Predicate: CanSubmerge(Substance X, Substance Y) \n" +\
            "Object: Material, Material \n" +\
            "Predicate: CanScratch(Material X, Material Y) \n" +\
            "Object: Show, Artwork \n" +\
            "Predicate: CanBeAdaptedFrom(Show X, Artwork Y) \n\n" + \
            "Object: " + concept1 + ", " + concept2 + " \n" + \
            "Predicate: \n" 
        [/INST]
        """

        return {
            'index': index,
            'image':image,
            'question':prompt,
            'orginal_question':question,
            'target': target,
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



class FOL_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 image_path,
                 n_ex_context,
                 context_prefix='context:',
                 question_prefix='question:',
                 em_title_prefix='entity:',
                 em_passage_prefix='description:',):
        
        self.data = data
        self.image_path = image_path
        self.context_prefix = context_prefix
        self.n_ex_context = n_ex_context
        self.question_prefix = question_prefix
        self.em_title_prefix = em_title_prefix
        self.em_passage_prefix = em_passage_prefix
        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            return target + ' </s>'
        elif 'answers_list' in example:
            return random.choice(example['answers_list']) + ' </s>'
        else:
            return None

    def __getitem__(self, index):
        example = self.data[index]

        image = example['image_name']
        image_dir = image.split('_')[1]
        image_path = os.path.join(self.image_path,image_dir,image)
        image = Image.open(image_path)


        question = self.question_prefix + " " + example['question']
    
        target = self.get_target(example)

        if 'ex_ctxs' in example and self.n_ex_context is not None:
            f = self.em_title_prefix + " {} \n " + self.em_passage_prefix + " {} \n"
            if len(example['ex_ctxs']) < self.n_ex_context:
                contexts = [example['ex_ctxs'][0]] * (self.n_ex_context - len(example['ex_ctxs'])) \
                           + example['ex_ctxs']
            else:
                contexts = example['ex_ctxs'][:self.n_ex_context]
            passages = [f.format(c['title'], c['text']) for c in contexts]
        
            if len(contexts) == 0:
                contexts = [question]
        else:
            passages, scores = None, None

        instruction = 'Answer the question base on the given information. \n Respond in this format "Answer:Unsure , Rationle:" if not sure about the answer. Think step by step using entity and description. '
        instruction_1 = 'Answer the question base on the given information. \n Respond "Answer:Unsure , Rationle:" if not sure about the answer.Think step by step. '
        instruction_2 = 'Answer the question base on the given information. Answer in word. Think step by step. \n Respond in this format "Answer:{answer} , Rationle:{question}". '
        instruction_3 = 'From the image answer the question in word. Think in logic rule.\n'
        format = 'Respond in Answer:(), Logic Rule:() format.\n'


        # prompt = "[INST] <image>\n"+ instruction+'\n'+  ''.join(passages) + question+" [/INST]"
        prompt = "[INST] <image>\n" +instruction_3+ question + format+" [/INST]"


        return {
            'index': index,
            'image':image,
            'question':prompt,
            'target': target,

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


class OKVQA_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 image_path ):

        self.data = data
        self.image_path = image_path

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            return target + ' </s>'
        elif 'answers_list' in example:
            return random.choice(example['answers_list']) + ' </s>'
        else:
            return None

    def crop_image(self,image_path, box_coordinates):
        """
        주어진 좌표로 이미지를 자르는 함수

        Parameters:
        image_path (str): 자를 이미지의 경로
        box_coordinates (tuple): 자를 좌표 (left, upper, right, lower)

        Returns:
        Image: 자른 이미지 객체
        """
        try:
            # 이미지 열기
            image = Image.open(image_path)

            # 이미지 자르기
            cropped_image = image.crop(box_coordinates)

            return cropped_image
        except Exception as e:
            print(f"Error: {e}")
            return None

    def __getitem__(self, index):
        example = self.data[index]

        image_name = example['image_name']
        image_dir = image_name.split('_')[1]

        image_path = os.path.join(self.image_path, image_dir, image_name)
        image = Image.open(image_path)

        boxes = example['boxes']

        crop_images=[]

        for box in boxes:

            # print(box)
            cropped_image = self.crop_image(image_path, box)
            crop_images.append(cropped_image)

        target = self.get_target(example)


        prompt = "[INST] <image>\n What is shown in this image? [/INST] Answer:"

        return {
            'index': index,
            'image_name':image_name,
            'image': image,
            'crop_images':crop_images,
            'question': prompt,
            'target': target,

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



def load_aokvqa(aokvqa_dir, split, version='v1p0'):
    assert split in ['train', 'val', 'test', 'test_w_ans']
    dataset = json.load(open(
        os.path.join(aokvqa_dir, f"aokvqa_{version}_{split}.json")
    ))
    return dataset

def get_coco_path(split, image_id, coco_dir):
    return os.path.join(coco_dir, f"{split}2017", f"{image_id:012}.jpg")

def get_coco14_path(split, image_id, coco_dir):
    return os.path.join(coco_dir, f"{split}2014", f"COCO_{split}2014_{image_id:012}.jpg")


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

    return data


def collate_fn(batch):
    images = [item['image'] for item in batch]
    prompts = [item['question'] for item in batch]

    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    inputs = processor(prompts, images, return_tensors="pt", padding=True)

    question = [item['orginal_question'] for item in batch]
    targets = [item['target'] for item in batch]

    return inputs,images,question, targets



def caption_collate_fn(batch):
    image_name = [item['image_name'] for item in batch]
    images = [item['image'] for item in batch]
    prompts = [item['question'] for item in batch]
    crop_images = [item['crop_images'] for item in batch]

    #
    # processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    # inputs = processor(prompts, images, return_tensors="pt", padding=True)
    #
    # targets = [item['target'] for item in batch]

    return image_name,images,prompts,crop_images


def aokvqa_collate_fn(batch):
    # 'index': index,
    # 'image_path': image_path,
    # 'image': image,
    # 'question': question,
    # 'choices': choices,
    # 'difficult_direct_answer': difficult_direct_answer,
    # 'target': target,

    images = [item['image'] for item in batch]
    image_path = [item['image_path'] for item in batch]
    question = [item['question'] for item in batch]
    choices = [item['choices'] for item in batch]
    target = [item['target'] for item in batch]
    difficult_direct_answer = [item['difficult_direct_answer'] for item in batch]
    question_id = [item['question_id'] for item in batch]


    return images,image_path,question,choices,difficult_direct_answer,target,question_id


def tdiuc_collate_fn(batch):
    # 'index': index,
    # 'image_path': image_path,
    # 'image': image,
    # 'question': question,
    # 'target': target,
    # 'question_id': question_id,

    images = [item['image'] for item in batch]
    image_path = [item['image_path'] for item in batch]
    question = [item['question'] for item in batch]
    target = [item['target'] for item in batch]
    question_id = [item['question_id'] for item in batch]
    question_type = [item['question_type'] for item in batch]


    return images,image_path,question,target,question_id,question_type


def scienceqa_collate_fn(batch):
    #    'question_id': question_id,
    #         'image_path': image_path,
    #         'image': image,
    #         'question': question,
    #         'target_str': target,
    #         'value': value,
    #         'choices' :choices
    question_id = [item['question_id'] for item in batch]
    images = [item['image'] for item in batch]
    image_path = [item['image_path'] for item in batch]
    question = [item['question'] for item in batch]
    target_str = [item['target_str'] for item in batch]
    value = [item['value'] for item in batch]
    choices = [item['choices'] for item in batch]


    return question_id,images,image_path,question,target_str,value,choices




def trainer_v3collate_fn(batch):
    #'index': index,
    # 'image':image,
    # 'question':question,
    # 'pred_classes':pred_classes,
    # 'crop_images':crop_images,
    # 'target': target,

    images = [item['image'] for item in batch]
    image_path = [item['image_path'] for item in batch]
    question = [item['question'] for item in batch]
    pred_classes = [item['pred_classes'] for item in batch]
    crop_images = [item['crop_images'] for item in batch]
    target = [item['target'] for item in batch]
    caption = [item['caption'] for item in batch]


    return images,image_path,question,caption,pred_classes, crop_images,target


def trainer_premise_collate_fn(batch):
#   'question_id': question_id,
#             'image_path':image_path,
#             'image':image,
#             'question':question,
#             'target': target,
#             'all_premise':all_premise

    images = [item['image'] for item in batch]
    image_path = [item['image_path'] for item in batch]
    question = [item['question'] for item in batch]
    question_id = [item['question_id'] for item in batch]
    all_premise = [item['all_premise'] for item in batch]
    target = [item['target'] for item in batch]
 


    return question_id, images,image_path,question,target,all_premise
