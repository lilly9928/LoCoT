import argparse
import pathlib
import json
import logging
import os
import regex
import string
import numpy as np

def load_okvqa(okvqa_dir, split):
    assert split in ['train', 'val', 'test']
    dataset = json.load(open(
        os.path.join(okvqa_dir, f"okvqa_{split}.json")
    ))
    return dataset

logger = logging.getLogger(__name__)

"""
Evaluation code from DPR: https://github.com/facebookresearch/DPR
"""


#################################################
########        READER EVALUATION        ########
#################################################

# Normalization from SQuAD evaluation script https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score( prediction, ground_truth, all_ground_truth, index, length, gt_number,logger):
    try:
        length = int(np.ceil(length / 1))

        if index is not None and gt_number == 0:
            logger.warning('Batch: [{}/{}]  |  Prediction: {:12}  |  Ground Truth: {}'.format(index, length, prediction,
                                                                                     all_ground_truth))
    except Exception:
        pass

    return normalize_answer(prediction) == normalize_answer(ground_truth)


def cems( prediction, ground_truths, logger,index=None, length=None):
    correct_num = 0
    for i, gt in enumerate(ground_truths):
        correct_num += exact_match_score(prediction, gt, ground_truths, index, length, i,logger)
    cur_acc = min(float(correct_num / 3), 1.0)
    return cur_acc


def ScoreEvaluation(answer,answer_list,dataset_length,index):
    score = cems( answer, answer_list, logger, index, dataset_length)
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--okvqa_dir', type=pathlib.Path, required=True, dest='okvqa_dir')
    parser.add_argument('--split', type=str, choices=['test', 'val'], required=True)
    parser.add_argument('--pred', type=str, required=True, dest='prediction_file')
    args = parser.parse_args()
    
    gt_dir = '/data2/KJE/VQA/okvqa/okvqa_annotations/mscoco_val2014_annotations.json'
    
    dataset = load_okvqa(args.okvqa_dir, args.split)
    with open(gt_dir, 'r') as f:
        gt_data = json.load(f)
        
    with open(args.prediction_file, 'r') as f:
        pred_data = json.load(f)
        
    exactmatch=[]
    annot_data = []
    
    for item in gt_data['annotations']:
        annot_data.append(item)
        
        
    for i in range(len(annot_data)):
        sample = annot_data[i]
        pred_answer = pred_data[sample['question_id']]['prediction']
        
        if pred_answer != None :
            score = ScoreEvaluation(pred_answer,pred_data[sample['questionId']]['gt'], len(annot_data), i)
        else:
            exactmatch.append(0.0)
        
        exactmatch.append(score)
            
        print("accuracy =", 100 * np.mean(exactmatch))

