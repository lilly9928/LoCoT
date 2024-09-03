import argparse
import pathlib
import json
import glob
import os



def load_aokvqa(aokvqa_dir, split, version='v1p0'):
    assert split in ['train', 'val', 'test', 'test_w_ans']
    dataset = json.load(open(
        os.path.join(aokvqa_dir, f"aokvqa_{version}_{split}.json")
    ))
    return dataset

def get_coco_path(split, image_id, coco_dir):
    return os.path.join(coco_dir, f"{split}2017", f"{image_id:012}.jpg")

def eval_aokvqa(dataset, preds, multiple_choice=False, strict=True):

    if isinstance(dataset, list):
        dataset = { dataset[i]['question_id'] : dataset[i] for i in range(len(dataset)) }

    if multiple_choice is False:
        dataset = {k:v for k,v in dataset.items() if v['difficult_direct_answer'] is False}

    if strict:
        dataset_qids = set(dataset.keys())
        preds_qids = set(preds.keys())
        assert dataset_qids.issubset(preds_qids)

    # dataset = q_id (str) : dataset element (dict)
    # preds = q_id (str) : prediction (str)

    acc = []
    fail_case = []

    for q in dataset.keys():
        if q not in preds.keys():
            acc.append(0.0)
            continue

        pred = preds[q]
        choices = dataset[q]['choices']
        direct_answers = dataset[q]['direct_answers']

        ## Multiple Choice setting
        if multiple_choice:
            if strict:
                assert pred in choices, 'Prediction must be a valid choice'
            correct_choice_idx = dataset[q]['correct_choice_idx']
            acc.append( float(pred == choices[correct_choice_idx]) )
        ## Direct Answer setting
        else:
            num_match = sum([pred == da for da in direct_answers])
            vqa_acc = min(1.0, num_match / 3.0)
            acc.append(vqa_acc)
            if vqa_acc == 0:
                fail_case.append({
                    "predict" : pred,
                    "GT": direct_answers,
                    "id": q
                })

    acc = sum(acc) / len(acc) * 100

    return acc, fail_case



def make_failCase_file(data_path,dataset,file_dir,predictData_path):

    for prediction_file in glob.glob(data_path):
        val_data = json.load(open(prediction_file, 'r'))

    for predict in glob.glob(predictData_path):
        predict_data = json.load(open(predict, 'r'))

    da_predictions = {}
    whole_fail_data = []
    whole_fail_data_final = []

    for i in range(len(val_data)):
        for q in val_data[i].keys():
            if 'direct_answer' in val_data[i][q].keys():
                da_predictions[q] = val_data[i][q]['direct_answer'].lower().strip()

    if da_predictions != {}:
        da_acc, fail_case = eval_aokvqa(
            dataset,
            da_predictions,
            multiple_choice=False,
            strict=False
        )

    for case in fail_case:
        for data in dataset:
            orignal_id = data['question_id']
            id = case['id']

            if id == orignal_id:
                whole_fail_data.append({
                    "image_id":data['image_id'],
                    "question":data['question'],
                     "direct_answers":data['direct_answers'],
                      "predict":case['predict']
                })

    for case in whole_fail_data:
        for data in predict_data:
            image_path = data['imagepath'][0]
            pr_id = image_path.split('/data2/KJE/VQA/COCO/images/val2017/')[-1].split('.')[0]
            id = case['image_id']
            id = f"{id:012}"

            if id == pr_id:
                whole_fail_data_final.append({
                    "imagepath": data['imagepath'][0],
                    "question": data['question'][0],
                    "premise_for_answer": data['premise_for_answer'],
                    "object_candidate": data['object_candidate'],
                    "new_premise": data['new_premise'],
                    "gt": data['gt'][0],
                    "final_answer": data['final_answer'],
                    "difficult_direct_answer": data['difficult_direct_answer']
                })


    with open(os.path.join(file_dir,'aokvqa_fail_case_full.json'), "w") as json_file:
        json.dump(whole_fail_data_final, json_file, indent=4)

    print("Saved FAIL CASE PATH: ",os.path.join(file_dir,'aokvqa_fail_case_full.json'),"Acc. ",da_acc)

