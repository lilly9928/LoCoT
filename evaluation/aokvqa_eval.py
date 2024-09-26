
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

    i = 1
    for q in dataset.keys():
        # print(f'{i} - Id:', q)
        if q not in preds.keys():
            # print("q",q,"not in preds.keys")
            acc.append(0.0)
            continue
        i += 1 
        pred = preds[q]
        choices = dataset[q]['choices']
        direct_answers = dataset[q]['direct_answers']

        ## Multiple Choice setting
        if multiple_choice:
            if strict:
                assert pred in choices, 'Prediction must be a valid choice'
            correct_choice_idx = dataset[q]['correct_choice_idx']
            acc.append( float(pred.lower() == choices[correct_choice_idx]) )
        ## Direct Answer setting
        else:
            num_match = sum([pred.lower() == da for da in direct_answers])
            vqa_acc = min(1.0, num_match / 3.0)
            acc.append(vqa_acc)

    print('sum', sum(acc), 'len', len(acc), 'acc', sum(acc)/len(acc)*100)
    acc = sum(acc) / len(acc) * 100

    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aokvqa-dir', default='/data2/KJE/VQA/aokvqa/aokvqa',type=pathlib.Path, required=True, dest='aokvqa_dir')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test_w_ans'], required=True)
    parser.add_argument('--preds', type=str, required=True, dest='prediction_files')
    args = parser.parse_args()

    dataset = load_aokvqa(args.aokvqa_dir, args.split)

    for prediction_file in glob.glob(args.prediction_files):
        predictions_ = json.load(open(prediction_file, 'r'))

        print(len(predictions_))

        # Multiple choice

        mc_predictions = {}

        predictions={}
        for item in predictions_:
            predictions.update(item)

        # Direct Answer

        da_predictions = {}

        for q in predictions.keys():
            if 'direct_answer' in predictions[q].keys():
                da_predictions[q] = predictions[q]['direct_answer']

        if da_predictions != {}:
            da_acc = eval_aokvqa(
                dataset,
                da_predictions,
                multiple_choice=False,
                strict=False
            )
            print(prediction_file, 'DA', da_acc)
