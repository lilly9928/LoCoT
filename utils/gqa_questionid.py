import json
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def process_sample(args):
    sample, gqa_question_map = args
    question = sample['question'][0]
    question_id = gqa_question_map.get(question)
    if question_id is not None:
        return {
            'questionId': int(question_id),
            'prediction': sample['final_answer']
        }
    return None

def main():
    predict_path = '/data3/KJE/modelLog/Logic_LLama/gqa_llava_mistral7b_0724_all/testdev_predictions_raw.json'
    with open(predict_path, 'r') as f:
        predict = json.load(f)

    gqa_file = '/data2/NS/GQA/preprocess/all_testdev_data.json'
    with open(gqa_file, 'r') as f:
        gqa = json.load(f)

    # Create a dictionary for faster lookup
    gqa_questions = gqa['questions']
    gqa_question_map = {q['question']: q['questionId'] for q in gqa_questions}

    # Use parallel processing
    with Pool(cpu_count()) as p:
        args = [(sample, gqa_question_map) for sample in predict]
        results = list(tqdm(p.imap_unordered(process_sample, args), total=len(predict)))
    
    # Filter out None results
    new_predict = [res for res in results if res is not None]

    print(len(new_predict))
    output_path = '/data3/KJE/modelLog/Logic_LLama/gqa_llava_mistral7b_0724_all/gqa_predict_eval.json'
    with open(output_path, 'w') as f:
        json.dump(new_predict, f)

if __name__ == '__main__':
    main()
           