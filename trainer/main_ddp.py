import sys
sys.path.append('/home/user2/code/WIL_DeepLearningProject_2/NS2')

import os
import torch
from torch.utils.data import DataLoader, DistributedSampler
import json
from torchvision import transforms
import okvqa_data as data_module  # 모듈 이름을 변경하여 전역 변수와 충돌 방지
from options import Options
from torch.nn.parallel import DistributedDataParallel as DDP
import transformers
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import re
from tqdm import tqdm

def extract_question_answer_rationale(text):
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
    targets= normalize_string(targets)
    predictions = normalize_string(predictions)
    return predictions == targets

def main():
    # Initialize the process group
    torch.distributed.init_process_group(backend='nccl')

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    transformers.logging.set_verbosity_error()
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()
    opt.cache_dir = "/data2/KJE/hg_weight"

    image_path = '/data2/KJE/VQA/COCO/images'
    data_path = '/data2/KJE/ModelLogs/revive/processed_data/test.pkl'
    n_ex_context = 40
    batch_size = 1

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_examples = data_module.load_data(data_path)
    dataset = data_module.FOL_Dataset(data=train_examples, image_path=image_path, n_ex_context=n_ex_context)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler, collate_fn=data_module.collate_fn)

    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", cache_dir=opt.cache_dir)
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",
                                                              torch_dtype=torch.float16, low_cpu_mem_usage=True,
                                                              pad_token_id=2,cache_dir=opt.cache_dir)
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    results_data = []
    error_data = []
    correct_count = 0
    total_count = 0

    for batch in tqdm(dataloader):
        inputs, targets = batch
        inputs = {k: v.to(local_rank) for k, v in inputs.items()}

        outputs = model.module.generate(**inputs, max_new_tokens=200)
        results = [processor.decode(output, skip_special_tokens=True) for output in outputs]

        question, answer, rationale = extract_question_answer_rationale(results[0])

        if rationale:
            results_data.append({
                "question": question,
                "answer": answer,
                "rationale": rationale,
                "targets":targets
            })
        else:
            error_data.append({
                "question": question,
                "answer": answer,
                "targets": targets
            })

        for target in targets:
            if calculate_accuracy(answer, target):
                correct_count += 1
            total_count += 1


    accuracy = correct_count / total_count * 100 if total_count > 0 else 0

    save_path = '/data2/KJE/ModelLogs/Logic_LLama/llava-v1.6-mistral-7b_orignal'
    temp_save_path = os.path.join(save_path, f'temp_results_rank_{local_rank}.json')
    temp_error_save_path = os.path.join(save_path, f'temp_error_rank_{local_rank}.json')

    with open(temp_save_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=4)

    with open(temp_error_save_path, 'w', encoding='utf-8') as f:
        json.dump(error_data, f, ensure_ascii=False, indent=4)

    # Ensure all processes complete their writing
    torch.distributed.barrier()

    if local_rank == 0:
        # Merge results from all ranks
        final_results = []
        final_error_results = []

        for rank in range(torch.distributed.get_world_size()):
            temp_file = os.path.join(save_path, f'temp_results_rank_{rank}.json')
            with open(temp_file, 'r', encoding='utf-8') as f:
                final_results.extend(json.load(f))
            os.remove(temp_file)

            temp_error_file = os.path.join(save_path, f'temp_error_rank_{rank}.json')
            with open(temp_error_file, 'r', encoding='utf-8') as f:
                final_error_results.extend(json.load(f))
            os.remove(temp_error_file)

        with open(os.path.join(save_path, f'okvqa_test_results_acc_{accuracy}.json'), 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)

        with open(os.path.join(save_path, 'okvqa_test_error.json'), 'w', encoding='utf-8') as f:
            json.dump(final_error_results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
