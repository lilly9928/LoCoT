import sys
sys.path.append('/home/user2/code/WIL_DeepLearningProject_2/NS2')

import torch
from torch.utils.data import DataLoader, DistributedSampler
import json
from torchvision import transforms
import okvqa_data as data
from options import Options
from torch.nn.parallel import DistributedDataParallel as DDP
import transformers
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import re
import os
from tqdm import tqdm


def extract_answer(text):
    # Use regular expression to find text between [INST] and [/INST]
    answer_match = re.search(r'Answer:\s*(.*)\s*$', text)

    if answer_match:
        return answer_match.group(1)
    else:
        return None


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

    # 기본 설정
    image_path = '/data2/KJE/VQA/COCO/images'
    data_path = '/data2/KJE/ModelLogs/revive/processed_data/test.pkl'
    batch_size = 1


    # 데이터셋 불러오기
    train_examples = data.load_data(data_path)

    dataset = data.OKVQA_Dataset(data=train_examples, image_path=image_path)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler, collate_fn=data.caption_collate_fn)

    # Model and processor initialization
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", cache_dir=opt.cache_dir)
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",
                                                              torch_dtype=torch.float16, low_cpu_mem_usage=True, pad_token_id=2, cache_dir=opt.cache_dir)

    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    results_data = []
    crop_captions = []

    # Process each batch
    for batch in tqdm(dataloader):
        with torch.cuda.amp.autocast():
            images_name, images, prompts, crop_images = batch

            inputs = processor(prompts, images, return_tensors="pt", padding=True)
            inputs = {k: v.to(local_rank) for k, v in inputs.items()}

            outputs = model.module.generate(**inputs, max_new_tokens=200)
            results = [processor.decode(output, skip_special_tokens=True) for output in outputs]
            caption=extract_answer(results[0])

            for crop_image in crop_images[0]:
                inputs = processor(prompts, crop_image, return_tensors="pt", padding=True)
                inputs = {k: v.to(local_rank) for k, v in inputs.items()}

                crop_outputs = model.module.generate(**inputs, max_new_tokens=200)
                crop_results = [processor.decode(output, skip_special_tokens=True) for output in crop_outputs]
                crop_result = extract_answer(crop_results[0])
                crop_captions.append(crop_result)

        results_data.append({
            'image_name': images_name,
            'caption': caption,
            'object_captions': crop_captions
        })

    # Save intermediate results for each rank
    save_path = '/data2/KJE/ModelLogs/Logic_LLama/data/preprocess_okvqa'
    temp_save_path = os.path.join(save_path, f'temp_results_rank_{local_rank}.json')

    with open(temp_save_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=4)

    # Ensure all processes complete their writing
    torch.distributed.barrier()

    if local_rank == 0:
        # Merge results from all ranks
        final_results = []

        for rank in range(torch.distributed.get_world_size()):
            temp_file = os.path.join(save_path, f'temp_results_rank_{rank}.json')
            with open(temp_file, 'r', encoding='utf-8') as f:
                final_results.extend(json.load(f))
            os.remove(temp_file)

        with open(os.path.join(save_path, f'caption_gen_llava-v1.6-mistral-7b-hf_okvqa_test.json'), 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
