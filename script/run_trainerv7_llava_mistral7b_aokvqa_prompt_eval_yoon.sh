# 분산 훈련을 위한 기본 설정
NUM_GPUS=6 # 사용할 GPU 수
NODE_RANK=0  # 현재 노드의 순위 (멀티 노드 훈련 시 필요)
NUM_NODES=1  # 사용할 노드 수 (멀티 노드 훈련 시 필요)
MASTER_ADDR=localhost  # 마스터 노드의 주소
MASTER_PORT=25683  # 마스터 노드의 포트
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

# Python 스크립트 실행

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    trainer/trainer_v7_logic_with_image_premise_onlyvlm.py \
    --lvlm_name="llava-hf/llava-v1.6-vicuna-7b-hf" \
    --name='aokvqa/aokvqa_llava_vicuna'\
    --data='aokvqa'\
    --data_path='/data2/KJE/VQA/aokvqa/aokvqa/aokvqa_v1p0_val.json'
   

<<'END'
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    trainer/trainer_v7_logic_with_image_premise_onlyvlm_yoon.py \
    --lvlm_name="llava-hf/llava-v1.6-vicuna-7b-hf" \
    --prompt="cot" \
    --name='aokvqa/aokvqa_llava_vicuna_with_CoT'\
    --data='aokvqa'\
    --data_path='/data2/KJE/VQA/aokvqa/aokvqa/aokvqa_v1p0_val.json'


CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    trainer/trainer_v7_logic_with_image_premise_onlyvlm_yoon.py \
    --lvlm_name="llava-hf/llava-v1.6-vicuna-7b-hf" \
    --prompt="no" \
    --name='aokvqa/aokvqa_llava_vicuna'\
    --data='aokvqa'\
    --data_path='/data2/KJE/VQA/aokvqa/aokvqa/aokvqa_v1p0_val.json'




CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    trainer/trainer_v7_logic_with_image_premise_onlyvlm_yoon.py \
    --lvlm_name="llava-hf/llava-1.5-7b-hf" \
    --prompt="long_cot" \
    --name='aokvqa/aokvqa_llava_1.5-7b-hf_with_Long_CoT'\
    --data='aokvqa'\
    --data_path='/data2/KJE/VQA/aokvqa/aokvqa/aokvqa_v1p0_val.json'


CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    trainer/trainer_v7_logic_with_image_premise_onlyvlm_yoon.py \
    --lvlm_name="llava-hf/llava-1.5-7b-hf" \
    --prompt="cot" \
    --name='aokvqa/aokvqa_llava_1.5-7b-hf_with_CoT'\
    --data='aokvqa'\
    --data_path='/data2/KJE/VQA/aokvqa/aokvqa/aokvqa_v1p0_val.json'

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    trainer/trainer_v7_logic_with_image_premise_onlyvlm_yoon.py \
    --lvlm_name="llava-hf/llava-1.5-7b-hf" \
    --prompt="no" \
    --name='aokvqa/aokvqa_llava_1.5-7b-hf'\
    --data='aokvqa'\
    --data_path='/data2/KJE/VQA/aokvqa/aokvqa/aokvqa_v1p0_val.json'




CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    trainer/trainer_v7_logic_with_image_premise_onlyvlm_yoon.py \
    --lvlm_name="llava-hf/llama3-llava-next-8b-hf" \
    --prompt="long_cot" \
    --name='aokvqa/aokvqa_llava_next_with_Long_CoT'\
    --data='aokvqa'\
    --data_path='/data2/KJE/VQA/aokvqa/aokvqa/aokvqa_v1p0_val.json'

    

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    trainer/trainer_v7_logic_with_image_premise_onlyvlm_yoon.py \
    --lvlm_name="llava-hf/llama3-llava-next-8b-hf" \
    --prompt="cot" \
    --name='aokvqa/aokvqa_llava_next_with_CoT'\
    --data='aokvqa'\
    --data_path='/data2/KJE/VQA/aokvqa/aokvqa/aokvqa_v1p0_val.json'


CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    trainer/trainer_v7_logic_with_image_premise_onlyvlm_yoon.py \
    --lvlm_name="llava-hf/llama3-llava-next-8b-hf" \
    --prompt="no" \
    --name='aokvqa/aokvqa_llava_next'\
    --data='aokvqa'\
    --data_path='/data2/KJE/VQA/aokvqa/aokvqa/aokvqa_v1p0_val.json'
END