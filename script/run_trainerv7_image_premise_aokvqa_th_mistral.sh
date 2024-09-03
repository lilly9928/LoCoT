# 분산 훈련을 위한 기본 설정
NUM_GPUS=1 # 사용할 GPU 수
NODE_RANK=0  # 현재 노드의 순위 (멀티 노드 훈련 시 필요)
NUM_NODES=1  # 사용할 노드 수 (멀티 노드 훈련 시 필요)
MASTER_ADDR=localhost  # 마스터 노드의 주소
MASTER_PORT=25670  # 마스터 노드의 포트
CUDA_VISIBLE_DEVICES=3
NUM_TH="0.7"

for num in 0.7
    do 
    # Python 스크립트 실행
    
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=$NUM_NODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        trainer/trainer_v7_logic_with_image_premise_yoon.py \
        --name="aokvqa/aokvqa_blip_with_Long_CoT"\
        --data='aokvqa'\
        --data_path='/data2/KJE/VQA/aokvqa/aokvqa/aokvqa_v1p0_val.json'\
        --threshold=$num\
        --lvlm_name="Salesforce/blip2-flan-t5-xl"\
        --cot="cot"\
    
    

    done




<<'END'
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=$NUM_NODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        trainer/trainer_v7_logic_with_image_premise.py \
        --name="aokvqa/aokvqa_llava_with_image_premise_vicuna_13b_long_cot"\
        --data='aokvqa'\
        --data_path='/data2/KJE/VQA/aokvqa/aokvqa/aokvqa_v1p0_val.json'\
        --threshold=$num\
        --lvlm_name="llava-hf/llava-v1.6-vicuna-13b-hf"\
        --cot="long"\

   

    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=$NUM_NODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        trainer/trainer_v7_logic_with_image_premise_yoon.py \
        --name="aokvqa/aokvqa_vilt_with_Long_CoT"\
        --data='aokvqa'\
        --data_path='/data2/KJE/VQA/aokvqa/aokvqa/aokvqa_v1p0_val.json'\
        --threshold=$num\
        --lvlm_name="dandelin/vilt-b32-mlm"\
        --cot="long"\



        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun \
        --nproc_per_node=$NUM_GPUS \
        --nnodes=$NUM_NODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        trainer/trainer_v7_logic_with_image_premise_yoon.py \
        --name="aokvqa/aokvqa_blip_with_Long_CoT"\
        --data='aokvqa'\
        --data_path='/data2/KJE/VQA/aokvqa/aokvqa/aokvqa_v1p0_val.json'\
        --threshold=$num\
        --lvlm_name="Salesforce/blip2-flan-t5-xl"\
        --cot="cot"\
        --debug='True'\
END