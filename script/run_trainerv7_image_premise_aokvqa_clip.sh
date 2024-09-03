# 분산 훈련을 위한 기본 설정
NUM_GPUS=1 # 사용할 GPU 수
NODE_RANK=0  # 현재 노드의 순위 (멀티 노드 훈련 시 필요)
NUM_NODES=1  # 사용할 노드 수 (멀티 노드 훈련 시 필요)
MASTER_ADDR=localhost  # 마스터 노드의 주소
MASTER_PORT=25690 # 마스터 노드의 포트
CUDA_VISIBLE_DEVICES=5



# Python 스크립트 실행
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    trainer/trainer_v7_logic_with_image_premise_clip.py \
    --name='aokvqa/aokvqa_v7_clip_2c2p_f5_result' \
    --data='aokvqa' \
    --data_path='/data2/KJE/VQA/aokvqa/aokvqa/aokvqa_v1p0_val.json' \
    --num_object_premises=2 \
    --num_llm_premises=2

