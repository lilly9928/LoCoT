# 분산 훈련을 위한 기본 설정
NUM_GPUS=4  # 사용할 GPU 수
NODE_RANK=0  # 현재 노드의 순위 (멀티 노드 훈련 시 필요)
NUM_NODES=4  # 사용할 노드 수 (멀티 노드 훈련 시 필요)
MASTER_ADDR=localhost  # 마스터 노드의 주소
MASTER_PORT=25677  # 마스터 노드의 포트


# Python 스크립트 실행
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    captionGeneration.py