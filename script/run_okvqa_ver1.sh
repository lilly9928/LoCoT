## 분산 훈련을 위한 기본 설정
#NUM_GPUS=3  # 사용할 GPU 수
#NODE_RANK=0  # 현재 노드의 순위 (멀티 노드 훈련 시 필요)
#NUM_NODES=3  # 사용할 노드 수 (멀티 노드 훈련 시 필요)
#MASTER_ADDR=localhost  # 마스터 노드의 주소
#MASTER_PORT=25677  # 마스터 노드의 포트
CUDA_VISIBLE_DEVICES=3

# Python 스크립트 실행

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun \
    trainer/trainer_final_v1.py\
    --image_path='/data2/KJE/VQA/COCO/images'\
    --data_path='/data2/KJE/ModelLogs/revive/processed_data/test.pkl'\
    --name='LogicM_v1'\

