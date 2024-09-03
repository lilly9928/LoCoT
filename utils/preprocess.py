import sys
sys.path.append('/home/user2/code/WIL_DeepLearningProject_2/NS/Symbolic_LLM')
import data_loader.aokvqa_dataloader_with_image as data
import os
import utils.utils as utils
import cv2
from PIL import Image
import json
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from tqdm import tqdm 

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("../configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = '/data2/KJE/weights/model_final_68b088.pkl'
predictor = DefaultPredictor(cfg)

coco_categories = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

coco_path = '/data2/KJE/VQA/COCO/images'
_data = '/data2/KJE/ModelLogs/revive/processed_data/test.pkl'
save_path = '/data2/KJE/ModelLogs/Logic_LLama/data/preprocess_okvqa'
okvqa_data = data.load_data( _data)

save_datas = []

for idx in tqdm(range(len(okvqa_data))):
    image = okvqa_data[idx]['image_name']
    image_dir = image.split('_')[1]
    image_path = os.path.join(coco_path,image_dir,image)

    image_ = cv2.imread(image_path)
    outputs = predictor(image_)


    pred_ids= outputs["instances"].pred_classes
    pred_boxes = [box.tolist() for box in outputs["instances"].pred_boxes]

    pred_classes = []
    for id in pred_ids:
        pred_classes.append(coco_categories[id])

    save_datas.append({
        'image_name': image,
        'pred_classes':pred_classes,
        'pred_boxes':pred_boxes
    })

with open(os.path.join(save_path, f'okvqa_test_object.json'), 'w', encoding='utf-8') as f:
            json.dump(save_datas, f, ensure_ascii=False, indent=4)



