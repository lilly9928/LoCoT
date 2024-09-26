# Evaluation
Please prepare `predictions_{data}_{split}.json` files (for `data: {aokvqa, okvqa, gqa}` and `split: {test, val, testdev}`, 'testdev' only for 'gqa' data). 

## A-OKVQA
To evaluate the model on the validation set of the A-OKVQA dataset, run the following command:
```
python evaluation/aokvqa_eval.py --aokvqa-dir ${AOKVQA_DIR} --split val --preds ./predict_aokvqa_val.json
```
## OK-VQA
To evaluate the model on the validation set of the OK-VQA dataset, run the following command:
```
python evaluation/okvqa_eval.py --okvqa-dir ${OKVQA_DIR} --split val --preds ./predict_okvqa_val.json
```
## GQA
To evaluate the model on teh validation set of the GQA dataset, run the following command:
```
python evaluation/gqa_eval.py --tier val 
```