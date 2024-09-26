import argparse
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def add_optim_options(self):
        self.parser.add_argument('--warmup_steps', type=int, default=1000)
        self.parser.add_argument('--total_steps', type=int, default=10000)
        self.parser.add_argument('--scheduler_steps', type=int, default=None,
                        help='total number of step for the scheduler, if None then scheduler_total_step = total_step')
        self.parser.add_argument('--accumulation_steps', type=int, default=1)
        self.parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
        self.parser.add_argument('--lr', type=float, default=0.000075, help='learning rate')
        self.parser.add_argument('--clip', type=float, default=1., help='gradient clipping')
        self.parser.add_argument('--optim', type=str, default='adamw')
        self.parser.add_argument('--scheduler', type=str, default='linear')
        self.parser.add_argument('--weight_decay', type=float, default=0.1)
        self.parser.add_argument('--fixed_lr', action='store_true')
        

    def add_eval_options(self):
        self.parser.add_argument('--write_results', action='store_true', help='save results')

    def add_reader_options(self):
        self.parser.add_argument('--data', type=str, default='aokvqa',
                                 help='data name ')
        self.parser.add_argument('--split', type=str, default="val", choices=["train", "test", "val", "testdev"], help="Split, e.g. train, test, val, testdev")
        self.parser.add_argument('--image_path', type=str, default='/data2/KJE/VQA/COCO/images',
                                 help='path of image path')
        self.parser.add_argument('--data_path', type=str, default='/data2/KJE/ModelLogs/revive/processed_data/test.pkl',
                                 help='path of data path')
        self.parser.add_argument('--object_path', type=str, default='/data2/KJE/ModelLogs/Logic_LLama/data/preprocess_okvqa/okvqa_test_object.json',
                                 help='path of object path')
        self.parser.add_argument('--cache_dir', type=str, default='/data3/hg_weight/hg_weight',
                                 help='path of cache dir')

    def add_model_options(self):
        self.parser.add_argument('--vlm_name', type=str, default="Salesforce/blip2-flan-t5-xxl",
                                 help='the name of the vlm')
        self.parser.add_argument('--lvlm_name', type=str, default="llava-hf/llava-v1.6-mistral-7b-hf",
                                 help='the name of the lvlm')
        self.parser.add_argument('--cot', type=str, default="no",
                                 help='whether to use cot')
        self.parser.add_argument('--llm4logic_name', type=str,
                                 default='mistralai/Mistral-7B-Instruct-v0.2',
                                 help='the name of the logic llm')
        self.parser.add_argument('--llm_name', type=str,
                                 default= "mistralai/Mistral-7B-Instruct-v0.2",
                                 help='the name of the llm')
        self.parser.add_argument('--inference_model', type=str,
                                 default= "vlm",
                                 help='a model for the inference')
        self.parser.add_argument('--conclusion_lm', type=str,
                                 default="vlm",
                                 help='a model for the conclusion')
        self.parser.add_argument('--scoring_LLM', type=bool,
                                 default=False,
                                 help='whether to use the scoring LLM')
        self.parser.add_argument('--peft_model_id', type=str, default="lilly9928/LogicLLM",
                                 help='the name of the peft model')
        self.parser.add_argument('--detection_config', type=str, default='../configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml',
                                 help='path of detection config')
        self.parser.add_argument('--detection_weight' , type=str, default='/data2/KJE/weights/model_final_68b088.pkl',
                                 help='path of detection weight')
        self.parser.add_argument('--prompt', type=str, default='no', help='Prompt type to use: no, cot, long_cot')

    def initialize_parser(self):
        # basic parameters
        self.parser.add_argument('--name', type=str, default='ddp_GQA', help='the name of the experiment')
        # self.parser.add_argument('--checkpoint_dir', type=str, default="/data2/KJE/ModelLogs/revive/checkpoint/", help='models are saved here')
        # self.parser.add_argument('--model_path', type=str, default='none', help='path for retraining')
        self.parser.add_argument('--hg_token', type=str, help='huggingface token key')
        self.parser.add_argument('--openai_token', type=str, help='openai token key')
        self.parser.add_argument('--logger_path', type=str, default="/data3/KJE/modelLog/Logic_LLama",
                                 help='logger path')
        self.parser.add_argument('--start_index', help='start index')
        self.parser.add_argument('--object_num', default=2, type=int,help='the number of objects')
        self.parser.add_argument('--beam_num_vlm', default=2, type=int,help='the number of beams of vlm')
        self.parser.add_argument('--threshold', default=0.5, type=float,help='threshold')
        self.parser.add_argument('--num_object_premises', default=2, type=int,help='the number of object premises')
        self.parser.add_argument('--num_llm_premises', default=2, type=int,help='the number of llm premises')
        # dataset parameters
        self.parser.add_argument("--per_gpu_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for training.")
        self.parser.add_argument('--maxload', type=int, default=-1)

        self.parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
        self.parser.add_argument("--main_port", type=int, default=-1,
                        help="Main port (for multi-node SLURM jobs)")
        self.parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")
        # training parameters
        self.parser.add_argument('--device', type=str, choices=["cuda", "cpu"], default="cuda", help='which device to train on')
        self.parser.add_argument('--print_freq', type=int, default=100,
                        help='print loss every <print_freq> steps during training')
        self.parser.add_argument('--eval_freq', type=int, default=1000,
                        help='evaluate model every <eval_freq> steps during training')
        self.parser.add_argument('--save_freq', type=int, default=2500,
                        help='save model every <save_freq> steps during training')
        self.parser.add_argument('--eval_print_freq', type=int, default=1000,
                        help='print intermdiate results of evaluation every <eval_print_freq> steps')

        self.parser.add_argument('--debug', type=bool, default=False,
                                 help='print debug')


    def print_options(self, opt):
        message = '\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default_value = self.parser.get_default(k)
            if v != default_value:
                comment = f'\t(default: {default_value})'
            message += f'{str(k):>30}: {str(v):<40}{comment}\n'

        expr_dir = Path(opt.logger_path)/ opt.name
        model_dir = expr_dir / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(expr_dir/'opt.log', 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

        logger.info(message)

    def parse(self):
        opt = self.parser.parse_args()
        return opt

    def debug_parse(self):
        opt = self.parser.parse_args(args=[])

        return opt


def get_options(use_reader=False,
                use_retriever=False,
                use_optim=False,
                use_eval=False):
    options = Options()
    if use_reader:
        options.add_reader_options()
    if use_optim:
        options.add_optim_options()
    if use_eval:
        options.add_eval_options()
    return options.parse()
