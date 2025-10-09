##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys

# --- KHẮC PHỤC TRIỆT ĐỂ LỖI PATH TRÊN KAGGLE/NOTEBOOK ---

# 1. Xác định đường dẫn thư mục hiện tại của script (OVRP/POMO)
# Lệnh này hoạt động dù bạn chạy bằng 'python' hay trong Notebook cell.
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd() 

# 2. Thêm THƯ MỤC HIỆN TẠI vào sys.path
# Mục đích: Cho phép các file cùng cấp (như VRPEnv.py và VRProblemDef.py) nhìn thấy nhau.
if current_dir not in sys.path:
    sys.path.insert(0, current_dir) # Ưu tiên tìm kiếm các file cùng cấp

# 3. Tính toán và thêm PROJECT ROOT (3 cấp trên)
# Mục đích: Cho phép tìm kiếm các package top-level (như utils)
# Vị trí hiện tại của script là .../Baseline/OVRP/POMO/ (3 cấp sâu)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))

if project_root not in sys.path:
    sys.path.insert(1, project_root)

# ------------------------------------------------------------------------------------------


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from VRPTrainer import VRPTrainer as Trainer


##########################################################################################
# parameters

env_params = {
    'problem_type': 'OVRP',
    'problem_size': 50,
    'pomo_size': 50,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [8001, 8051],
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 100,
    'train_episodes': 10 * 1000,
    'train_batch_size': 64,
    'prev_model_path': None,
    'logging': {
        'model_save_interval': 10,
        'img_save_interval': 10,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_OVRP_50.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
    'model_load': {
        'enable': False,  # enable loading pre-trained model
    }
}

logger_params = {
    'log_file': {
        'desc': 'train_OVRP_n50_with_instNorm',
        'filename': 'run_log'
    }
}


##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    copy_all_src(trainer.result_folder) 

    trainer.run()


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 4
    trainer_params['train_batch_size'] = 2


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    main()
