##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src
from VRPTrainer import VRPTrainer as Trainer


##########################################################################################
# Parameters for Adversarial Multi-Task Learning
# ----------------------------------------------------------------------------------------
# problem_type:
#   - 'unified' → mixed training across CVRP, OVRP, VRPB, VRPTW, VRPL
#   - can also be single-task: 'CVRP', 'OVRP', etc.
# ----------------------------------------------------------------------------------------

env_params = {
    'problem_type': 'unified',   # 🆕 multi-task setting
    'problem_size': 100,
    'pomo_size': 100,
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

    # 🆕 number of tasks for unified training (CVRP, OVRP, VRPB, VRPTW, VRPL)
    'num_tasks': 5,
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6,
    },
    'scheduler': {
        'milestones': [8001, 8051],
        'gamma': 0.1,
    },
    # 🆕 Separate optimizer for discriminator (optional)
    'discriminator_optimizer': {
        'lr': 1e-4
    },
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 10000,
    'train_episodes': 10 * 1000,
    'train_batch_size': 64,
    'prev_model_path': None,

    'logging': {
        'model_save_interval': 100,
        'img_save_interval': 100,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_unified_100.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },

    'model_load': {
        'enable': False,  # No pre-trained model
    },

    # 🆕 Adversarial learning configuration
    'lambda_adv': 0.3,   # weight for adversarial term
    'disc_steps': 1,     # number of discriminator updates per batch
}

logger_params = {
    'log_file': {
        'desc': 'train_unified_n100_advMTL',
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

    trainer = Trainer(
        env_params=env_params,
        model_params=model_params,
        optimizer_params=optimizer_params,
        trainer_params=trainer_params
    )

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
