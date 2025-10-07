##########################################################################################
# Machine Environment Config
##########################################################################################

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config
##########################################################################################

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")   # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# Imports
##########################################################################################

import logging
from utils.utils import create_logger, copy_all_src
from VRPTrainer import VRPTrainer as Trainer


##########################################################################################
# Parameters
##########################################################################################

# Environment configuration
env_params = {
    'problem_type': 'unified',      # mix of CVRP, OVRP, VRPB, VRPTW, VRPL
    'problem_size': 50,
    'pomo_size': 50,
}

# Model configuration
model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128 ** 0.5,
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',

    # 🧩 Added for adversarial multi-task learning
    'num_tasks': 5,  # CVRP, OVRP, VRPB, VRPTW, VRPL
}

# Optimizer configurations
optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [8001, 8051],
        'gamma': 0.1
    },
    # 🧩 Additional optimizer for adversarial discriminator
    'discriminator_optimizer': {
        'lr': 1e-4
    }
}

# Trainer configuration
trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 100,
    'train_episodes': 10 * 1000,
    'train_batch_size': 32,
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
        'enable': False,
    },

    # 🧩 Adversarial training parameters
    'lambda_adv': 0.3,   # weight for adversarial loss term
    'disc_steps': 1,     # how many discriminator updates per batch
}

# Logging configuration
logger_params = {
    'log_file': {
        'desc': f"train_{env_params['problem_type']}_n{env_params['problem_size']}_adv",
        'filename': 'run_log'
    }
}


##########################################################################################
# Main Entry Point
##########################################################################################

def main():
    """Main training entry point for adversarial multi-task VRP solver."""
    if DEBUG_MODE:
        _set_debug_mode()

    # Setup logger
    create_logger(**logger_params)
    _print_config()

    # Initialize trainer
    trainer = Trainer(
        env_params=env_params,
        model_params=model_params,
        optimizer_params=optimizer_params,
        trainer_params=trainer_params
    )

    # Copy source code for reproducibility
    copy_all_src(trainer.result_folder)

    # Start training
    trainer.run()


##########################################################################################
# Helper Functions
##########################################################################################

def _set_debug_mode():
    """Reduce runtime for quick debugging."""
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 4
    trainer_params['train_batch_size'] = 2


def _print_config():
    """Log all configurations for reproducibility."""
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))

    for g_key in globals().keys():
        if g_key.endswith('params'):
            logger.info(f"{g_key}: {globals()[g_key]}")


##########################################################################################
# Run
##########################################################################################

if __name__ == "__main__":
    main()
