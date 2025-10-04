
import torch
from logging import getLogger

from VRPEnv import VRPEnv as Env
from VRP_adversarial_model import VRPModel_AMTL as Model
import torch.nn.functional as F
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from VRP_adversarial_model import TaskDiscriminator
from utils.utils import *


class VRPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        USE_CUDA = 0
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        num_tasks = self.model_params.get('num_tasks', None)
        if num_tasks is None:
            raise ValueError("model_params must include 'num_tasks' for adversarial training.")

        emb_dim = self.model_params['embedding_dim']
        # instantiate discriminator (device will be consistent since same default tensor type)
        self.discriminator = TaskDiscriminator(emb_dim=emb_dim, num_tasks=num_tasks)
        # optimizer for discriminator
        disc_opt_cfg = self.optimizer_params.get('discriminator_optimizer', {'lr': 1e-3})
        self.disc_optimizer = Optimizer(self.discriminator.parameters(), **disc_opt_cfg)

        # adversarial weight
        self.lambda_adv = self.trainer_params.get('lambda_adv', 0.5)
        # how many discriminator steps per batch (usually 1)
        self.disc_steps = self.trainer_params.get('disc_steps', 1)
        
        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            # existing checkpoint logic...
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            # optionally load discriminator state if present in checkpoint
            if 'discriminator_state_dict' in checkpoint:
                self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # if discriminator optimizer state stored
            if 'disc_optimizer_state_dict' in checkpoint:
                self.disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            # LR Decay
            self.scheduler.step()

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            # Save latest images, every epoch
            if epoch > 1:
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            # Save Model
            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data(),
                    # new:
                    'discriminator_state_dict': self.discriminator.state_dict(),
                    'disc_optimizer_state_dict': self.disc_optimizer.state_dict(),
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            # Save Image
            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            # All-done announcement
            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, avg_loss = self._train_one_batch(batch_size)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size):

        # Prep
        ###############################################
        self.model.train()
        self.discriminator.train()
        self.env.load_problems(batch_size)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        # --- obtain task labels for this batch --------------------------------
        # Preferably reset_state contains task ids (tensor of shape [batch])
        if hasattr(reset_state, 'task_id'):
            # assume shape [batch]
            task_ids = reset_state.task_id.to(torch.long)
        elif hasattr(self.env, 'current_task_ids'):
            task_ids = torch.tensor(self.env.current_task_ids, dtype=torch.long)
        else:
            # fallback: assume single global task (all zeros)
            # **Better:** make sure your Env returns task ids for multi-task training.
            task_ids = torch.zeros(batch_size, dtype=torch.long)
        # Move to same device as model encoded nodes:
        task_ids = task_ids.to(self.model.encoded_nodes.device)
        # ---------------------------------------------------------------------

        # ----------------- Step A: Train Discriminator -----------------------
        # Use encoder outputs detached so encoder is not updated here.
        encoded_nodes_detached = self.model.encoded_nodes.detach()  # [B, N, E]

        # Optionally perform multiple discriminator steps
        for _ in range(self.disc_steps):
            self.disc_optimizer.zero_grad()
            disc_logits = self.discriminator(encoded_nodes_detached)  # [B, num_tasks]
            loss_disc = F.cross_entropy(disc_logits, task_ids)
            loss_disc.backward()
            self.disc_optimizer.step()
        # ---------------------------------------------------------------------

        # POMO Rollout (same as before)
        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0), device=self.model.encoded_nodes.device)
        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob, disc_logits_for_encoder = self.model(state)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss: REINFORCE-style (same as before)
        advantage = reward - reward.float().mean(dim=1, keepdims=True)   # [B, pomo]
        log_prob = prob_list.log().sum(dim=2)                            # [B, pomo]
        loss_per_sample = -advantage * log_prob                           # [B, pomo]
        loss_mean = loss_per_sample.mean()                                # scalar

        # ----------------- Step B: Encoder+Decoder Adversarial Update -------
        # Compute discriminator logits WITHOUT detach so gradients flow into encoder
        loss_disc_for_encoder = F.cross_entropy(disc_logits_for_encoder, task_ids)

        # Total loss: minimize routing objective AND *fool* discriminator
        total_loss = loss_mean - (self.lambda_adv * loss_disc_for_encoder)

        # Backprop and step main optimizer (encoder+decoder)
        self.model.zero_grad()
        # note: if discriminator shares params with model (it shouldn't), you may need to zero it too
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        # ---------------------------------------------------------------------

        # Score (same as before)
        max_pomo_reward, _ = reward.max(dim=1)  # [B]
        score_mean = -max_pomo_reward.float().mean()

        return score_mean.item(), loss_mean.item()

