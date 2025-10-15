import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger

from VRPEnv import VRPEnv as Env
from VRPModel import VRPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import * # expects AverageMeter, TimeEstimator, util_* helpers

class VRPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        
        # --- basic control ---
        self.disc_steps = int(self.trainer_params.get('disc_steps', 1))
        
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda handling (left as in your file; default off)
        USE_CUDA = self.trainer_params.get('use_cuda', 0)
        USE_CUDA = 0  # keep CPU by default unless you change
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

        self.optimizer_main = Optimizer(
            list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()),
            **self.optimizer_params['optimizer']
        )

        # discriminator optimizer: use provided lr or default to 0.1 * main lr
        disc_lr = float(self.optimizer_params.get('discriminator_optimizer', {}).get(
            'lr', self.optimizer_params['optimizer']['lr'] * 0.1))
        # allow other kwargs except lr
        disc_opt_kwargs = {k: v for k, v in self.optimizer_params.get('discriminator_optimizer', {}).items() if k != 'lr'}
        self.optimizer_disc = Optimizer(
            self.model.discriminator.parameters(),
            lr=disc_lr,
            **disc_opt_kwargs
        )
        
        self.optimizer = self.optimizer_main
        self.scheduler = Scheduler(self.optimizer_main, **self.optimizer_params['scheduler'])

        # --- adversarial base config ---
        self.fixed_lambda = float(self.trainer_params.get('lambda_adv', 1.5))  # lambda_base
        self.current_step = 0

        # compute total training steps
        train_episodes = int(self.trainer_params.get('train_episodes', 1))
        train_batch_size = int(self.trainer_params.get('train_batch_size', 1))
        epochs = int(self.trainer_params.get('epochs', 1))
        steps_per_epoch = math.ceil(train_episodes / max(1, train_batch_size))
        self.total_train_steps = max(1, epochs * steps_per_epoch)

        # --- Adaptive Control Parameters & smoothing/noise defaults ---
        self.alpha_lambda = float(self.trainer_params.get('alpha_lambda', 1.5))
        self.disc_steps_min = int(self.trainer_params.get('disc_steps_min', 1))
        self.disc_steps_max = int(self.trainer_params.get('disc_steps_max', 5))

        # EMA smoothing for disc acc (used by adaptive decisions)
        # Giữ 0.95 để duy trì sự ổn định.
        self.running_momentum = float(self.trainer_params.get('running_momentum', 0.95)) 
        self.running_disc_acc = float(self.trainer_params.get('running_disc_acc_init', 0.5))
        
        # label smoothing and input noise for discriminator
        self.label_smoothing = float(self.trainer_params.get('label_smoothing', 0.1))  # 0.0 means off
        self.enc_noise_std = float(self.trainer_params.get('enc_noise_std', 1e-3))    # gaussian std

        # lambda schedule params
        self.lambda_k = float(self.trainer_params.get('lambda_k', 10.0))
        # Updated ramp fraction for slower Lambda growth (Stability)
        self.lambda_ramp_fraction = float(self.trainer_params.get('lambda_ramp_fraction', 0.3)) 
        self.lambda_smax = float(self.trainer_params.get('lambda_smax', 1.5))

        # safety / gradient clipping
        self.max_grad_norm = float(self.trainer_params.get('max_grad_norm', 1.0))

        # Restore (if enabled)
        self.start_epoch = 1
        model_load = trainer_params.get('model_load', {'enable': False})
        if model_load.get('enable', False):
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict']) # Load scheduler state
            self.optimizer_disc.load_state_dict(checkpoint['optimizer_disc_state_dict']) # Load Disc optimizer state
            self.current_step = checkpoint.get('current_step', 0)
            
            # FIX: Load running_disc_acc from checkpoint 
            if 'running_disc_acc' in checkpoint:
                self.running_disc_acc = checkpoint['running_disc_acc']
                self.logger.info(f"Loaded running_disc_acc: {self.running_disc_acc:.3f}")
            
            self.logger.info('Saved Model Loaded !!')

        self.time_estimator = TimeEstimator()

    # ---------------- helper ----------------
    def set_disc_lr(self, new_lr: float):
        """Set discriminator lr on the fly."""
        for g in self.optimizer_disc.param_groups:
            g['lr'] = float(new_lr)

    # ---------------------------------------------------------------------
    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs'] + 1):
            self.logger.info('=================================================================')

            train_score, train_loss, avg_loss_vrp, avg_loss_adv, avg_loss_disc, avg_disc_acc = self._train_one_epoch(epoch)

            # lr scheduler step for main optimizer
            self.scheduler.step()
            
            # Logging (record metrics)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)
            self.result_log.append('loss_vrp', epoch, avg_loss_vrp)
            self.result_log.append('loss_adv', epoch, avg_loss_adv)
            self.result_log.append('loss_disc', epoch, avg_loss_disc)
            self.result_log.append('disc_acc', epoch, avg_disc_acc)

            # Adaptive adjustment (use EMA smoothed acc inside)
            self._adjust_adversarial_balance(avg_disc_acc)

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info(f"Epoch {epoch:3d}/{self.trainer_params['epochs']:3d}: Time Est.: Elapsed[{elapsed_time_str}], Remain[{remain_time_str}]")

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            if epoch > 1:
                image_prefix = f'{self.result_folder}/latest'
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                                     self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                                     self.result_log, labels=['train_loss', 'loss_vrp', 'loss_adv', 'loss_disc', 'disc_acc'])

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer_main.state_dict(),
                    'optimizer_disc_state_dict': self.optimizer_disc.state_dict(), # Saving Disc optimizer state
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data(),
                    'current_step': self.current_step, 
                    'running_disc_acc': self.running_disc_acc, # Saving running_disc_acc 
                }
                torch.save(checkpoint_dict, f'{self.result_folder}/checkpoint-{epoch}.pt')

            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = f'{self.result_folder}/img/checkpoint-{epoch}'
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                                     self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                                     self.result_log, labels=['train_loss', 'loss_vrp', 'loss_adv', 'loss_disc', 'disc_acc'])

            if all_done:
                self.logger.info(" * Training Done * ")
                util_print_log_array(self.logger, self.result_log)

    # ---------------------------------------------------------------------
    def _train_one_epoch(self, epoch):
        score_AM, loss_AM, loss_vrp_AM, loss_adv_AM, loss_disc_AM, disc_acc_AM = [AverageMeter() for _ in range(6)]

        train_num_episode = int(self.trainer_params['train_episodes'])
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(int(self.trainer_params['train_batch_size']), remaining)

            (avg_score, avg_loss_total, avg_loss_vrp,
             avg_loss_adv, avg_loss_disc, avg_disc_acc) = self._train_one_batch(batch_size) 
            
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss_total, batch_size)
            loss_vrp_AM.update(avg_loss_vrp, batch_size)
            loss_adv_AM.update(avg_loss_adv, batch_size)
            loss_disc_AM.update(avg_loss_disc, batch_size)
            disc_acc_AM.update(avg_disc_acc, batch_size)

            episode += batch_size

            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info(
                        f"Epoch {epoch:3d}: Train {episode:3d}/{train_num_episode:3d}({100. * episode / train_num_episode:1.1f}%)   "
                        f"Score: {score_AM.avg:.4f}, Total: {loss_AM.avg:.4f} (VRP: {loss_vrp_AM.avg:.4f}, Adv: {loss_adv_AM.avg:.4f}, Disc: {loss_disc_AM.avg:.4f}, Acc: {disc_acc_AM.avg:.3f})"
                    )

        self.logger.info(
            f"Epoch {epoch:3d}: Train (100%)   Score: {score_AM.avg:.4f}, Total: {loss_AM.avg:.4f} "
            f"(VRP: {loss_vrp_AM.avg:.4f}, Adv: {loss_adv_AM.avg:.4f}, Disc: {loss_disc_AM.avg:.4f}, Acc: {disc_acc_AM.avg:.3f})"
        )

        return score_AM.avg, loss_AM.avg, loss_vrp_AM.avg, loss_adv_AM.avg, loss_disc_AM.avg, disc_acc_AM.avg

    # ---------------------------------------------------------------------
    def _train_one_batch(self, batch_size):
        self.model.train()
        self.env.load_problems(batch_size)
        reset_state, task_labels, _ = self.env.reset()
        self.model.pre_forward(reset_state)
        
        # -------------------------------
        # lambda schedule (sigmoid with ramp fraction)
        # -------------------------------
        p = float(self.current_step) / float(max(1, self.total_train_steps))
        p_adj = min(1.0, p / max(1e-9, self.lambda_ramp_fraction))
        schedule_val = (2.0 / (1.0 + math.exp(-self.lambda_k * p_adj))) - 1.0
        lambda_scheduled = float(self.fixed_lambda) * schedule_val  # in [0, lambda_base]

        # -------------------------------
        # Train discriminator (with small noise + label smoothing)
        # -------------------------------
        acc_loss_disc, acc_disc_correct, acc_disc_total = 0.0, 0, 0
        num_classes = None

        for _ in range(self.disc_steps):
            self.optimizer_disc.zero_grad()
            self.model.discriminator.train()
            
            # detach encoding and (optionally) add gaussian noise
            with torch.no_grad():
                enc_detached = self.model.encoded_nodes.clone().detach()
            if self.enc_noise_std and self.enc_noise_std > 0.0:
                enc_input = enc_detached + (self.enc_noise_std * torch.randn_like(enc_detached))
            else:
                enc_input = enc_detached

            logits_disc = self.model.discriminator(enc_input, reverse=False)
            if num_classes is None:
                num_classes = logits_disc.size(1)

            # label smoothing -> soft targets
            ls = float(self.label_smoothing)
            if ls > 0.0:
                with torch.no_grad():
                    smooth_targets = torch.full((task_labels.size(0), num_classes),
                                                 ls / (num_classes - 1),
                                                 device=task_labels.device, dtype=logits_disc.dtype)
                    smooth_targets.scatter_(1, task_labels.unsqueeze(1), 1.0 - ls)
                loss_disc = -(smooth_targets * F.log_softmax(logits_disc, dim=1)).sum(dim=1).mean()
            else:
                loss_disc = F.cross_entropy(logits_disc, task_labels)

            acc_loss_disc += loss_disc.item()
            with torch.no_grad():
                preds = logits_disc.argmax(dim=1)
                acc_disc_correct += (preds == task_labels).sum().item()
                acc_disc_total += task_labels.size(0)

            loss_disc.backward()
            self.optimizer_disc.step()
        
        avg_loss_disc = acc_loss_disc / max(1, self.disc_steps)
        avg_disc_acc = float(acc_disc_correct) / float(max(1, acc_disc_total))

        # -------------------------------
        # update running (EMA) disc acc for adaptive decisions
        # -------------------------------
        initial_running_acc = float(self.trainer_params.get('running_disc_acc_init', 0.5))
        self.running_disc_acc = (self.running_momentum * getattr(self, 'running_disc_acc', initial_running_acc) +
                                 (1.0 - self.running_momentum) * avg_disc_acc)
        use_disc_acc_for_adaptive = float(self.running_disc_acc)

        # -------------------------------
        # Adaptive lambda scaling (based on running acc)
        # -------------------------------
        scale_factor = 1.0 + self.alpha_lambda * (0.5 - use_disc_acc_for_adaptive)
        scale_clamped = max(0.0, min(self.lambda_smax, scale_factor))
        lambda_p = lambda_scheduled * scale_clamped
        # final clamp to not exceed base and not negative
        lambda_p = float(max(0.0, min(lambda_p, float(self.fixed_lambda))))

        # -------------------------------
        # VRP rollout (policy) - collect trajectory
        # -------------------------------
        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        state, reward, done = self.env.pre_step()
        
        while not done:
            selected, prob = self.model(state)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        log_prob = prob_list.log().sum(dim=2)
        loss_vrp = (-advantage * log_prob).mean()
        loss_vrp_item = loss_vrp.item()

        # -------------------------------
        # Encoder+Decoder update (adversarial via GRL)
        # -------------------------------
        self.optimizer_main.zero_grad()
        logits_adv = self.model.discriminator(self.model.encoded_nodes, reverse=True)
        loss_adv = F.cross_entropy(logits_adv, task_labels)
        loss_adv_item = loss_adv.item()

        loss_total = loss_vrp + lambda_p * loss_adv
        loss_total.backward()

        # gradient clipping for encoder+decoder
        nn.utils.clip_grad_norm_(
            list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()),
            self.max_grad_norm
        )
        self.optimizer_main.step()

        # advance step counter
        self.current_step += 1

        max_pomo_reward, _ = reward.max(dim=1)
        score_mean = -max_pomo_reward.float().mean()

        # return metrics: score, total loss, vrp loss, adv loss, disc loss, disc acc (raw epoch avg)
        return score_mean.item(), loss_total.item(), loss_vrp_item, loss_adv_item, avg_loss_disc, avg_disc_acc

    # ---------------------------------------------------------------------
    def _adjust_adversarial_balance(self, disc_acc):
        """
        Adaptive adjustment of disc_steps and discriminator learning rate.
        FIXED: TIGHTENED thresholds significantly to force Disc Acc down from the 90%+ spikes.
        Target EMA Acc region: 25% - 50%.
        """
        prev_lr = float(self.optimizer_disc.param_groups[0]['lr'])
        use_acc = float(getattr(self, 'running_disc_acc', disc_acc))
        
        # Dampening factor (kept at 1.2x)
        LR_CHANGE_FACTOR = 1.2 

        # adaptive disc_steps
        # Tăng steps nếu Acc > 50% (Disc quá mạnh, cần học chậm lại)
        if use_acc > 0.50: 
            self.disc_steps = min(self.disc_steps + 1, self.disc_steps_max)
        # Giảm steps nếu Acc < 0.25 (Disc quá yếu, cần học nhanh hơn)
        elif use_acc < 0.25:
            self.disc_steps = max(self.disc_steps - 1, self.disc_steps_min)

        # adaptive lr
        new_lr = prev_lr
        
        # Reduce LR if Disc is too strong (Acc > 0.50) --> TIGHTENED THRESHOLD!
        if use_acc > 0.50:
            new_lr = max(prev_lr / LR_CHANGE_FACTOR, 1e-8)
        # Increase LR if Disc is failing to learn (Acc < 0.25) --> TIGHTENED THRESHOLD!
        elif use_acc < 0.25:
            new_lr = min(prev_lr * LR_CHANGE_FACTOR, 1e-3)

        for param_group in self.optimizer_disc.param_groups:
            param_group['lr'] = new_lr

        self.logger.debug(f"[Adaptive] Disc Steps: {self.disc_steps}, Disc LR: {new_lr:.2e}, EMA Acc: {use_acc:.3f}")