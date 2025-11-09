import torch
import torch.nn.functional as F
import random
import os
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from torch.utils.data import DataLoader, Dataset

from utils import *

class Trainer:
    """
    3-PHASE SEQUENTIAL TRAINER
    
    Phase 1 (Epochs 1-40): Train RL Decoder
        - Data: RANDOM (on-the-fly)
        - Train: Encoder + RL Decoder
        - Freeze: Refiner (disabled)
        - Loss: RL loss only
        - Goal: Match baseline performance (fair comparison)
    
    Phase 2 (Epochs 41-70): Pre-train Refiner (Supervised)
        - Data: STATIC (with LKH3 optimal)
        - Train: Encoder (fine-tune) + Refiner (learn)
        - Freeze: RL Decoder
        - Loss: Supervised Refiner loss
        - Goal: Learn to refine toward optimal
    
    Phase 3 (Epochs 71-100): Fine-tune Refiner (RL)
        - Data: RANDOM (on-the-fly)
        - Train: Encoder (fine-tune) + Refiner (fine-tune)
        - Freeze: RL Decoder
        - Loss: RL Refiner loss
        - Goal: Generalize refinement to unseen problems
    """
    
    def __init__(self, args, env_params, model_params, optimizer_params, trainer_params):
        self.args = args
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # Loss weights
        self.lambda_recon = trainer_params.get('lambda_recon', 0.0)
        self.rl_weight = trainer_params.get('rl_weight', 1.0)
        self.refiner_weight = trainer_params.get('refiner_weight', 0.1)

        self.device = args.device
        self.log_path = args.log_path
        self.result_log = {"val_score": [], "val_gap": []}
        self.current_phase = 1

        # Main Components
        self.envs = get_env(self.args.problem)
        self.model = get_model(self.args.model_type)(**self.model_params).to(self.device)
        
        # Create 3 optimizers for 3 phases
        self.optimizer_phase1, self.optimizer_phase2, self.optimizer_phase3 = self.create_optimizers(
            self.model,
            self.optimizer_params['optimizer']['lr'],
            self.optimizer_params['optimizer'].get('lr_phase2', 1e-4),
            self.optimizer_params['optimizer'].get('lr_phase3', 1e-5)
        )
        
        self.scheduler_phase1 = Scheduler(self.optimizer_phase1, **self.optimizer_params['scheduler'])
        self.scheduler_phase2 = Scheduler(self.optimizer_phase2, **self.optimizer_params['scheduler'])
        self.scheduler_phase3 = Scheduler(self.optimizer_phase3, **self.optimizer_params['scheduler'])

        num_param(self.model)

        # Static DataLoader (for Phase 2 only)
        self.static_train_loader = None
        if hasattr(args, 'train_dataset_dir') and args.train_dataset_dir:
            self.static_train_loader = self._create_static_loader(
                self.args.train_dataset_dir,
                batch_size=self.trainer_params['train_batch_size']
            )

        # Restore Checkpoint
        self.start_epoch = 1
        if args.checkpoint is not None:
            self._load_checkpoint(args.checkpoint)

        # Utility
        self.time_estimator = TimeEstimator()

    @staticmethod
    def create_optimizers(model, lr_phase1=1e-4, lr_phase2=1e-4, lr_phase3=1e-5):
        """
        Create 3 optimizers for 3 phases
        
        Phase 1: Encoder + RL Decoder
        Phase 2: Encoder + Refiner
        Phase 3: Encoder + Refiner (smaller LR)
        """
        print(f">> Creating Optimizers:")
        print(f"   Phase 1 (LR={lr_phase1}): Encoder + RL Decoder")
        print(f"   Phase 2 (LR={lr_phase2}): Encoder + Refiner")
        print(f"   Phase 3 (LR={lr_phase3}): Encoder + Refiner")
        
        # Phase 1: Encoder + RL Decoder
        encoder_decoder_params = list(model.encoder.parameters()) + list(model.decoder_rl.parameters())
        optimizer_phase1 = torch.optim.Adam(encoder_decoder_params, lr=lr_phase1)
        
        # Phase 2: Encoder + Refiner
        if model.use_refiner:
            encoder_refiner_params = list(model.encoder.parameters()) + list(model.diffusion_refiner.parameters())
            optimizer_phase2 = torch.optim.Adam(encoder_refiner_params, lr=lr_phase2)
            
            # Phase 3: Same params as Phase 2, but smaller LR
            optimizer_phase3 = torch.optim.Adam(encoder_refiner_params, lr=lr_phase3)
        else:
            optimizer_phase2 = None
            optimizer_phase3 = None
        
        return optimizer_phase1, optimizer_phase2, optimizer_phase3

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint and restore training state"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.start_epoch = 1 + checkpoint['epoch']
        self.current_phase = checkpoint.get('current_phase', 1)
        
        print(f">> Checkpoint Loaded: Epoch {checkpoint['epoch']}, Phase {self.current_phase}")
        
        # Load appropriate optimizer
        if self.current_phase == 1:
            self.optimizer_phase1.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler_phase1.last_epoch = checkpoint['epoch'] - 1
        elif self.current_phase == 2:
            if 'optimizer_phase2_state_dict' in checkpoint:
                self.optimizer_phase2.load_state_dict(checkpoint['optimizer_phase2_state_dict'])
                self.scheduler_phase2.last_epoch = checkpoint['epoch'] - 1
            self._switch_to_phase2()
        elif self.current_phase == 3:
            if 'optimizer_phase3_state_dict' in checkpoint:
                self.optimizer_phase3.load_state_dict(checkpoint['optimizer_phase3_state_dict'])
                self.scheduler_phase3.last_epoch = checkpoint['epoch'] - 1
            self._switch_to_phase3()

    def _switch_to_phase2(self):
        """Switch from Phase 1 to Phase 2"""
        print("="*80)
        print("🔄 SWITCHING TO PHASE 2: SUPERVISED REFINER PRE-TRAINING")
        print("🔒 Freezing: RL Decoder")
        print("🔓 Training: Encoder + Refiner")
        print("📊 Data: Static (with LKH3 optimal)")
        print("="*80)
        
        self.model.freeze_rl_decoder()
        self.model.unfreeze_encoder_refiner()
        self.current_phase = 2

    def _switch_to_phase3(self):
        """Switch from Phase 2 to Phase 3"""
        print("="*80)
        print("🔄 SWITCHING TO PHASE 3: RL REFINER FINE-TUNING")
        print("🔒 Freezing: RL Decoder")
        print("🔓 Training: Encoder + Refiner")
        print("📊 Data: Random (generalization)")
        print("="*80)
        
        # Keep RL Decoder frozen, continue training Encoder + Refiner
        self.current_phase = 3

    def run(self):
        """Main training loop with 3 sequential phases"""
        self.time_estimator.reset(self.start_epoch)
        
        # Phase transition points
        phase1_epochs = self.trainer_params.get('phase1_epochs', 40)
        phase2_epochs = self.trainer_params.get('phase2_epochs', 70)
        total_epochs = self.trainer_params['epochs']

        print("="*80)
        print(f"📊 3-PHASE TRAINING PLAN:")
        print(f"   Phase 1 (RL Decoder):     Epochs 1-{phase1_epochs}")
        print(f"   Phase 2 (SL Refiner):     Epochs {phase1_epochs+1}-{phase2_epochs}")
        print(f"   Phase 3 (RL Refiner):     Epochs {phase2_epochs+1}-{total_epochs}")
        print("="*80)

        for epoch in range(self.start_epoch, total_epochs + 1):
            print('='*80)

            # Check phase transitions
            if self.current_phase == 1 and epoch > phase1_epochs:
                self._switch_to_phase2()
                self._save_checkpoint(epoch - 1, is_phase_switch=True, phase_name='phase1_final')
            
            elif self.current_phase == 2 and epoch > phase2_epochs:
                self._switch_to_phase3()
                self._save_checkpoint(epoch - 1, is_phase_switch=True, phase_name='phase2_final')
            
            # Training
            if self.current_phase == 1:
                print(f"🔵 Phase 1: RL Decoder Training (Epoch {epoch}/{phase1_epochs})")
                train_score, train_loss = self._train_one_epoch_phase1(epoch)
                self.scheduler_phase1.step()
            
            elif self.current_phase == 2:
                print(f"🟡 Phase 2: Supervised Refiner (Epoch {epoch}/{phase2_epochs})")
                train_score, train_loss = self._train_one_epoch_phase2(epoch)
                self.scheduler_phase2.step()
            
            else:  # Phase 3
                print(f"🟢 Phase 3: RL Refiner Fine-tuning (Epoch {epoch}/{total_epochs})")
                train_score, train_loss = self._train_one_epoch_phase3(epoch)
                self.scheduler_phase3.step()

            # Time estimation
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, total_epochs)
            print("⏱️  Time Est.: Elapsed[{}], Remain[{}]".format(elapsed_time_str, remain_time_str))

            # Validation
            if epoch % self.trainer_params['validation_interval'] == 0:
                self._validate(epoch)

            # Save Checkpoint
            all_done = (epoch == total_epochs)
            model_save_interval = self.trainer_params['model_save_interval']
            
            if all_done or (epoch % model_save_interval == 0):
                self._save_checkpoint(epoch)
                
        print("="*80)
        print("✅ 3-PHASE TRAINING COMPLETED!")
        print("="*80)

    def _save_checkpoint(self, epoch, is_phase_switch=False, phase_name=None):
        """Save checkpoint"""
        print(f"💾 Saving checkpoint (Epoch {epoch}, Phase {self.current_phase})...")
        
        checkpoint_dict = {
            'epoch': epoch,
            'problem': self.args.problem,
            'current_phase': self.current_phase,
            'model_state_dict': self.model.state_dict(),
            'result_log': self.result_log
        }
        
        # Save appropriate optimizer
        if self.current_phase == 1:
            checkpoint_dict['optimizer_state_dict'] = self.optimizer_phase1.state_dict()
            checkpoint_dict['scheduler_state_dict'] = self.scheduler_phase1.state_dict()
        elif self.current_phase == 2:
            checkpoint_dict['optimizer_phase2_state_dict'] = self.optimizer_phase2.state_dict()
            checkpoint_dict['scheduler_phase2_state_dict'] = self.scheduler_phase2.state_dict()
        else:
            checkpoint_dict['optimizer_phase3_state_dict'] = self.optimizer_phase3.state_dict()
            checkpoint_dict['scheduler_phase3_state_dict'] = self.scheduler_phase3.state_dict()
        
        if is_phase_switch and phase_name:
            filename = f'{phase_name}_epoch-{epoch}.pt'
        else:
            filename = f'epoch-{epoch}.pt'
            
        torch.save(checkpoint_dict, f'{self.log_path}/{filename}')
        print(f"✅ Saved: {filename}")

    # =========================================================================
    # STATIC DATA LOADER (Phase 2 only)
    # =========================================================================

    def _create_static_loader(self, data_dir, batch_size):
        """Create DataLoader for static dataset with optimal solutions"""
        print(f"📂 Loading static training dataset from: {data_dir}")
        
        pkl_filename = f"{self.args.problem}_train_with_lkh3.pkl"
        pkl_path = os.path.join(data_dir, self.args.problem, pkl_filename)
        
        if not os.path.exists(pkl_path):
            print(f"⚠️  Warning: Static dataset not found at {pkl_path}")
            print(f"   Phase 2 will skip (no supervised training)")
            return None
        
        try:
            dataset_list = load_dataset(pkl_path, disable_print=False)
            print(f"✅ Loaded {len(dataset_list)} instances with optimal solutions")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
        
        class StaticVRPDataset(Dataset):
            def __init__(self, data_list, device):
                self.data = data_list
                self.device = device
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                problem, optimal_tour = self.data[idx]
                if not isinstance(optimal_tour, torch.Tensor):
                    optimal_tour = torch.tensor(optimal_tour, dtype=torch.long)
                return problem, optimal_tour
        
        dataset = StaticVRPDataset(dataset_list, self.device)
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self._collate_fn
        )
        
        return loader

    def _collate_fn(self, batch):
        """Custom collate function"""
        problems, tours = zip(*batch)
        
        batched_tours = torch.stack(tours).to(self.device)
        
        if isinstance(problems[0], dict):
            batched_problems = {}
            for key in problems[0].keys():
                values = [p[key] for p in problems]
                if isinstance(values[0], torch.Tensor):
                    batched_problems[key] = torch.stack(values).to(self.device)
                else:
                    batched_problems[key] = values
        else:
            batched_problems = tuple(torch.stack([p[i] for p in problems]).to(self.device)
                                    for i in range(len(problems[0])))
        
        return batched_problems, batched_tours

    def _tour_to_heatmap(self, tour, num_nodes):
        """Convert optimal tour to visit heatmap"""
        batch_size = tour.size(0)
        device = tour.device
        
        heatmap = torch.zeros(batch_size, num_nodes, device=device)
        
        for step in range(tour.size(1)):
            node_idx = tour[:, step]
            heatmap.scatter_add_(
                1,
                node_idx.unsqueeze(1),
                torch.ones(batch_size, 1, device=device)
            )
        
        max_visits = heatmap.max(dim=1, keepdim=True)[0] + 1e-8
        heatmap = heatmap / max_visits
        
        return heatmap

    # =========================================================================
    # PHASE 1: TRAIN RL DECODER (RANDOM DATA)
    # =========================================================================

    def _train_one_epoch_phase1(self, epoch):
        """Phase 1: Train Encoder + RL Decoder with random data"""
        episode = 0
        score_AM, loss_AM = AverageMeter(), AverageMeter()
        train_num_episode = self.trainer_params['train_episodes']

        while episode < train_num_episode:
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            # Random data (on-the-fly)
            env = random.sample(self.envs, 1)[0](**self.env_params)
            data = env.get_random_problems(batch_size, self.env_params["problem_size"])
            env.load_problems(batch_size, problems=data, aug_factor=1)
            reset_state, _, _ = env.reset()

            losses = self.train_phase1_step(reset_state)
            
            score_AM.update(losses['avg_cost'], batch_size)
            loss_AM.update(losses['total'], batch_size)
            episode += batch_size
            
            if episode % (train_num_episode // 10) == 0:
                print(f"  Progress: {episode}/{train_num_episode}, Cost: {score_AM.avg:.2f}, Loss: {loss_AM.avg:.4f}")
            
        print(f"📊 Phase 1 Summary: Avg Cost: {score_AM.avg:.4f}, Loss: {loss_AM.avg:.4f}")
        return score_AM.avg, loss_AM.avg

    def train_phase1_step(self, reset_state):
        """Single training step for Phase 1"""
        self.model.train()
        self.optimizer_phase1.zero_grad()
        
        # Encode
        self.model.pre_forward(reset_state)
        
        # RL Loss (REINFORCE)
        state = reset_state.clone()
        trajectories = self.model._forward_rollout(state)
        
        log_probs = torch.log(trajectories['probs'] + 1e-10).sum(dim=2)
        costs = self.model.rollout_manager.compute_solution_cost(state, trajectories)
        
        baseline = costs.mean(dim=1, keepdim=True).detach()
        advantages = costs - baseline
        rl_loss = (advantages * log_probs).mean()

        recon_loss = self.model.compute_slot_reconstruction_loss(reset_state)
        
        total_loss = rl_loss + self.lambda_recon * recon_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.optimizer_phase1.param_groups[0]['params'], max_norm=1.0)
        self.optimizer_phase1.step()
        
        losses = {
            'total': total_loss.item(),
            'rl': rl_loss.item(),
            'avg_cost': costs.mean().item(),
        }
        
        return losses

    # =========================================================================
    # PHASE 2: SUPERVISED REFINER (STATIC DATA)
    # =========================================================================

    def _train_one_epoch_phase2(self, epoch):
        """Phase 2: Train Encoder + Refiner with static data (supervised)"""
        if self.static_train_loader is None:
            print("⚠️  No static dataset available, skipping Phase 2")
            return 0.0, 0.0

        loss_AM, ref_loss_AM = AverageMeter(), AverageMeter()

        for batch_idx, (problem_batch, optimal_tours) in enumerate(self.static_train_loader):
            
            env = random.sample(self.envs, 1)[0](**self.env_params)
            batch_size = optimal_tours.size(0)
            
            env.load_problems(batch_size, problems=problem_batch, aug_factor=1)
            reset_state, _, _ = env.reset()
            
            # Convert optimal tour to heatmap
            num_nodes = self.model_params['problem_size'] + 1
            optimal_heatmap = self._tour_to_heatmap(optimal_tours, num_nodes)
            
            losses = self.train_phase2_step(reset_state, optimal_heatmap)
            
            loss_AM.update(losses['total'], batch_size)
            ref_loss_AM.update(losses['refiner_supervised'], batch_size)
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: Refiner_SL={losses['refiner_supervised']:.4f}")
        
        print(f"📊 Phase 2 Summary: Loss: {loss_AM.avg:.4f}, Refiner_SL: {ref_loss_AM.avg:.4f}")
        return ref_loss_AM.avg, loss_AM.avg

    def train_phase2_step(self, reset_state, optimal_solutions):
        """Single training step for Phase 2"""
        self.model.train()
        self.optimizer_phase2.zero_grad()
        
        # Encode
        self.model.pre_forward(reset_state)
        
        # Supervised Refiner Loss
        refiner_loss = self.model.compute_refinement_loss_supervised(optimal_solutions)
        
        recon_loss = self.model.compute_slot_reconstruction_loss(reset_state)

        total_loss = refiner_loss + self.lambda_recon * recon_loss
        
        if total_loss > 0:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.optimizer_phase2.param_groups[0]['params'], max_norm=1.0)
            self.optimizer_phase2.step()
        
        losses = {
            'total': total_loss.item(),
            'refiner_supervised': refiner_loss.item(),
        }
        
        return losses

    # =========================================================================
    # PHASE 3: RL REFINER (RANDOM DATA)
    # =========================================================================

    def _train_one_epoch_phase3(self, epoch):
        """Phase 3: Fine-tune Encoder + Refiner with random data (RL)"""
        episode = 0
        loss_AM, improve_AM = AverageMeter(), AverageMeter()
        train_num_episode = self.trainer_params['train_episodes']
        
        baseline_ema = None
        ema_alpha = 0.95

        while episode < train_num_episode:
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            # Random data (generalization)
            env = random.sample(self.envs, 1)[0](**self.env_params)
            data = env.get_random_problems(batch_size, self.env_params["problem_size"])
            env.load_problems(batch_size, problems=data, aug_factor=1)
            reset_state, _, _ = env.reset()

            # Initialize baseline
            if baseline_ema is None:
                with torch.no_grad():
                    self.model.pre_forward(reset_state)
                    state_copy = reset_state.clone()
                    traj = self.model._forward_rollout(state_copy)
                    baseline_ema = self.model.rollout_manager.compute_solution_cost(
                        state_copy, traj
                    ).mean().item()
            
            losses = self.train_phase3_step(reset_state, baseline=baseline_ema)
            baseline_ema = ema_alpha * baseline_ema + (1 - ema_alpha) * losses['cost_initial']

            loss_AM.update(losses['total'], batch_size)
            improve_AM.update(losses['improvement'], batch_size)
            episode += batch_size
            
            if episode % (train_num_episode // 10) == 0:
                print(f"  Progress: {episode}/{train_num_episode}, Improve: {improve_AM.avg:.2f}")

        print(f"📊 Phase 3 Summary: Improvement: {improve_AM.avg:.4f}, Loss: {loss_AM.avg:.4f}")
        return improve_AM.avg, loss_AM.avg

    def train_phase3_step(self, reset_state, baseline=None):
        """Single training step for Phase 3"""
        self.model.train()
        self.optimizer_phase3.zero_grad()
        
        # Encode
        self.model.pre_forward(reset_state)
        
        # RL Refiner Loss
        state = reset_state.clone()
        refiner_loss, info = self.model.compute_refinement_loss_rl(state, baseline)
        
        recon_loss = self.model.compute_slot_reconstruction_loss(reset_state)

        total_loss = refiner_loss + self.lambda_recon * recon_loss

        if total_loss > 0:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.optimizer_phase3.param_groups[0]['params'], max_norm=1.0)
            self.optimizer_phase3.step()
        
        losses = {
            'total': total_loss.item(),
            'refiner_rl': refiner_loss.item(),
            **info
        }
        
        return losses

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def _validate(self, epoch):
        """Validation across all test tasks"""
        print(f"\n🔍 Validation (Epoch {epoch}, Phase {self.current_phase})...")
        
        if not hasattr(self.args, 'val_dataset_dir') or not self.args.val_dataset_dir:
            print("⚠️  No validation dataset directory specified")
            return
        
        val_tasks = self._get_validation_tasks()
        
        all_scores = {}
        for task_name, val_path in val_tasks.items():
            try:
                print(f"\n  Testing on {task_name}...")
                env = self.envs[0](**self.env_params)
                
                score, gap = self._val_and_stat(
                    dir=self.args.val_dataset_dir,
                    val_path=val_path,
                    env=env,
                    compute_gap=True
                )
                
                all_scores[f'{task_name}_score'] = score
                all_scores[f'{task_name}_gap'] = gap
                
            except Exception as e:
                print(f"  ❌ Validation for {task_name} failed: {e}")

        self.result_log["val_score"].append({f'epoch_{epoch}': all_scores})
        print(f"\n✅ Validation completed: {all_scores}\n")

    def _get_validation_tasks(self):
        """Define validation tasks"""
        problem = self.args.problem.lower()
        size = self.env_params['problem_size']
        
        tasks = {}
        if 'cvrp' in problem:
            tasks[f'CVRP{size}'] = f"cvrp/cvrp{size}_val.pkl"
        if 'vrptw' in problem or 'tw' in problem:
            tasks[f'VRPTW{size}'] = f"vrptw/vrptw{size}_val.pkl"
        
        if not tasks:
            tasks[f'{problem.upper()}{size}'] = f"{problem}/{problem}{size}_val.pkl"
        
        return tasks

    def _val_one_batch(self, data, env, aug_factor=1, eval_type="argmax"):
        """Validate one batch"""
        self.model.eval()
        self.model.set_eval_type(eval_type)
        
        batch_size = data[0].size(0) if isinstance(data, tuple) else len(data)
        
        env.load_problems(batch_size, problems=data, aug_factor=aug_factor)
        reset_state, _, _ = env.reset()
        self.model.pre_forward(reset_state)
        
        # Phase 1: Use RL only
        # Phase 2 & 3: Use Refiner (if available)
        if self.current_phase == 1 or not self.model.use_refiner:
            state, reward, done = env.pre_step()
            while not done:
                selected, _ = self.model(state, mode='rl')
                state, reward, done = env.step(selected)
        
        else:
            # Use Refiner for better quality
            with torch.no_grad():
                state_copy = reset_state.clone()
                trajectories = self.model._forward_rollout(state_copy)
                
                num_nodes = self.model.encoded_nodes.size(1)
                initial_heatmap = self.model.rollout_manager.solution_to_heatmap(
                    trajectories, num_nodes
                ).mean(dim=1)
                
                refined_heatmap = self.model.diffusion_refiner.refine(
                    initial_heatmap,
                    self.model.slots,
                    self.model.encoded_nodes,
                    enable_grad=False
                )
                
                refined_actions = self.model._heatmap_to_actions(refined_heatmap, reset_state)
                cost_refined = self.model._compute_cost_from_actions(refined_actions, reset_state)
                reward = -cost_refined.unsqueeze(1)
                reward = reward.unsqueeze(0).expand(aug_factor, -1, -1)
        
        aug_reward = reward.reshape(aug_factor, batch_size, -1)
        
        max_pomo_reward, _ = aug_reward.max(dim=2)
        no_aug_score = -max_pomo_reward[0, :].float()
        
        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)
        aug_score = -max_aug_pomo_reward.float()

        return no_aug_score, aug_score

    def _val_and_stat(self, dir, val_path, env, batch_size=500, val_episodes=1000, compute_gap=False):
        """Validation with statistics"""
        no_aug_score_list, aug_score_list = [], []
        no_aug_gap_list, aug_gap_list = [], []
        
        episode = 0
        no_aug_score = torch.zeros(0).to(self.device)
        aug_score = torch.zeros(0).to(self.device)

        while episode < val_episodes:
            remaining = val_episodes - episode
            bs = min(batch_size, remaining)
            
            data = env.load_dataset(os.path.join(dir, val_path), offset=episode, num_samples=bs)
            
            no_aug, aug = self._val_one_batch(data, env, aug_factor=8, eval_type="argmax")
            
            no_aug_score = torch.cat((no_aug_score, no_aug), dim=0)
            aug_score = torch.cat((aug_score, aug), dim=0)
            episode += bs

        no_aug_score_avg = round(no_aug_score.mean().item(), 4)
        aug_score_avg = round(aug_score.mean().item(), 4)
        
        no_aug_score_list.append(no_aug_score_avg)
        aug_score_list.append(aug_score_avg)

        if compute_gap:
            try:
                opt_sol_path = get_opt_sol_path(dir, env.problem, env.problem_size)
                opt_sol = load_dataset(opt_sol_path, disable_print=True)[:val_episodes]
                opt_sol = [sol[0] for sol in opt_sol]
                
                gap_no_aug = [(no_aug_score[j].item() - opt_sol[j]) / opt_sol[j] * 100 
                             for j in range(val_episodes)]
                no_aug_gap_avg = round(sum(gap_no_aug) / len(gap_no_aug), 4)
                no_aug_gap_list.append(no_aug_gap_avg)
                
                gap_aug = [(aug_score[j].item() - opt_sol[j]) / opt_sol[j] * 100 
                          for j in range(val_episodes)]
                aug_gap_avg = round(sum(gap_aug) / len(gap_aug), 4)
                aug_gap_list.append(aug_gap_avg)
                
                print(f"  📊 Val Results: NO_AUG → Score: {no_aug_score_avg:.4f}, Gap: {no_aug_gap_avg:.2f}% | "
                      f"AUG → Score: {aug_score_avg:.4f}, Gap: {aug_gap_avg:.2f}%")
                
                return aug_score_avg, aug_gap_avg
            
            except Exception as e:
                print(f"  ⚠️  Could not compute gap: {e}")
                compute_gap = False

        if not compute_gap:
            print(f"  📊 Val Results: NO_AUG → Score: {no_aug_score_avg:.4f} | "
                  f"AUG → Score: {aug_score_avg:.4f}")
            return aug_score_avg, 0.0


# =========================================================================
# USAGE EXAMPLE - 3-PHASE SEQUENTIAL TRAINING
# =========================================================================

"""
COMPLETE 3-PHASE SEQUENTIAL TRAINING PIPELINE

# =========================================================================
# 1. SETUP
# =========================================================================

import argparse

args = argparse.Namespace(
    problem='cvrptw',
    model_type='mtl',
    device=torch.device('cuda:0'),
    checkpoint=None,
    log_path='./logs/cvrptw_3phase',
    train_dataset_dir='./data/train',  # For Phase 2 (static data with LKH3)
    val_dataset_dir='./data/val',
)

# =========================================================================
# 2. PARAMETERS
# =========================================================================

env_params = {
    'problem_size': 50,
    'pomo_size': 50,
}

model_params = {
    'embedding_dim': 128,
    'encoder_layer_num': 6,
    'head_num': 8,
    'qkv_dim': 16,
    'ff_hidden_dim': 512,
    'eval_type': 'softmax',
    'problem': 'cvrptw',
    'problem_size': 50,
    'slot_num': 16,
    'slot_iter_num': 3,
    'norm': 'layer',
    'norm_loc': 'norm_last',
    'sqrt_embedding_dim': 11.31,
    'logit_clipping': 10,
    'use_diffusion_refiner': True,
    'refiner_timesteps': 10,
    'refiner_hidden_dim': 256,
    'refiner_noise_scale': 0.1,
    'enable_slot_reconstruction': False,
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,          # Phase 1 LR
        'lr_phase2': 1e-4,   # Phase 2 LR
        'lr_phase3': 1e-5,   # Phase 3 LR (smaller for fine-tune)
    },
    'scheduler': {
        'milestones': [51, 76],
        'gamma': 0.1,
    }
}

trainer_params = {
    'epochs': 100,
    'phase1_epochs': 40,   # Phase 1 ends at epoch 40
    'phase2_epochs': 70,   # Phase 2 ends at epoch 70
    'train_episodes': 10000,
    'train_batch_size': 64,
    'validation_interval': 5,
    'model_save_interval': 10,
    'lambda_recon': 0.0,
    'rl_weight': 1.0,
    'refiner_weight': 0.1,
}

# =========================================================================
# 3. CREATE TRAINER
# =========================================================================

trainer = Trainer(
    args=args,
    env_params=env_params,
    model_params=model_params,
    optimizer_params=optimizer_params,
    trainer_params=trainer_params
)

# =========================================================================
# 4. RUN TRAINING
# =========================================================================

trainer.run()

# Training will automatically:
# - Phase 1 (Epochs 1-40):   Train RL Decoder with random data
# - Phase 2 (Epochs 41-70):  Train Refiner with static data (supervised)
# - Phase 3 (Epochs 71-100): Fine-tune Refiner with random data (RL)

# =========================================================================
# 5. DATASET PREPARATION
# =========================================================================

# Phase 1: No dataset needed (random on-the-fly)

# Phase 2: Prepare static dataset with LKH3 solutions
#   File: ./data/train/cvrptw/cvrptw_train_with_lkh3.pkl
#   Format: List of (problem_dict, optimal_tour) tuples
#   
#   Generate using:
#     python generate_optimal_dataset.py --problem cvrptw --size 50 --num 10000

# Phase 3: No dataset needed (random on-the-fly)

# Validation dataset:
#   ./data/val/cvrptw/cvrptw50_val.pkl  (problems)
#   ./data/val/cvrptw/cvrptw50_opt.pkl  (optimal solutions)

# =========================================================================
# 6. EXPECTED BEHAVIOR
# =========================================================================

# Phase 1 (Epochs 1-40):
#   - Model trains like standard RL (POMO/AM)
#   - Fair comparison with baselines
#   - Validation: RL performance only
#   - Checkpoint: phase1_final_epoch-40.pt

# Phase 2 (Epochs 41-70):
#   - RL Decoder frozen
#   - Refiner learns from LKH3
#   - Encoder fine-tunes to support Refiner
#   - Validation: RL + Refiner (improved)
#   - Checkpoint: phase2_final_epoch-70.pt

# Phase 3 (Epochs 71-100):
#   - RL Decoder still frozen
#   - Refiner fine-tunes on random data
#   - Generalization to unseen problems
#   - Validation: Best performance
#   - Checkpoint: epoch-100.pt (final)

# =========================================================================
# 7. COMPARING WITH BASELINES
# =========================================================================

# Fair Comparison (Recommended):
# 1. Train baselines (POMO, AM) on same random data distribution
# 2. Compare Phase 1 results with baselines (fair - same data)
# 3. Show improvement from Phase 1 → Phase 3 (your contribution)

# Table Example:
# | Model              | Training Data | Test Score | Gap (%) |
# |--------------------|---------------|------------|---------|
# | POMO (baseline)    | Random        | 10.50      | 2.5%    |
# | AM (baseline)      | Random        | 10.80      | 5.3%    |
# | Ours (Phase 1)     | Random        | 10.45      | 2.3%    | ← Fair
# | Ours (Phase 2)     | Static+Random | 10.30      | 1.6%    |
# | Ours (Phase 3)     | Static+Random | 10.20      | 1.1%    | ← Final

# Key findings:
# - Phase 1 matches/beats baselines (fair comparison)
# - Phase 2 + 3 provide +0.25 improvement (our contribution)

# =========================================================================
# 8. INFERENCE
# =========================================================================

# Load final model
checkpoint = torch.load('./logs/cvrptw_3phase/epoch-100.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Fast inference (RL only - Phase 1 quality)
model.eval()
model.set_eval_type('argmax')
with torch.no_grad():
    model.pre_forward(test_batch)
    state = test_batch.clone()
    state, reward, done = env.pre_step()
    while not done:
        selected, _ = model(state, mode='rl')
        state, reward, done = env.step(selected)
    cost_rl = -reward.max(dim=1)[0].mean()

# High-quality inference (RL + Refiner - Phase 3 quality)
with torch.no_grad():
    model.pre_forward(test_batch)
    state = test_batch.clone()
    
    # Generate initial with RL
    trajectories = model._forward_rollout(state)
    
    # Refine
    num_nodes = model.encoded_nodes.size(1)
    initial_heatmap = model.rollout_manager.solution_to_heatmap(
        trajectories, num_nodes
    ).mean(dim=1)
    
    refined_heatmap = model.diffusion_refiner.refine(
        initial_heatmap,
        model.slots,
        model.encoded_nodes,
        enable_grad=False
    )
    
    refined_actions = model._heatmap_to_actions(refined_heatmap, state)
    cost_refined = model._compute_cost_from_actions(refined_actions, state).mean()

print(f"Cost (RL only):     {cost_rl:.2f}")
print(f"Cost (RL+Refiner):  {cost_refined:.2f}")
print(f"Improvement:        {(cost_rl - cost_refined):.2f}")

# =========================================================================
# 9. ABLATION STUDIES (Optional)
# =========================================================================

# To strengthen your paper, consider these ablations:

# Ablation 1: Skip Phase 2 (No supervised pre-training)
#   - Train Phase 1 → directly to Phase 3
#   - Show that Phase 2 helps bootstrap Refiner

# Ablation 2: Skip Phase 3 (No RL fine-tuning)
#   - Train Phase 1 → Phase 2 only
#   - Show that Phase 3 improves generalization

# Ablation 3: Different phase durations
#   - 30/40/30 vs 40/30/30 vs 50/30/20
#   - Find optimal balance

# =========================================================================
# 10. MONITORING & DEBUGGING
# =========================================================================

# Watch for these during training:

# Phase 1:
#   - RL loss should decrease steadily
#   - Validation cost should decrease
#   - Should match baseline performance

# Phase 2:
#   - Refiner supervised loss should converge (→ ~0.01)
#   - Validation should show immediate improvement
#   - If loss stays high, check:
#       * LKH3 solutions are correct
#       * Heatmap conversion works
#       * Encoder is not frozen

# Phase 3:
#   - Improvement metric should be positive
#   - Cost_refined < Cost_initial
#   - If improvement is negative:
#       * Increase training epochs
#       * Reduce LR
#       * Check if Phase 2 converged

# =========================================================================
# 11. TIPS & TRICKS
# =========================================================================

# 1. Phase 1 duration:
#    - Train until convergence (usually 40-50 epochs)
#    - Must match baseline quality for fair comparison

# 2. Phase 2 duration:
#    - Stop when supervised loss plateaus
#    - Usually 20-30 epochs sufficient

# 3. Phase 3 duration:
#    - Longer is better for generalization
#    - At least 20-30 epochs

# 4. Learning rates:
#    - Phase 1: 1e-4 (standard)
#    - Phase 2: 1e-4 (Refiner is new, needs higher LR)
#    - Phase 3: 1e-5 (fine-tuning, use smaller LR)

# 5. Validation:
#    - Validate every 5 epochs
#    - Watch for Phase 1→2 jump (should see improvement)
#    - Watch for Phase 2→3 improvement (smaller, but consistent)

# 6. Checkpoints:
#    - Always save phase transition checkpoints
#    - They're useful for ablations and analysis
"""