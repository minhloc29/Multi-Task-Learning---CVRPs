import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['MTLModel']

"""
MTLModel with RL Decoder + Diffusion Refiner - 3-PHASE SEQUENTIAL VERSION (FIXED)

CRITICAL FIXES:
1. Added tw_width feature (6 features total)
2. Fixed _compute_cost_from_actions with proper state handling
3. Added tour_to_heatmap helper function
4. Improved _heatmap_to_actions with feasibility checking
5. Added validation in freeze/unfreeze methods
"""

# =========================================================================
# HELPER FUNCTIONS (MOVED TO TOP)
# =========================================================================

def tour_to_heatmap(tours, num_nodes):
    """
    Convert tour sequences to visit frequency heatmap
    
    Args:
        tours: (batch, tour_length) tour sequences with node indices
        num_nodes: Total number of nodes (including depot)
    
    Returns:
        heatmap: (batch, num_nodes) normalized visit frequencies
    """
    batch_size = tours.size(0)
    device = tours.device
    
    heatmap = torch.zeros(batch_size, num_nodes, device=device)
    
    # Count visits for each node
    for i in range(tours.size(1)):
        node_idx = tours[:, i]
        heatmap.scatter_add_(1, node_idx.unsqueeze(1), 
                            torch.ones(batch_size, 1, device=device))
    
    # Normalize by max visits
    max_visits = heatmap.max(dim=1, keepdim=True)[0] + 1e-8
    heatmap = heatmap / max_visits
    
    return heatmap


def reshape_by_heads(qkv, head_num):
    batch_s = qkv.size(0)
    n = qkv.size(1)
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    q_transposed = q_reshaped.transpose(1, 2)
    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    out = torch.matmul(weights, v)
    out_transposed = out.transpose(1, 2)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)

    return out_concat


def _get_encoding(encoded_nodes, node_index_to_pick):
    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)

    return picked_nodes


# =========================================================================
# SLOT ATTENTION MODULES
# =========================================================================

class IterativeAttention(nn.Module):
    def __init__(self, embedding_dim, slot_dim, head_num, qkv_dim):
        super().__init__()
        self.scale = qkv_dim ** -0.5
        
        self.Wq = nn.Linear(slot_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, slot_dim)

        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.norm = nn.LayerNorm(slot_dim)
        
        self.head_num = head_num
        self.qkv_dim = qkv_dim
        
    def forward(self, slot_features, node_features, return_attention=False):
        B, N, D_emb = node_features.shape
        K_total, D_slot = slot_features.shape
        K = K_total // B

        q = reshape_by_heads(self.Wq(slot_features).reshape(B, K, -1), head_num=self.head_num)
        k = reshape_by_heads(self.Wk(node_features), head_num=self.head_num)
        v = reshape_by_heads(self.Wv(node_features), head_num=self.head_num)

        scores = torch.matmul(q, k.transpose(2, 3)) * self.scale
        weights = F.softmax(scores, dim=-1)

        attention_map = weights.mean(dim=1) if return_attention else None

        slot_updates = torch.matmul(weights, v)
        slot_updates = slot_updates.transpose(1, 2).reshape(B, K, -1)
        slot_updates = self.multi_head_combine(slot_updates).reshape(B * K, D_slot)

        slots_normalized = self.norm(slot_features)
        slots_updated = self.gru(slot_updates, slots_normalized)

        if return_attention:
            return slots_updated, attention_map
        return slots_updated


class SlotAttentionModule(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        slot_num = model_params.get('slot_num', 16)
        head_num = model_params['head_num']
        qkv_dim = model_params['qkv_dim']
        
        self.slot_num = slot_num
        self.iter_num = model_params.get('slot_iter_num', 3)
        self.embedding_dim = embedding_dim
        
        self.slots_mu = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.slots_log_sigma = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
        self.iterative_attention = IterativeAttention(embedding_dim, embedding_dim, head_num, qkv_dim)
        
        self.last_slots = None
        self.last_attention = None

    def forward(self, node_features):
        B, N, D = node_features.shape
        
        mu = self.slots_mu.expand(B, self.slot_num, D)
        log_sigma = self.slots_log_sigma.expand(B, self.slot_num, D)
        sigma = torch.exp(log_sigma)
        
        eps = torch.randn_like(mu)
        initial_slots = mu + sigma * eps
        
        slots = initial_slots.reshape(B * self.slot_num, D)
        
        attention_maps = []
        for i in range(self.iter_num):
            if i == self.iter_num - 1:
                slots, attn = self.iterative_attention(slots, node_features, return_attention=True)
                attention_maps.append(attn)
            else:
                slots = self.iterative_attention(slots, node_features, return_attention=False)
        
        slots_out = slots.reshape(B, self.slot_num, D)
        
        self.last_slots = slots_out
        self.last_attention = attention_maps[-1] if attention_maps else None

        return slots_out


# =========================================================================
# DIFFUSION REFINER
# =========================================================================

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class LightweightDenoiser(nn.Module):
    def __init__(self, embedding_dim, num_nodes, hidden_dim=256, num_heads=8):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_nodes = num_nodes
        
        time_dim = embedding_dim * 2
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(embedding_dim),
            nn.Linear(embedding_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        self.route_proj = nn.Linear(num_nodes, hidden_dim)
        
        self.slot_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            kdim=embedding_dim,
            vdim=embedding_dim,
            batch_first=True
        )
        
        self.node_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            kdim=embedding_dim,
            vdim=embedding_dim,
            batch_first=True
        )
        
        self.process = nn.Sequential(
            nn.Linear(hidden_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.output_proj = nn.Linear(hidden_dim, num_nodes)
        
    def forward(self, x, t, slots, encoded_nodes):
        batch_size = x.size(0)
        
        t_emb = self.time_mlp(t)
        h = self.route_proj(x.unsqueeze(-1))
        
        h_slot, _ = self.slot_cross_attn(h, slots, slots)
        h = h + h_slot
        
        h_node, _ = self.node_cross_attn(h, encoded_nodes, encoded_nodes)
        h = h + h_node
        
        t_emb_expanded = t_emb.unsqueeze(1).expand(-1, h.size(1), -1)
        h = torch.cat([h, t_emb_expanded], dim=-1)
        
        h = self.process(h)
        correction = self.output_proj(h).squeeze(-1)
        
        return correction


class DiffusionRefiner(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params['embedding_dim']
        
        self.timesteps = model_params.get('refiner_timesteps', 10)
        self.num_nodes = model_params.get('problem_size', 100) + 1
        
        self.betas = cosine_beta_schedule(self.timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        self.denoiser = LightweightDenoiser(
            embedding_dim=embedding_dim,
            num_nodes=self.num_nodes,
            hidden_dim=model_params.get('refiner_hidden_dim', 256),
            num_heads=model_params['head_num']
        )
        
        self.noise_scale = model_params.get('refiner_noise_scale', 0.1)
        
    def extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def add_noise(self, x_clean, t):
        noise = torch.randn_like(x_clean) * self.noise_scale
        
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_clean.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_clean.shape)
        
        return sqrt_alphas_cumprod_t * x_clean + sqrt_one_minus_alphas_cumprod_t * noise
    
    def denoise_step(self, x_noisy, t, slots, encoded_nodes):
        correction = self.denoiser(x_noisy, t, slots, encoded_nodes)
        
        betas_t = self.extract(self.betas, t, x_noisy.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_noisy.shape)
        sqrt_recip_alphas_t = self.extract(1. / torch.sqrt(self.alphas), t, x_noisy.shape)
        
        x_denoised = sqrt_recip_alphas_t * (
            x_noisy - betas_t * correction / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t[0] > 0:
            noise = torch.randn_like(x_noisy) * self.noise_scale
            posterior_variance_t = self.extract(self.betas, t, x_noisy.shape)
            x_denoised = x_denoised + torch.sqrt(posterior_variance_t) * noise
        
        return x_denoised
    
    def refine(self, initial_solution, slots, encoded_nodes, num_steps=None, enable_grad=False):
        if not enable_grad:
            with torch.no_grad():
                return self._refine_impl(initial_solution, slots, encoded_nodes, num_steps)
        else:
            return self._refine_impl(initial_solution, slots, encoded_nodes, num_steps)
    
    def _refine_impl(self, initial_solution, slots, encoded_nodes, num_steps=None):
        device = initial_solution.device
        batch_size = initial_solution.size(0)
        
        if num_steps is None:
            num_steps = self.timesteps
        
        start_t = min(self.timesteps - 1, num_steps)
        
        t_start = torch.full((batch_size,), start_t, device=device, dtype=torch.long)
        x = self.add_noise(initial_solution, t_start)
        
        for i in reversed(range(start_t + 1)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.denoise_step(x, t, slots, encoded_nodes)
        
        return x
    
    def compute_supervised_loss(self, clean_solution, slots, encoded_nodes):
        batch_size = clean_solution.size(0)
        device = clean_solution.device
        
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        
        noisy_solution = self.add_noise(clean_solution, t)
        predicted_correction = self.denoiser(noisy_solution, t, slots, encoded_nodes)
        
        true_correction = noisy_solution - clean_solution
        
        loss = F.mse_loss(predicted_correction, true_correction)
        return loss


# =========================================================================
# ROLLOUT UTILITIES
# =========================================================================

class RolloutManager:
    """Manages episode rollout for RL training and refinement"""
    
    @staticmethod
    def rollout_episode(model, state, max_steps=None, deterministic=False):
        device = state.depot_xy.device
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        
        if max_steps is None:
            max_steps = state.node_xy.size(1) + 1
        
        actions_list = []
        probs_list = []
        states_list = []
        
        step = 0
        done = False
        
        while not done and step < max_steps:
            if deterministic:
                old_eval = model.eval_type
                model.eval_type = 'argmax'
            
            selected, prob = model._forward_rl(state, selected=None)
            
            if deterministic:
                model.eval_type = old_eval
            
            actions_list.append(selected.clone())
            probs_list.append(prob.clone())
            states_list.append({
                'current_node': state.current_node.clone(),
                'load': state.load.clone(),
                'current_time': state.current_time.clone(),
                'ninf_mask': state.ninf_mask.clone(),
            })
            
            state, reward, done = state.step(selected)
            step += 1
        
        trajectories = {
            'actions': torch.stack(actions_list, dim=2),
            'probs': torch.stack(probs_list, dim=2),
            'states': states_list,
            'final_reward': reward if not done else None,
        }
        
        return trajectories
    
    @staticmethod
    def compute_solution_cost(state, trajectories):
        if hasattr(state, 'get_cost'):
            return state.get_cost()
        elif hasattr(state, 'length'):
            return state.length
        else:
            raise NotImplementedError("State must have get_cost() or length attribute")
    
    @staticmethod
    def solution_to_heatmap(trajectories, num_nodes):
        actions = trajectories['actions']
        batch_size, pomo_size, num_steps = actions.shape
        device = actions.device
        
        heatmap = torch.zeros(batch_size, pomo_size, num_nodes, device=device)
        
        for step in range(num_steps):
            action = actions[:, :, step]
            heatmap.scatter_add_(2, action.unsqueeze(2), 
                                 torch.ones_like(action, dtype=torch.float).unsqueeze(2))
        
        max_visits = heatmap.max(dim=2, keepdim=True)[0] + 1e-8
        heatmap = heatmap / max_visits
        
        return heatmap


# =========================================================================
# MTL MODEL
# =========================================================================

class MTLModel(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.eval_type = self.model_params['eval_type']
        self.problem = self.model_params['problem']

        self.encoder = MTL_Encoder(**model_params)
        self.decoder_rl = MTL_Decoder(**model_params)
        
        self.use_refiner = model_params.get('use_diffusion_refiner', False)
        if self.use_refiner:
            self.diffusion_refiner = DiffusionRefiner(**model_params)
        
        self.enable_reconstruction = model_params.get('enable_slot_reconstruction', False)
        if self.enable_reconstruction:
            self.reconstruction_head = nn.Sequential(
                nn.Linear(model_params['embedding_dim'], 256),
                nn.ReLU(),
                nn.Linear(256, 6)  # FIXED: 6 features (x, y, demand, tw_start, tw_end, tw_width)
            )
        
        self.rollout_manager = RolloutManager()
        
        self.encoded_nodes = None
        self.slots = None
        self.device = torch.device('cuda', torch.cuda.current_device()) if 'device' not in model_params.keys() else model_params['device']

    def pre_forward(self, reset_state):
        """Encode problem instance"""
        depot_xy = reset_state.depot_xy
        node_xy = reset_state.node_xy
        node_demand = reset_state.node_demand
        node_tw_start = reset_state.node_tw_start
        node_tw_end = reset_state.node_tw_end
        
        # FIXED: Calculate tw_width and include as 6th feature
        tw_width = node_tw_end - node_tw_start
        
        node_xy_demand_tw = torch.cat(
            (node_xy, node_demand[:, :, None], 
             node_tw_start[:, :, None], node_tw_end[:, :, None],
             tw_width[:, :, None]),  # Added tw_width
            dim=2
        )

        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand_tw)
        self.slots = self.encoder.slot_attention_module.last_slots
        
        self.decoder_rl.set_kv(self.encoded_nodes, slots=self.slots)

    def set_eval_type(self, eval_type):
        self.eval_type = eval_type

    def forward(self, state, selected=None, mode='rl'):
        """
        Forward pass with different modes
        
        Args:
            state: VRP state
            selected: Pre-selected actions (optional)
            mode: 'rl' for standard RL, 'refine' for RL+refinement, 'rollout' for full episode
        
        Returns:
            Depends on mode:
            - 'rl': (selected, prob)
            - 'refine': refined_solution
            - 'rollout': trajectories dict
        """
        if mode == 'rl':
            return self._forward_rl(state, selected)
        elif mode == 'refine':
            assert self.use_refiner, "Diffusion refiner not enabled"
            return self._forward_with_refinement(state)
        elif mode == 'rollout':
            return self._forward_rollout(state)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _forward_rl(self, state, selected=None):
        """Standard RL forward (single step)"""
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.selected_count == 0:
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long).to(self.device)
            prob = torch.ones(size=(batch_size, pomo_size))

        elif state.selected_count == 1:
            selected = state.START_NODE
            prob = torch.ones(size=(batch_size, pomo_size))

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            attr = torch.cat(
                (state.load[:, :, None], state.current_time[:, :, None],
                 state.length[:, :, None], state.open[:, :, None]), 
                dim=2
            )
            
            probs = self.decoder_rl(encoded_last_node, attr, ninf_mask=state.ninf_mask)
            
            if selected is None:
                while True:
                    if self.training or self.eval_type == 'softmax':
                        try:
                            selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                                .squeeze(dim=1).reshape(batch_size, pomo_size)
                        except Exception as exception:
                            print(f">> Catch Exception: {exception}")
                            exit(0)
                    else:
                        selected = probs.argmax(dim=2)
                    
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    if (prob != 0).all():
                        break
            else:
                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)

        return selected, prob
    
    def _forward_rollout(self, state):
        """Full episode rollout"""
        return self.rollout_manager.rollout_episode(self, state)
    
    def _forward_with_refinement(self, state):
        """
        Generate solution with RL then refine with diffusion
        
        Args:
            state: Initial VRP state
        
        Returns:
            refined_trajectories: Refined solution trajectories
        """
        # Step 1: Generate initial solution with RL
        with torch.no_grad():
            initial_trajectories = self.rollout_manager.rollout_episode(
                self, state, deterministic=True
            )
        
        # Step 2: Convert to heatmap
        num_nodes = self.encoded_nodes.size(1)
        initial_heatmap = self.rollout_manager.solution_to_heatmap(
            initial_trajectories, num_nodes
        )
        
        # Average over POMO for refinement
        batch_size, pomo_size = initial_heatmap.shape[:2]
        heatmap_avg = initial_heatmap.mean(dim=1)  # (batch, num_nodes)
        
        # Step 3: Refine with diffusion
        refined_heatmap = self.diffusion_refiner.refine(
            heatmap_avg,
            self.slots,
            self.encoded_nodes,
            num_steps=self.diffusion_refiner.timesteps,
            enable_grad=False
        )
        
        # Step 4: Convert back to actions (greedy decoding from heatmap)
        refined_actions = self._heatmap_to_actions(refined_heatmap, state)
        
        return {
            'initial_trajectories': initial_trajectories,
            'refined_heatmap': refined_heatmap,
            'refined_actions': refined_actions,
        }
    
    def _heatmap_to_actions(self, heatmap, state):
        """
        FIXED: Convert heatmap to action sequence with feasibility checking
        
        Args:
            heatmap: (batch, num_nodes) visit probabilities
            state: VRP state for feasibility checking
        
        Returns:
            actions: (batch, max_steps) action sequence
        """
        batch_size = heatmap.size(0)
        num_nodes = heatmap.size(1)
        max_steps = num_nodes
        device = heatmap.device
        
        actions = []
        
        # Clone initial state for simulation
        current_load = torch.ones(batch_size, device=device) * state.VEHICLE_CAPACITY
        current_time = torch.zeros(batch_size, device=device)
        current_node = torch.zeros(batch_size, dtype=torch.long, device=device)
        visited = torch.zeros(batch_size, num_nodes, device=device, dtype=torch.bool)
        visited[:, 0] = True  # Depot visited
        
        for step in range(max_steps - 1):
            # Mask visited nodes
            masked_heatmap = heatmap.clone()
            masked_heatmap[visited] = -float('inf')
            
            # Check feasibility for CVRPTW
            feasible_mask = self._get_feasible_mask(
                current_node, current_load, current_time, 
                visited, state
            )
            masked_heatmap[~feasible_mask] = -float('inf')
            
            # If no feasible node, return to depot
            all_infeasible = (masked_heatmap == -float('inf')).all(dim=1)
            if all_infeasible.any():
                next_node = torch.zeros(batch_size, dtype=torch.long, device=device)
                next_node[~all_infeasible] = masked_heatmap[~all_infeasible].argmax(dim=1)
            else:
                next_node = masked_heatmap.argmax(dim=1)
            
            actions.append(next_node)
            
            # Update state
            visited.scatter_(1, next_node.unsqueeze(1), True)
            
            # Update load and time (simplified - use actual state update if available)
            is_depot = (next_node == 0)
            current_load = torch.where(
                is_depot,
                torch.ones_like(current_load) * state.VEHICLE_CAPACITY,
                current_load - state.node_demand[torch.arange(batch_size), next_node]
            )
            
            # Travel time (Euclidean distance)
            current_xy = state.depot_xy if step == 0 else state.node_xy[torch.arange(batch_size), current_node]
            next_xy = torch.where(
                is_depot.unsqueeze(1).expand(-1, 2),
                state.depot_xy,
                state.node_xy[torch.arange(batch_size), next_node]
            )
            travel_time = torch.norm(next_xy - current_xy, dim=1)
            current_time = torch.where(
                is_depot,
                torch.zeros_like(current_time),
                current_time + travel_time
            )
            
            # Wait for time window
            if not is_depot.all():
                tw_start = state.node_tw_start[torch.arange(batch_size), next_node]
                current_time = torch.maximum(current_time, tw_start)
            
            current_node = next_node
            
            # Early termination if all customers visited
            if visited[:, 1:].all():
                break
        
        # Return to depot
        actions.append(torch.zeros(batch_size, dtype=torch.long, device=device))
        
        return torch.stack(actions, dim=1)  # (batch, max_steps)
    
    def _get_feasible_mask(self, current_node, current_load, current_time, visited, state):
        """
        Check which nodes are feasible to visit next
        
        Returns:
            feasible_mask: (batch, num_nodes) boolean mask
        """
        batch_size = current_node.size(0)
        num_nodes = visited.size(1)
        device = current_node.device
        
        feasible_mask = torch.ones(batch_size, num_nodes, device=device, dtype=torch.bool)
        
        # Already visited nodes are infeasible
        feasible_mask = feasible_mask & ~visited
        
        # Depot is always feasible
        feasible_mask[:, 0] = True
        
        # Check capacity constraint
        for i in range(batch_size):
            demands = state.node_demand[i]
            feasible_mask[i] = feasible_mask[i] & (demands <= current_load[i])
        
        # Check time window constraint
        for i in range(batch_size):
            current_xy = state.node_xy[i, current_node[i]] if current_node[i] > 0 else state.depot_xy[i]
            
            for j in range(1, num_nodes):  # Skip depot
                if not feasible_mask[i, j]:
                    continue
                
                # Calculate arrival time
                next_xy = state.node_xy[i, j]
                travel_time = torch.norm(next_xy - current_xy)
                arrival_time = current_time[i] + travel_time
                
                # Check if can arrive before time window closes
                tw_end = state.node_tw_end[i, j]
                if arrival_time > tw_end:
                    feasible_mask[i, j] = False
        
        return feasible_mask
    
    def compute_refinement_loss_supervised(self, optimal_solution):
        """
        STRATEGY 1: Supervised training with ground truth
        Train refiner to denoise solutions toward optimal
        
        Args:
            optimal_solution: (batch, num_nodes) from LKH3/HGS (heatmap format)
        
        Returns:
            loss: Supervised loss for refiner
        """
        if not self.use_refiner:
            return torch.tensor(0.0, device=self.device)
        
        loss = self.diffusion_refiner.compute_supervised_loss(
            optimal_solution,
            self.slots,
            self.encoded_nodes
        )
        
        return loss
    
    def compute_refinement_loss_rl(self, state, baseline=None):
        """
        STRATEGY 2: RL training (REINFORCE)
        CRITICAL FIX v2: Sử dụng REINFORCE với log_prob từ
        hàm helper _compute_cost_from_refined_heatmap
        """
        if not self.use_refiner:
            return torch.tensor(0.0, device=self.device), {}
        
        # --- Step 1 & 2: Tạo cost ban đầu và heatmap ban đầu (giống như cũ) ---
        with torch.no_grad():
            initial_trajectories = self.rollout_manager.rollout_episode(
                self, state, deterministic=False
            )
            cost_initial = self.rollout_manager.compute_solution_cost(
                state, initial_trajectories
            ).mean(dim=1)  # (batch,)
        
        num_nodes = self.encoded_nodes.size(1)
        initial_heatmap = self.rollout_manager.solution_to_heatmap(
            initial_trajectories, num_nodes
        ).mean(dim=1)  # (batch, num_nodes)
        
        # --- Step 3: Refine heatmap (giống như cũ) ---
        refined_heatmap = self.diffusion_refiner.refine(
            initial_heatmap,
            self.slots,
            self.encoded_nodes,
            num_steps=5,  
            enable_grad=True
        )
        
        # --- Step 4 (FIXED): Lấy cả cost (detached) và log_prob (có gradient) ---
        # Không cần gọi _heatmap_to_actions ở đây nữa.
        cost_refined_detached, log_prob_policy = self._compute_cost_from_refined_heatmap(
            refined_heatmap, state
        )
        
        # --- Step 5 & 6 (FIXED): Tính reward và advantage từ cost đã detached ---
        reward = cost_initial.detach() - cost_refined_detached # Cả 2 đều detached
        
        if baseline is not None:
            advantage = reward - baseline.detach()
        else:
            advantage = reward - reward.mean().detach()
        
        # --- Step 7 (FIXED): Loss REINFORCE ---
        # Đây là công thức Policy Gradient: -(advantage * log_prob)
        # Gradient sẽ đi từ 'log_prob_policy' ngược về 'refined_heatmap'
        loss = -(advantage.detach() * log_prob_policy).mean()
        
        info = {
            'cost_initial': cost_initial.mean().item(),
            'cost_refined': cost_refined_detached.mean().item(), # Dùng cost detached
            'improvement': (cost_initial.mean() - cost_refined_detached.mean()).item(),
            'advantage': advantage.mean().item(),
            'log_prob': log_prob_policy.mean().item()
        }
        
        return loss, info
    
    def _compute_cost_from_refined_heatmap(self, refined_heatmap, initial_state):
        """
        CRITICAL FIX v2: Use REINFORCE - compute cost AND log_probs
        
        Strategy: 
        1. Combine refined_heatmap with RL decoder to guide action selection
        2. Track log_probs of selected actions (these have gradient!)
        3. Return cost (detached) and log_probs (with gradient)
        4. Use Policy Gradient loss: -(advantage * log_prob)
        
        This solves the non-differentiable environment problem!
        
        Args:
            refined_heatmap: (batch, num_nodes) refined visit probabilities WITH GRADIENT
            initial_state: Initial VRP state
        
        Returns:
            cost: (batch,) total cost (detached, used as reward)
            log_prob_total: (batch,) sum of log probs (WITH gradient for REINFORCE)
        """
        batch_size = refined_heatmap.size(0)
        device = refined_heatmap.device
        
        # Clone state for rollout
        state = initial_state.clone()
        
        # Encode if not already done
        if self.encoded_nodes is None:
            self.pre_forward(initial_state)
        
        # CRITICAL: Convert heatmap to log-space for stable combination
        log_heatmap = torch.log(refined_heatmap + 1e-9)  # (batch, num_nodes)
        
        # Hyperparameters
        heatmap_weight = 0.5  # Weight for refined heatmap guidance
        temperature = 1.0     # Temperature for action distribution
        
        # Initialize log_prob accumulator
        pomo_size = state.POMO_IDX.size(1) if hasattr(state, 'POMO_IDX') else 1
        total_log_prob = torch.zeros(batch_size, pomo_size, device=device)
        
        # Rollout loop
        step_count = 0
        max_steps = state.node_xy.size(1) + 10
        
        while step_count < max_steps:
            if state.selected_count == 0:
                # Step 0: Start from depot
                selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long, device=device)
                state, _, done = state.step(selected)
                step_count += 1
                if done:
                    break
                continue
            
            elif state.selected_count == 1:
                # Step 1: Select start node
                selected = state.START_NODE
                state, _, done = state.step(selected)
                step_count += 1
                if done:
                    break
                continue
            
            else:
                # Step 2+: Use combined probabilities and track log_probs
                
                # Get RL decoder probabilities
                encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
                attr = torch.cat(
                    (state.load[:, :, None], state.current_time[:, :, None],
                     state.length[:, :, None], state.open[:, :, None]), 
                    dim=2
                )
                
                probs_rl = self.decoder_rl(encoded_last_node, attr, ninf_mask=state.ninf_mask)
                # probs_rl: (batch, pomo, num_nodes)
                
                # Convert to log-space
                log_probs_rl = torch.log(probs_rl + 1e-9)
                
                # Expand heatmap to match RL decoder shape
                log_heatmap_expanded = log_heatmap.unsqueeze(1).expand(-1, pomo_size, -1)
                # log_heatmap_expanded: (batch, pomo, num_nodes)
                
                # Combine in log-space
                combined_logits = (
                    (1 - heatmap_weight) * log_probs_rl + 
                    heatmap_weight * log_heatmap_expanded
                ) / temperature
                
                # Apply feasibility mask
                combined_logits = combined_logits + state.ninf_mask
                
                # Convert to probabilities
                combined_probs = F.softmax(combined_logits, dim=2)
                # combined_probs: (batch, pomo, num_nodes) WITH gradient connection
                
                # Select action (argmax for deterministic, or sample for stochastic)
                if self.training:
                    # Stochastic sampling for exploration
                    selected = combined_probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                        .squeeze(dim=1).reshape(batch_size, pomo_size)
                else:
                    # Greedy for evaluation
                    selected = combined_probs.argmax(dim=2)
                # selected: (batch, pomo)
                
                # CRITICAL: Get log_prob of selected actions
                # This is where gradient flows through!
                log_prob_step = torch.log(combined_probs + 1e-9).gather(
                    dim=2, 
                    index=selected.unsqueeze(2)
                ).squeeze(2)  # (batch, pomo)
                
                # Accumulate log_probs (only for active tours)
                # Mask out finished tours if needed
                if hasattr(state, 'finished_mask'):
                    active_mask = ~state.finished_mask
                    log_prob_step = log_prob_step * active_mask.float()
                
                total_log_prob = total_log_prob + log_prob_step
                
                # Step environment (discrete, non-differentiable)
                state, reward, done = state.step(selected)
                step_count += 1
                
                if done:
                    break
        
        # Get final cost (detached - used as reward signal)
        if hasattr(state, 'get_cost'):
            cost = state.get_cost()
        elif hasattr(state, 'length'):
            cost = state.length
        else:
            raise NotImplementedError("State must have get_cost() or length attribute")
        
        # Average over POMO if needed
        if cost.dim() > 1:
            cost = cost.mean(dim=1)  # (batch,)
        
        # Average log_probs over POMO
        log_prob_total = total_log_prob.mean(dim=1)  # (batch,)
        
        # Return:
        # - cost: detached (no gradient, used as reward)
        # - log_prob_total: WITH gradient (used in REINFORCE)
        return cost.detach(), log_prob_total
    
    def compute_slot_reconstruction_loss(self, reset_state):
        """FIXED: Auxiliary loss with 6 features"""
        if not self.enable_reconstruction or self.slots is None:
            return torch.tensor(0.0, device=self.device)
        
        node_xy = reset_state.node_xy
        node_demand = reset_state.node_demand
        node_tw_start = reset_state.node_tw_start
        node_tw_end = reset_state.node_tw_end
        tw_width = node_tw_end - node_tw_start  # FIXED: Calculate tw_width
        
        original_features = torch.cat(
            (node_xy, node_demand[:, :, None],
             node_tw_start[:, :, None], node_tw_end[:, :, None],
             tw_width[:, :, None]),  # FIXED: Added tw_width
            dim=2
        )
        
        attention = self.encoder.slot_attention_module.last_attention
        if attention is None:
            return torch.tensor(0.0, device=self.device)
        
        attention_customers = attention[:, :, 1:]  # Exclude depot
        
        slot_predictions = self.reconstruction_head(self.slots)
        reconstructed_features = torch.einsum(
            'bkn,bkf->bnf',
            attention_customers,
            slot_predictions
        )
        
        recon_loss = F.mse_loss(reconstructed_features, original_features[:, 1:, :])
        return recon_loss
    
    def compute_slot_contrastive_loss(self):
        """Auxiliary loss: encourage slot diversity"""
        if self.slots is None:
            return torch.tensor(0.0, device=self.device)
        
        slots_norm = F.normalize(self.slots, dim=-1)
        similarity = torch.bmm(slots_norm, slots_norm.transpose(1, 2))
        
        batch, K, _ = similarity.shape
        mask = torch.eye(K, device=self.device).unsqueeze(0).expand(batch, -1, -1)
        off_diag_sim = similarity * (1 - mask)
        
        contrastive_loss = (off_diag_sim ** 2).sum() / (batch * K * (K - 1))
        return contrastive_loss
    
    # =========================================================================
    # 3-PHASE FREEZE/UNFREEZE CONTROLS (FIXED with validation)
    # =========================================================================
    
    def freeze_all(self):
        """Freeze everything"""
        print("🔒 Freezing: ALL parameters")
        for param in self.parameters():
            param.requires_grad = False
        self._validate_freeze_state("ALL frozen")
    
    def freeze_rl_decoder(self):
        """Phase 2 & 3: Freeze RL Decoder ONLY"""
        print("🔒 Freezing: RL Decoder")
        for param in self.decoder_rl.parameters():
            param.requires_grad = False
        self._validate_freeze_state("RL Decoder frozen")
    
    def freeze_refiner(self):
        """Phase 1: Freeze Refiner (not used yet)"""
        print("🔒 Freezing: Refiner")
        if self.use_refiner:
            for param in self.diffusion_refiner.parameters():
                param.requires_grad = False
        self._validate_freeze_state("Refiner frozen")
    
    def unfreeze_encoder_decoder(self):
        """Phase 1: Unfreeze Encoder + RL Decoder"""
        print("🔓 Unfreezing: Encoder + RL Decoder")
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.decoder_rl.parameters():
            param.requires_grad = True
        if self.enable_reconstruction:
            for param in self.reconstruction_head.parameters():
                param.requires_grad = True
        self._validate_freeze_state("Encoder + RL Decoder unfrozen")
    
    def unfreeze_encoder_refiner(self):
        """Phase 2 & 3: Unfreeze Encoder + Refiner"""
        print("🔓 Unfreezing: Encoder + Refiner")
        for param in self.encoder.parameters():
            param.requires_grad = True
        if self.use_refiner:
            for param in self.diffusion_refiner.parameters():
                param.requires_grad = True
        if self.enable_reconstruction:
            for param in self.reconstruction_head.parameters():
                param.requires_grad = True
        self._validate_freeze_state("Encoder + Refiner unfrozen")
    
    def unfreeze_all(self):
        """Unfreeze ALL parameters"""
        print("🔓 Unfreezing: ALL parameters")
        for param in self.parameters():
            param.requires_grad = True
        self._validate_freeze_state("ALL unfrozen")
    
    def _validate_freeze_state(self, expected_state):
        """Validate freeze/unfreeze state"""
        encoder_frozen = all(not p.requires_grad for p in self.encoder.parameters())
        decoder_frozen = all(not p.requires_grad for p in self.decoder_rl.parameters())
        refiner_frozen = all(not p.requires_grad for p in self.diffusion_refiner.parameters()) if self.use_refiner else True
        
        print(f"   ✓ Encoder: {'FROZEN' if encoder_frozen else 'TRAINABLE'}")
        print(f"   ✓ RL Decoder: {'FROZEN' if decoder_frozen else 'TRAINABLE'}")
        if self.use_refiner:
            print(f"   ✓ Refiner: {'FROZEN' if refiner_frozen else 'TRAINABLE'}")
        print(f"   → Expected state: {expected_state}")


# =========================================================================
# ENCODER
# =========================================================================

class MTL_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding_depot = nn.Linear(2, embedding_dim)
        # FIXED: 6 features (x, y, demand, tw_start, tw_end, tw_width)
        self.embedding_node = nn.Linear(6, embedding_dim)
        
        self.slot_attention_module = SlotAttentionModule(**model_params)
        self.slot_to_node_cross_atten = EncoderLayer(**model_params)
        
        self.layers = nn.ModuleList([
            EncoderLayer(**model_params) for _ in range(encoder_layer_num)
        ])

    def forward(self, depot_xy, node_xy_demand_tw):
        embedded_depot = self.embedding_depot(depot_xy)
        embedded_node = self.embedding_node(node_xy_demand_tw)

        H_nodes = torch.cat((embedded_depot, embedded_node), dim=1)
        H_slots = self.slot_attention_module(H_nodes)
        H_reconstructed = self.slot_to_node_cross_atten.forward_cross(H_nodes, H_slots)
        
        out = H_reconstructed
        for layer in self.layers:
            out = layer(out)

        return out


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = FeedForward(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1):
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        if self.model_params['norm_loc'] == "norm_last":
            out_concat = multi_head_attention(q, k, v)
            multi_head_out = self.multi_head_combine(out_concat)
            out1 = self.addAndNormalization1(input1, multi_head_out)
            out2 = self.feedForward(out1)
            out3 = self.addAndNormalization2(out1, out2)
        else:
            out1 = self.addAndNormalization1(None, input1)
            out_concat = multi_head_attention(q, k, v)
            multi_head_out = self.multi_head_combine(out_concat)
            input2 = input1 + multi_head_out
            out2 = self.addAndNormalization2(None, input2)
            out2 = self.feedForward(out2)
            out3 = input2 + out2

        return out3
    
    def forward_cross(self, query_input, kv_input):
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(query_input), head_num=head_num)
        k = reshape_by_heads(self.Wk(kv_input), head_num=head_num)
        v = reshape_by_heads(self.Wv(kv_input), head_num=head_num)

        out_concat = multi_head_attention(q, k, v)
        multi_head_out = self.multi_head_combine(out_concat)

        if self.model_params['norm_loc'] == "norm_last":
            out1 = self.addAndNormalization1(query_input, multi_head_out)
            out2 = self.feedForward(out1)
            out3 = self.addAndNormalization2(out1, out2)
        else:
            out1 = self.addAndNormalization1(None, query_input)
            input2 = out1 + multi_head_out
            out2 = self.addAndNormalization2(None, input2)
            out2 = self.feedForward(out2)
            out3 = input2 + out2

        return out3


# =========================================================================
# DECODER
# =========================================================================

class MTL_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq_last = nn.Linear(embedding_dim + 4, head_num * qkv_dim, bias=False)
        
        self.Wk_nodes = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_nodes = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        
        self.Wk_slots = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_slots = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.slot_gate = nn.Linear(embedding_dim + 4, 1)

        self.k_nodes = None
        self.v_nodes = None
        self.k_slots = None
        self.v_slots = None
        self.single_head_key_nodes = None
        self.slots = None

    def set_kv(self, encoded_nodes, slots=None):
        head_num = self.model_params['head_num']

        self.k_nodes = reshape_by_heads(self.Wk_nodes(encoded_nodes), head_num=head_num)
        self.v_nodes = reshape_by_heads(self.Wv_nodes(encoded_nodes), head_num=head_num)
        self.single_head_key_nodes = encoded_nodes.transpose(1, 2)
        
        if slots is not None:
            self.slots = slots
            self.k_slots = reshape_by_heads(self.Wk_slots(slots), head_num=head_num)
            self.v_slots = reshape_by_heads(self.Wv_slots(slots), head_num=head_num)

    def forward(self, encoded_last_node, attr, ninf_mask):
        head_num = self.model_params['head_num']

        input_cat = torch.cat((encoded_last_node, attr), dim=2)
        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)

        out_concat_nodes = multi_head_attention(q_last, self.k_nodes, self.v_nodes,
                                                rank3_ninf_mask=ninf_mask)

        if self.slots is not None:
            out_concat_slots = multi_head_attention(q_last, self.k_slots, self.v_slots)
            
            gate_logit = self.slot_gate(input_cat)
            gate_weight = torch.sigmoid(gate_logit)
            
            out_concat = gate_weight * out_concat_slots + (1 - gate_weight) * out_concat_nodes
        else:
            out_concat = out_concat_nodes

        mh_atten_out = self.multi_head_combine(out_concat)

        score = torch.matmul(mh_atten_out, self.single_head_key_nodes)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)

        return probs


# =========================================================================
# HELPER MODULES
# =========================================================================

class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.add = True if 'norm_loc' in model_params.keys() and model_params['norm_loc'] == "norm_last" else False
        
        if model_params["norm"] == "batch":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=True)
        elif model_params["norm"] == "batch_no_track":
            self.norm = nn.BatchNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif model_params["norm"] == "instance":
            self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif model_params["norm"] == "layer":
            self.norm = nn.LayerNorm(embedding_dim)
        elif model_params["norm"] == "rezero":
            self.norm = torch.nn.Parameter(torch.Tensor([0.]), requires_grad=True)
        else:
            self.norm = None

    def forward(self, input1=None, input2=None):
        if isinstance(self.norm, nn.InstanceNorm1d):
            added = input1 + input2 if self.add else input2
            transposed = added.transpose(1, 2)
            normalized = self.norm(transposed)
            back_trans = normalized.transpose(1, 2)
        elif isinstance(self.norm, nn.BatchNorm1d):
            added = input1 + input2 if self.add else input2
            batch, problem, embedding = added.size()
            normalized = self.norm(added.reshape(batch * problem, embedding))
            back_trans = normalized.reshape(batch, problem, embedding)
        elif isinstance(self.norm, nn.LayerNorm):
            added = input1 + input2 if self.add else input2
            back_trans = self.norm(added)
        elif isinstance(self.norm, nn.Parameter):
            back_trans = input1 + self.norm * input2 if self.add else self.norm * input2
        else:
            back_trans = input1 + input2 if self.add else input2

        return back_trans


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))


# =========================================================================
# 3-PHASE TRAINING UTILITIES
# =========================================================================

class ThreePhaseTrainer:
    """
    Utility class for 3-Phase Sequential Training
    
    Phase 1: Train Encoder + RL Decoder on random data
    Phase 2: Pre-train Encoder + Refiner with supervised learning (LKH3)
    Phase 3: Fine-tune Encoder + Refiner with RL on random data
    """
    
    def __init__(self, model):
        self.model = model
        self.current_phase = 1
    
    def setup_phase1(self):
        """Setup for Phase 1: RL Decoder Training"""
        print("="*80)
        print("PHASE 1: Training RL Decoder (Encoder + RL Decoder)")
        print("Data: Random on-the-fly")
        print("="*80)
        
        self.model.freeze_refiner()
        self.model.unfreeze_encoder_decoder()
        self.current_phase = 1
    
    def setup_phase2(self):
        """Setup for Phase 2: Supervised Refiner Pre-training"""
        print("="*80)
        print("PHASE 2: Supervised Refiner Pre-training (Encoder + Refiner)")
        print("Data: Static with LKH3 solutions")
        print("="*80)
        
        self.model.freeze_rl_decoder()
        self.model.unfreeze_encoder_refiner()
        self.current_phase = 2
    
    def setup_phase3(self):
        """Setup for Phase 3: RL Refiner Fine-tuning"""
        print("="*80)
        print("PHASE 3: RL Refiner Fine-tuning (Encoder + Refiner)")
        print("Data: Random on-the-fly")
        print("="*80)
        
        # Keep same freeze state as Phase 2
        self.current_phase = 3
    
    @staticmethod
    def create_optimizers(model, lr_phase1=1e-4, lr_phase2=1e-4, lr_phase3=1e-5):
        """Create optimizers for all 3 phases"""
        
        # Phase 1: Encoder + RL Decoder
        encoder_decoder_params = (
            list(model.encoder.parameters()) + 
            list(model.decoder_rl.parameters())
        )
        if model.enable_reconstruction:
            encoder_decoder_params += list(model.reconstruction_head.parameters())
        
        optimizer_phase1 = torch.optim.Adam(encoder_decoder_params, lr=lr_phase1)
        
        # Phase 2 & 3: Encoder + Refiner
        if model.use_refiner:
            encoder_refiner_params = list(model.encoder.parameters())
            encoder_refiner_params += list(model.diffusion_refiner.parameters())
            if model.enable_reconstruction:
                encoder_refiner_params += list(model.reconstruction_head.parameters())
            
            optimizer_phase2 = torch.optim.Adam(encoder_refiner_params, lr=lr_phase2)
            optimizer_phase3 = torch.optim.Adam(encoder_refiner_params, lr=lr_phase3)
        else:
            optimizer_phase2 = None
            optimizer_phase3 = None
        
        return optimizer_phase1, optimizer_phase2, optimizer_phase3


# =========================================================================
# USAGE EXAMPLE (Fixed Version with Correct Phase 3)
# =========================================================================

"""
COMPLETE 3-PHASE TRAINING EXAMPLE (FIXED)

# =========================================================================
# INITIALIZATION
# =========================================================================

import torch
from torch.utils.data import DataLoader

# Model configuration
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
    'enable_slot_reconstruction': True,
}

model = MTLModel(**model_params).cuda()
trainer = ThreePhaseTrainer(model)

# Create optimizers
opt1, opt2, opt3 = ThreePhaseTrainer.create_optimizers(
    model, 
    lr_phase1=1e-4, 
    lr_phase2=1e-4, 
    lr_phase3=1e-5
)

# =========================================================================
# PHASE 1: TRAIN RL DECODER (Epochs 1-3000)
# =========================================================================

trainer.setup_phase1()

for epoch in range(1, 3001):
    model.train()
    
    for batch_idx, batch in enumerate(random_train_loader):
        opt1.zero_grad()
        
        # Encode problem
        model.pre_forward(batch)
        
        # RL rollout
        state = batch.clone()
        trajectories = model._forward_rollout(state)
        
        # REINFORCE loss
        log_probs = torch.log(trajectories['probs'] + 1e-10).sum(dim=2)
        costs = model.rollout_manager.compute_solution_cost(state, trajectories)
        
        # Baseline: average cost over POMO
        baseline = costs.mean(dim=1, keepdim=True).detach()
        advantages = costs - baseline
        rl_loss = (advantages * log_probs).mean()
        
        # Auxiliary losses (optional)
        recon_loss = model.compute_slot_reconstruction_loss(batch)
        contrastive_loss = model.compute_slot_contrastive_loss()
        
        total_loss = rl_loss + 0.1 * recon_loss + 0.01 * contrastive_loss
        
        # Backprop
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(opt1.param_groups[0]['params'], max_norm=1.0)
        opt1.step()
        
        if batch_idx % 100 == 0:
            print(f"[Phase 1] Epoch {epoch}, Batch {batch_idx}: "
                  f"RL={rl_loss.item():.4f}, "
                  f"Cost={costs.mean().item():.2f}")
    
    # Validation
    if epoch % 100 == 0:
        val_cost = validate(model, val_loader)
        print(f"[Phase 1] Epoch {epoch}: Val Cost = {val_cost:.2f}")
        torch.save(model.state_dict(), f'checkpoints/phase1_epoch{epoch}.pt')

print("✅ Phase 1 completed!")
torch.save(model.state_dict(), 'checkpoints/phase1_final.pt')

# =========================================================================
# PHASE 2: SUPERVISED REFINER PRE-TRAINING (Epochs 3001-4000)
# =========================================================================

trainer.setup_phase2()

# Load static dataset with LKH3 solutions
static_dataset = load_static_cvrptw_dataset()  # Your custom loader
lkh3_solutions = load_lkh3_solutions()          # Pre-computed optimal tours

static_loader = DataLoader(
    list(zip(static_dataset, lkh3_solutions)),
    batch_size=64,
    shuffle=True
)

for epoch in range(3001, 4001):
    model.train()
    
    for batch_idx, (batch, optimal_tours) in enumerate(static_loader):
        opt2.zero_grad()
        
        # Encode problem
        model.pre_forward(batch)
        
        # Convert optimal tour to heatmap
        num_nodes = model.encoded_nodes.size(1)
        optimal_heatmap = tour_to_heatmap(optimal_tours, num_nodes).cuda()
        
        # Supervised loss
        refiner_loss = model.compute_refinement_loss_supervised(optimal_heatmap)
        
        # Auxiliary losses
        recon_loss = model.compute_slot_reconstruction_loss(batch)
        
        total_loss = refiner_loss + 0.1 * recon_loss
        
        # Backprop
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(opt2.param_groups[0]['params'], max_norm=1.0)
        opt2.step()
        
        if batch_idx % 100 == 0:
            print(f"[Phase 2] Epoch {epoch}, Batch {batch_idx}: "
                  f"Refiner SL={refiner_loss.item():.4f}")
    
    # Validation
    if epoch % 50 == 0:
        val_cost = validate_with_refiner(model, val_loader)
        print(f"[Phase 2] Epoch {epoch}: Val Cost (refined) = {val_cost:.2f}")
        torch.save(model.state_dict(), f'checkpoints/phase2_epoch{epoch}.pt')

print("✅ Phase 2 completed!")
torch.save(model.state_dict(), 'checkpoints/phase2_final.pt')

# =========================================================================
# PHASE 3: RL REFINER FINE-TUNING (Epochs 4001-5000) - FIXED!
# =========================================================================

trainer.setup_phase3()

baseline_ema = None
ema_alpha = 0.95

for epoch in range(4001, 5001):
    model.train()
    
    for batch_idx, batch in enumerate(random_train_loader):
        opt3.zero_grad()
        
        # Encode problem
        model.pre_forward(batch)
        
        # RL Refiner loss (NOW FIXED - properly uses refined_heatmap!)
        state = batch.clone()
        refiner_rl_loss, info = model.compute_refinement_loss_rl(
            state, 
            baseline=baseline_ema
        )
        
        # Update EMA baseline
        if baseline_ema is None:
            baseline_ema = info['cost_initial']
        else:
            baseline_ema = ema_alpha * baseline_ema + (1 - ema_alpha) * info['cost_initial']
        
        # Auxiliary losses
        recon_loss = model.compute_slot_reconstruction_loss(batch)
        
        total_loss = refiner_rl_loss + 0.1 * recon_loss
        
        # Backprop
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(opt3.param_groups[0]['params'], max_norm=1.0)
        opt3.step()
        
        if batch_idx % 100 == 0:
            print(f"[Phase 3] Epoch {epoch}, Batch {batch_idx}: "
                  f"Refiner RL={refiner_rl_loss.item():.4f}, "
                  f"Improvement={info['improvement']:.2f}, "
                  f"Initial={info['cost_initial']:.2f}, "
                  f"Refined={info['cost_refined']:.2f}")
    
    # Validation
    if epoch % 50 == 0:
        val_cost = validate_with_refiner(model, val_loader)
        improvement = (baseline_ema - val_cost) / baseline_ema * 100
        print(f"[Phase 3] Epoch {epoch}: Val Cost = {val_cost:.2f}, "
              f"Improvement = {improvement:.1f}%")
        torch.save(model.state_dict(), f'checkpoints/phase3_epoch{epoch}.pt')

print("✅ Phase 3 completed!")
torch.save(model.state_dict(), 'checkpoints/final_3phase_model.pt')

# =========================================================================
# FINAL EVALUATION
# =========================================================================

model.eval()
with torch.no_grad():
    # Test on large instances
    test_results = []
    for batch in test_loader:
        model.pre_forward(batch)
        
        # Generate solution with refinement
        result = model._forward_with_refinement(batch)
        
        initial_cost = model.rollout_manager.compute_solution_cost(
            batch, result['initial_trajectories']
        ).mean()
        
        # Decode refined solution
        refined_actions = result['refined_actions']
        refined_cost = compute_cost_from_actions(refined_actions, batch)
        
        improvement = (initial_cost - refined_cost) / initial_cost * 100
        test_results.append({
            'initial': initial_cost.item(),
            'refined': refined_cost.item(),
            'improvement': improvement.item()
        })
    
    avg_improvement = sum(r['improvement'] for r in test_results) / len(test_results)
    print(f"\\n{'='*80}")
    print(f"FINAL TEST RESULTS:")
    print(f"Average Improvement: {avg_improvement:.2f}%")
    print(f"Average Initial Cost: {sum(r['initial'] for r in test_results)/len(test_results):.2f}")
    print(f"Average Refined Cost: {sum(r['refined'] for r in test_results)/len(test_results):.2f}")
    print(f"{'='*80}")
"""