
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

########################################
# GRADIENT REVERSAL LAYER (GRL)
########################################

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Trả về -lambda * grad_output, None (cho lambda_)
        return -ctx.lambda_ * grad_output, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

########################################
# TASK DISCRIMINATOR (Cải thiện)
########################################

class TaskDiscriminator(nn.Module):
    def __init__(self, embedding_dim, hidden=128, num_tasks=5, lambda_=1.0, dropout=0.3):
        super().__init__()
        # GRL sẽ được áp dụng nếu reverse=True
        self.grl = GradientReversal(lambda_) 
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),  # ✅ Thêm dropout để tránh overfit
            nn.Linear(hidden, num_tasks)
        )

    def forward(self, enc, reverse=False):
        # Lấy trung bình embedding của tất cả các node (Depot + Customers)
        pooled = enc.mean(dim=1)  # [batch, embedding]
        if reverse:
            # Chỉ áp dụng GRL khi huấn luyện Encoder (Adversarial)
            pooled = self.grl(pooled)
        return self.net(pooled)  # [batch, num_tasks]

########################################
# MAIN MODEL
########################################

class VRPModel(nn.Module):

    def __init__(self, num_tasks=5, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = VRP_Encoder(**model_params)
        self.decoder = VRP_Decoder(**model_params)
        self.discriminator = TaskDiscriminator(
            embedding_dim=model_params['embedding_dim'],
            num_tasks=num_tasks,
            dropout=model_params.get('discriminator_dropout', 0.3)  # ✅ Có thể config từ ngoài
        )
        # Các biến trạng thái để lưu trữ encoded nodes
        self.encoded_nodes = None
        # shape: (batch, problem+1, EMBEDDING_DIM)

    def pre_forward(self, reset_state): # get from VRPEnv.py
        depot_xy = reset_state.depot_xy
        node_xy = reset_state.node_xy
        node_demand = reset_state.node_demand
        node_earlyTW = reset_state.node_earlyTW
        node_lateTW = reset_state.node_lateTW

        # Chuẩn bị input features cho Encoder: (x, y, demand, earlyTW, lateTW)
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
        node_TW = torch.cat((node_earlyTW[:, :, None],node_lateTW[:, :, None]),dim=2)
        node_xy_demand_TW = torch.cat((node_xy_demand,node_TW),dim=2)
        # shape: (batch, problem, 5)

        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand_TW)
        # shape: (batch, problem+1, embedding)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state, task_labels=None, adversarial=False):
        """
        Forward pass cho VRP model.
        """

        # --- Xử lý trạng thái chưa khởi tạo (bước đầu tiên) ---
        if state.BATCH_IDX is None or state.POMO_IDX is None:
            batch_size = self.encoded_nodes.size(0)
            pomo_size = self.encoded_nodes.size(1) - 1  # exclude depot
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size))
            return selected, prob

        # --- Trạng thái bình thường (từ bước thứ 2 trở đi) ---
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.selected_count == 0:
            # Bước 1: Luôn chọn Depot (node 0)
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size))

        elif state.selected_count == 1:
            # Bước 2: Bắt đầu từ các node 1...Pomo_Size
            selected = torch.arange(start=1, end=pomo_size + 1)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))

        else:
            # --- Lấy embedding của node cuối cùng đã chọn ---
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)

            # --- Decoder tính xác suất ---
            probs = self.decoder(
                encoded_last_node,
                state.load,
                state.time,
                state.length,
                state.route_open,
                ninf_mask=state.ninf_mask
            )
            # shape: (batch, pomo, problem+1)

            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:  # Tránh lỗi zero-probability của multinomial
                    with torch.no_grad():
                      # đảm bảo probs đã được xử lý bên trong decoder (softmax, nan->0, renorm)
                      probs_flat = probs.reshape(batch_size * pomo_size, -1)
                      # clamp + renormalize again to be extra-safe
                      probs_flat = probs_flat.clamp(min=1e-12)
                      probs_flat = probs_flat / probs_flat.sum(dim=1, keepdim=True)

                      selected_flat = probs_flat.multinomial(1).squeeze(dim=1)
                      selected = selected_flat.reshape(batch_size, pomo_size)

                    # Lấy xác suất của node đã chọn (an toàn)
                    prob = probs.gather(dim=2, index=selected.unsqueeze(2)).squeeze(2)
                    # nếu vẫn có phần tử 0, thay bằng epsilon nhỏ
                    prob = prob.clamp(min=1e-12)
                    if (prob != 0).all():
                        break
            else:
                # Đánh giá: Chọn node có xác suất cao nhất
                selected = probs.argmax(dim=2)
                prob = None  # Không cần xác suất

        return selected, prob

def _get_encoding(encoded_nodes, node_index_to_pick):
    # Dùng gather để lấy embedding của node dựa trên chỉ số
    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# TRANSFORMER ENCODER (Mới)
########################################

class MultiHeadAttention(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        head_num = model_params['head_num']
        qkv_dim = model_params['qkv_dim']
        
        # QKV Projection (Generic cho Encoder)
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

    def forward(self, q, k, v, mask=None):
        # q, k, v shape: (batch, N, embedding)
        
        head_num = self.Wq.out_features // self.Wq.in_features
        
        q_reshaped = reshape_by_heads(self.Wq(q), head_num=head_num)
        k_reshaped = reshape_by_heads(self.Wk(k), head_num=head_num)
        v_reshaped = reshape_by_heads(self.Wv(v), head_num=head_num)

        # Transformer Attention (Fully Connected)
        out_concat = multi_head_attention(
            q_reshaped, k_reshaped, v_reshaped, rank2_ninf_mask=mask
        )
        
        return self.multi_head_combine(out_concat) # (batch, N, embedding)


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        
        # 1. Multi-Head Attention (MHA)
        self.mha = MultiHeadAttention(**model_params)
        self.norm1 = AddAndBatchNormalization(**model_params)
        
        # 2. Feed-Forward Network (FFN)
        self.ffn = FeedForward(**model_params)
        self.norm2 = AddAndBatchNormalization(**model_params)

    def forward(self, x):
        # x.shape: (batch, N, embedding)
        
        # MHA Block: MHA + Residual + Norm
        mha_out = self.mha(x, x, x, mask=None)
        x = self.norm1(x, mha_out)
        
        # FFN Block: FFN + Residual + Norm
        ffn_out = self.ffn(x)
        x = self.norm2(x, ffn_out)
        
        return x # (batch, N, embedding)


class VRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params['embedding_dim']
        encoder_layer_num = model_params['encoder_layer_num']

        # Node feature embeddings
        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(5, embedding_dim)

        # Stack multiple Transformer Encoder Layers (Mới)
        self.layers = nn.ModuleList([
            EncoderLayer(**model_params)
            for _ in range(encoder_layer_num)
        ])

    def forward(self, depot_xy, node_xy_demand_TW):
        """
        depot_xy: (batch, 1, 2)
        node_xy_demand_TW: (batch, problem, 5)
        """
        # Embed features
        depot_emb = self.embedding_depot(depot_xy)
        node_emb = self.embedding_node(node_xy_demand_TW)
        x = torch.cat((depot_emb, node_emb), dim=1)  # (batch, problem+1, emb)

        # Pass through Transformer layers
        for layer in self.layers:
            x = layer(x) # Không cần adj_mask nữa

        return x  # (batch, problem+1, embedding)


########################################
# DECODER (Giữ nguyên)
########################################

class VRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        # Query: embedding + 4 trạng thái (load, time, length, route_open)
        self.Wq_last = nn.Linear(embedding_dim+4, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, problem+1, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+1)

    def forward(self, encoded_last_node, load, time, length, route_open, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)

        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        input_cat = torch.cat((encoded_last_node, load[:, :, None], time[:, :, None], length[:, :, None], route_open[:, :, None]), dim=2)
        # shape = (batch, pomo, EMBEDDING_DIM+4)

        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        # q_last shape: (batch, head_num, pomo, qkv_dim)

        q = q_last
        # shape: (batch, head_num, pomo, qkv_dim)
        
        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem+1)

        # safety guards: prevent division by zero / nan
        sqrt_embedding_dim = float(self.model_params.get('sqrt_embedding_dim', 1.0))
        if sqrt_embedding_dim == 0 or torch.isnan(torch.tensor(sqrt_embedding_dim)):
            sqrt_embedding_dim = 1.0

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem+1)

        score_clipped = float(self.model_params.get('logit_clipping', 10.0)) * torch.tanh(score_scaled)

        # add mask (ninf_mask may contain -inf or large negative numbers)
        score_masked = score_clipped + ninf_mask

        # Replace any NaN/inf introduced by mask operations
        # use a large negative for -inf-like values so softmax ~ 0 there
        score_masked = torch.nan_to_num(score_masked, nan=-1e9, posinf=1e9, neginf=-1e9)

        # numerical stable softmax: clamp, softmax, then renormalize to avoid all-zero rows
        # small eps to prevent zero-prob rows
        EPS = 1e-12
        # compute softmax
        probs = F.softmax(score_masked, dim=2)
        # replace possible nan/inf after softmax (shouldn't happen, but safe)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

        # ensure non-negative and renormalize (row sum > 0)
        probs = probs.clamp(min=0.0)
        row_sums = probs.sum(dim=2, keepdim=True)
        # if some rows sum to zero (all masked), give tiny uniform mass to avoid multinomial error
        zero_mask = (row_sums <= 0.0)
        if zero_mask.any():
            # create tiny uniform mass for those rows
            uniform = torch.full_like(probs, EPS)
            probs = torch.where(zero_mask.expand_as(probs), uniform, probs)
            row_sums = probs.sum(dim=2, keepdim=True)

        probs = probs / (row_sums + EPS)

        return probs



########################################
# NN SUB CLASS / FUNCTIONS (Giữ nguyên)
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem+1)

    # avoid zero division / nan in sqrt
    key_dim_val = float(key_dim) if isinstance(key_dim, int) else float(key_dim.item())
    if key_dim_val <= 0:
        key_dim_val = 1.0

    score_scaled = score / math.sqrt(key_dim_val)

    # apply masks if present (they may contain -inf)
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    # sanitize possible inf/nan in score before softmax
    score_scaled = torch.nan_to_num(score_scaled, nan=-1e9, posinf=1e9, neginf=-1e9)

    # stable softmax on last dim
    weights = nn.Softmax(dim=3)(score_scaled)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = weights.clamp(min=0.0)

    # ensure rows sum > 0
    row_sum = weights.sum(dim=3, keepdim=True)
    EPS = 1e-12
    zero_rows = (row_sum <= 0.0)
    if zero_rows.any():
        # distribute tiny uniform prob where necessary
        tiny = torch.full_like(weights, EPS)
        weights = torch.where(zero_rows.expand_as(weights), tiny, weights)
        row_sum = weights.sum(dim=3, keepdim=True)

    weights = weights / (row_sum + EPS)

    out = torch.matmul(weights, v)
    out_transposed = out.transpose(1, 2)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)

    return out_concat


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class AddAndBatchNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        # Normalized by EMB dim (đã sửa BatchNorm1d để khớp với logic ban đầu của bạn)
        self.norm_by_EMB = nn.BatchNorm1d(embedding_dim, affine=True) 

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        batch_s = input1.size(0)
        problem_s = input1.size(1)
        embedding_dim = input1.size(2)

        added = input1 + input2
        # Cần reshape để BatchNorm1d hoạt động trên feature dim (embedding_dim)
        normalized = self.norm_by_EMB(added.reshape(batch_s * problem_s, embedding_dim))
        back_trans = normalized.reshape(batch_s, problem_s, embedding_dim)

        return back_trans

class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))
