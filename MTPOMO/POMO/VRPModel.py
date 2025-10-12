
import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class TaskDiscriminator(nn.Module):
    def __init__(self, embedding_dim, hidden=128, num_tasks=5, lambda_=1.0):
        super().__init__()
        self.grl = GradientReversal(lambda_)
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_tasks)
        )

    def forward(self, enc, reverse=False):
        pooled = enc.mean(dim=1)  # [batch, embedding]
        if reverse:
            pooled = self.grl(pooled)
        return self.net(pooled)  # [batch, num_tasks]
class VRPModel(nn.Module):

    def __init__(self, num_tasks=5, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = VRP_Encoder(**model_params)
        self.decoder = VRP_Decoder(**model_params)
        self.discriminator = TaskDiscriminator(
            embedding_dim=model_params['embedding_dim'],
            num_tasks=num_tasks
        )


        self.encoded_nodes = None
        # shape: (batch, problem+1, EMBEDDING_DIM)

    def pre_forward(self, reset_state): # get from VRPEnv.py
        depot_xy = reset_state.depot_xy
        # shape: (batch, 1, 2)
        node_xy = reset_state.node_xy
        # shape: (batch, problem, 2)
        node_demand = reset_state.node_demand
        # shape: (batch, problem)
        node_earlyTW = reset_state.node_earlyTW
        # shape: (batch, problem)
        node_lateTW = reset_state.node_lateTW
        # shape: (batch, problem)
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
        # shape: (batch, problem, 3)
        node_TW = torch.cat((node_earlyTW[:, :, None],node_lateTW[:, :, None]),dim=2)
        # shape: (batch, problem, 2)
        node_xy_demand_TW = torch.cat((node_xy_demand,node_TW),dim=2)
        # shape: (batch, problem, 5)

        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand_TW)
        # shape: (batch, problem+1, embedding)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state, task_labels=None, adversarial=False):
        """
        Forward pass for VRP model with adversarial task discriminator.
        Handles initialization when state indices are None (first step).
        """

        # --- Handle uninitialized state (first move) ---
        if state.BATCH_IDX is None or state.POMO_IDX is None:
            batch_size = self.encoded_nodes.size(0)
            pomo_size = self.encoded_nodes.size(1) - 1  # exclude depot
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size))
            return selected, prob

        # --- Normal state ---
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.selected_count == 0:
            # First move: start from depot
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size))

        elif state.selected_count == 1:
            # Second move: assign unique starting nodes
            selected = torch.arange(start=1, end=pomo_size + 1)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))

        else:
            # --- Get embeddings of last selected nodes ---
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)

            # --- Decoder computes route probabilities ---
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
                while True:  # Avoid PyTorch multinomial zero-probability issue
                    with torch.no_grad():
                        selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, pomo_size)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    if (prob != 0).all():
                        break
            else:
                selected = probs.argmax(dim=2)
                prob = None  # Not needed for evaluation

        # --- Adversarial branch (if used) ---
        #if task_labels is not None:
            # Reverse gradient if adversarial
            #if adversarial:
                #enc_for_disc = self.encoded_nodes.detach()  # freeze encoder for discriminator training
            ##enc_for_disc = self.encoded_nodes  # allow gradient to flow for encoder training

            # Pass shared embeddings to task discriminator
            #task_logits = self.discriminator(enc_for_disc)
        #else:
           # task_logits = None

        return selected, prob

def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class VRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params['embedding_dim']
        encoder_layer_num = model_params['encoder_layer_num']
        heads = model_params.get('head_num', 4)
        dropout = model_params.get('dropout', 0.1)

        # Node feature embeddings
        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(5, embedding_dim)

        # Stack multiple GAT layers
        self.layers = nn.ModuleList([
            GraphAttentionLayer(
                in_dim=embedding_dim,
                out_dim=embedding_dim,
                heads=heads,
                dropout=dropout
            )
            for _ in range(encoder_layer_num)
        ])

    def forward(self, depot_xy, node_xy_demand_TW, adj_mask=None):
        """
        depot_xy: (batch, 1, 2)
        node_xy_demand_TW: (batch, problem, 5)
        adj_mask: (batch, problem+1, problem+1), bool tensor — adjacency (True if connected)
        """
        # Embed features
        depot_emb = self.embedding_depot(depot_xy)
        node_emb = self.embedding_node(node_xy_demand_TW)
        x = torch.cat((depot_emb, node_emb), dim=1)  # (batch, problem+1, emb)

        # Build a fully connected adjacency if none provided
        if adj_mask is None:
            N = x.size(1)
            adj_mask = torch.ones(x.size(0), N, N, dtype=torch.bool, device=x.device)

        # Pass through GAT layers
        for layer in self.layers:
            x = layer(x, adj_mask)

        return x  # (batch, problem+1, embedding)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1, leaky=0.2):
        super().__init__()
        self.heads = heads
        self.head_dim = out_dim // heads
        self.concat = True  # concatenate heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Parameter(torch.Tensor(1, heads, 2 * self.head_dim))
        self.leakyrelu = nn.LeakyReLU(leaky)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

        nn.init.xavier_uniform_(self.attn)

    def forward(self, x, adj_mask):
        """
        x: (batch, N, in_dim)
        adj_mask: (batch, N, N), bool adjacency matrix (True = neighbor)
        """
        B, N, _ = x.shape
        H, hd = self.heads, self.head_dim

        # Project node features to multi-head subspaces
        xW = self.W(x).view(B, N, H, hd)  # (B, N, H, hd)

        # Compute pairwise attention scores
        xi = xW.unsqueeze(2).expand(B, N, N, H, hd)
        xj = xW.unsqueeze(1).expand(B, N, N, H, hd)
        xcat = torch.cat([xi, xj], dim=-1)  # (B, N, N, H, 2*hd)

        # Attention coefficients: a^T [x_i || x_j]
        e = (xcat * self.attn).sum(dim=-1)  # (B, N, N, H)
        e = self.leakyrelu(e)

        # Mask non-edges
        e = e.masked_fill(~adj_mask.unsqueeze(-1), float('-inf'))

        # Softmax over neighbors
        alpha = torch.softmax(e, dim=2)  # (B, N, N, H)
        alpha = self.dropout(alpha)

        # Aggregate messages
        xj = xW.unsqueeze(1).expand(B, N, N, H, hd)
        out = (alpha.unsqueeze(-1) * xj).sum(dim=2)  # (B, N, H, hd)
        out = out.reshape(B, N, H * hd)

        # Residual connection + normalization
        out = self.norm(out + self.W(x))

        return F.elu(out)


########################################
# DECODER
########################################

class VRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        # self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # self.Wq_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim+4, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        # self.q1 = None  # saved q1, for multi-head attention
        # self.q2 = None  # saved q2, for multi-head attention

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, problem+1, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+1)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q1 = reshape_by_heads(self.Wq_1(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def set_q2(self, encoded_q2):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q2 = reshape_by_heads(self.Wq_2(encoded_q2), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, load, time,length, route_open, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # load.shape: (batch, pomo)
        # ninf_mask.shape: (batch, pomo, problem)

        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        input_cat = torch.cat((encoded_last_node, load[:, :, None], time[:, :, None],length[:, :, None], route_open[:, :, None]), dim=2)
        # shape = (batch, group, EMBEDDING_DIM+3)

        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        # q_last shape: (batch, head_num, pomo, qkv_dim)

        # q = self.q1 + self.q2 + q_last
        # # shape: (batch, head_num, pomo, qkv_dim)
        q = q_last
        # shape: (batch, head_num, pomo, qkv_dim)
        #print("ninf_mask",ninf_mask)
        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)
        #print("score",score)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)
        #print("score_scaled",score_scaled)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs


########################################
# NN SUB CLASS / FUNCTIONS
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
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

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
        self.norm_by_EMB = nn.BatchNorm1d(embedding_dim, affine=True)
        # 'Funny' Batch_Norm, as it will normalized by EMB dim

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        batch_s = input1.size(0)
        problem_s = input1.size(1)
        embedding_dim = input1.size(2)

        added = input1 + input2
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
