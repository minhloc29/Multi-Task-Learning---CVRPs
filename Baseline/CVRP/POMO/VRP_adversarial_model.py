from VRPModel import VRPModel
import torch.nn as nn

class TaskDiscriminator(nn.Module):
    """
    Adversarial discriminator predicting VRP type.
    Input: pooled encoder embedding [B, E]
    Output: logits [B, num_tasks]
    """
    def __init__(self, emb_dim, num_tasks):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, num_tasks)
        )

    def forward(self, encoded_nodes):
        # encoded_nodes: [B, N, E]
        pooled = encoded_nodes.mean(dim=1)        # [B, E]
        return self.net(pooled)   

class VRPModel_AMTL(VRPModel):
    def __init__(self, num_tasks, **model_params):
        super().__init__(**model_params)
        emb_dim = model_params['embedding_dim']
        self.discriminator = TaskDiscriminator(emb_dim, num_tasks)

    def forward(self, state, task_labels=None, adversarial=False):
        """
        Forward pass with adversarial branch.
        Returns route prediction + discriminator logits.
        """
        selected, prob = super().forward(state)
        # self.encoded_nodes: [B, N, E]

        disc_logits = self.discriminator(self.encoded_nodes)  # [B, num_tasks]

        # detach if training discriminator only (so encoder frozen)
        if adversarial:
            disc_logits = self.discriminator(self.encoded_nodes.detach())

        return selected, prob, disc_logits