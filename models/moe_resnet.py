"""
ResNet-20 + Mixture of Experts (MoE) head for CIFAR-100.

Same backbone as ResNet-20, but replaces the single FC classifier with
a gated mixture of expert MLPs. Uses top-k routing and an auxiliary
load-balancing loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from models.resnet import ResNet


class ExpertMLP(nn.Module):
    """A single expert: 2-layer MLP."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class MoELayer(nn.Module):
    """
    Mixture of Experts layer with top-k gating.

    Args:
        input_dim: feature dimension from backbone
        num_classes: number of output classes
        num_experts: number of expert networks
        top_k: number of experts to route each input to
        hidden_dim: hidden dimension inside each expert MLP
    """

    def __init__(self, input_dim=64, num_classes=100, num_experts=4,
                 top_k=2, hidden_dim=128):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Expert networks
        self.experts = nn.ModuleList([
            ExpertMLP(input_dim, hidden_dim, num_classes)
            for _ in range(num_experts)
        ])

        # Gating network
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim) feature vectors

        Returns:
            output: (batch_size, num_classes) weighted expert predictions
            aux_loss: scalar load-balancing loss
            gate_scores: (batch_size, num_experts) for analysis
        """
        batch_size = x.size(0)

        # Compute gate scores
        gate_logits = self.gate(x)                          # (B, E)
        gate_scores = F.softmax(gate_logits, dim=-1)        # (B, E)

        # Top-k selection
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)  # (B, K)

        # Normalize top-k scores to sum to 1
        top_k_scores = top_k_scores / (top_k_scores.sum(dim=-1, keepdim=True) + 1e-8)

        # Compute expert outputs for selected experts
        # For efficiency with small number of experts, compute all and select
        all_expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )  # (B, E, num_classes)

        # Gather top-k expert outputs
        top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, all_expert_outputs.size(-1))
        selected_outputs = torch.gather(all_expert_outputs, 1, top_k_indices_expanded)  # (B, K, C)

        # Weighted sum
        output = (top_k_scores.unsqueeze(-1) * selected_outputs).sum(dim=1)  # (B, C)

        # ─── Auxiliary load-balancing loss ─────────────────────────────────
        # Encourage each expert to receive roughly equal traffic
        # f_i = fraction of tokens routed to expert i
        # P_i = mean gate probability for expert i
        # loss = num_experts * sum(f_i * P_i)
        expert_mask = torch.zeros(batch_size, self.num_experts, device=x.device)
        expert_mask.scatter_(1, top_k_indices, 1.0)

        f = expert_mask.mean(dim=0)              # fraction routed to each expert
        P = gate_scores.mean(dim=0)              # mean gate probability
        aux_loss = self.num_experts * (f * P).sum()

        return output, aux_loss, gate_scores


class MoEResNet(nn.Module):
    """ResNet backbone + MoE classification head."""

    def __init__(self, num_classes=100, num_blocks=3, num_experts=4,
                 top_k=2, expert_hidden_dim=128):
        super().__init__()

        # Backbone (reuse ResNet without its FC layer)
        self.backbone = ResNet(num_classes=num_classes, num_blocks=num_blocks)
        # Remove the original FC — we'll use MoE instead
        self.backbone.fc = nn.Identity()

        # MoE head
        self.moe = MoELayer(
            input_dim=64,
            num_classes=num_classes,
            num_experts=num_experts,
            top_k=top_k,
            hidden_dim=expert_hidden_dim,
        )

    def forward(self, x):
        features = self.backbone.extract_features(x)  # (B, 64)
        output, aux_loss, gate_scores = self.moe(features)
        return output, aux_loss, gate_scores


def MoEResNet20(num_classes=100):
    return MoEResNet(
        num_classes=num_classes,
        num_blocks=3,
        num_experts=config.NUM_EXPERTS,
        top_k=config.TOP_K,
        expert_hidden_dim=config.EXPERT_HIDDEN_DIM,
    )
