import torch
from torch import nn
import torch.nn.functional as F


class GatedExpert(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.g = nn.Linear(d, ff, bias=False)
        self.u = nn.Linear(d, ff, bias=False)
        self.v = nn.Linear(ff, d, bias=False)

    def forward(self, h):
        return self.v(F.gelu(self.g(h)) * self.u(h))


class SparseExperts(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(
            cfg.hidden_dim + 2 * len(cfg.modalities), cfg.num_experts
        )
        self.experts = nn.ModuleList(
            GatedExpert(cfg.hidden_dim, cfg.expert_dim) for _ in range(cfg.num_experts)
        )

    def forward(self, h, rho, u):
        cfg = self.cfg
        if not cfg.expert_reliability:
            rho, u = torch.zeros_like(rho), torch.zeros_like(u)
        xi = self.router(torch.cat([h, rho, u], dim=-1))
        probabilities = torch.softmax(xi / cfg.tau_e, dim=-1)
        if self.training and cfg.expert_training == "gumbel_st":
            random = torch.rand_like(xi).clamp(1e-6, 1 - 1e-6)
            perturbed = xi - torch.log(-torch.log(random))
            chosen = perturbed.detach().topk(cfg.ke, dim=-1).indices
            hard = torch.zeros_like(xi).scatter(1, chosen, 1)
            soft = torch.softmax(perturbed / cfg.tau_g, dim=-1)
            mask = (hard - soft).detach() + soft
            weighted = mask * probabilities
            gates = weighted / weighted.sum(-1, keepdim=True).clamp_min(cfg.epsilon)
            # Exact ST surrogate needs inactive expert outputs in backward.
            # This branch evaluates ALL experts in training; inference remains sparse.
            expert_values = torch.stack([expert(h) for expert in self.experts], dim=1)
            output = (expert_values * gates[:, :, None]).sum(1)
            evaluated = cfg.num_experts * len(h)
        else:
            chosen = xi.detach().topk(cfg.ke, dim=-1).indices
            hard = torch.zeros_like(xi).scatter(1, chosen, 1)
            gates = probabilities * hard
            gates = gates / gates.sum(-1, keepdim=True).clamp_min(cfg.epsilon)
            output = torch.zeros_like(h)
            evaluated = 0
            for e, expert in enumerate(self.experts):
                indices = hard[:, e].nonzero(as_tuple=True)[0]
                if len(indices):
                    output = output.index_add(
                        0, indices, expert(h[indices]) * gates[indices, e, None]
                    )
                    evaluated += len(indices)
        raw_load = hard.mean(0)
        normalized_load = raw_load / cfg.ke
        raw_loss = cfg.num_experts * (probabilities.mean(0) * raw_load).sum()
        normalized_loss = raw_loss / cfg.ke
        return {
            "z": output,
            "expert_prob": probabilities,
            "expert_gate": gates,
            "expert_selected": hard,
            "expert_load_raw": raw_load,
            "expert_load_normalized": normalized_load,
            "balance_raw": raw_loss,
            "balance_normalized": normalized_loss,
            "balance": raw_loss if cfg.balance_convention == "raw" else normalized_loss,
            "expert_evaluations": evaluated,
        }
