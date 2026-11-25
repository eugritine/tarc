"""Model configuration and explicit validation for TARC."""

from dataclasses import asdict, dataclass, field
from typing import Dict, List
import copy
import json
import yaml


@dataclass
class ModelConfig:
    modalities: List[str] = field(default_factory=lambda: ["text", "audio", "vision"])
    input_dims: Dict[str, int] = field(default_factory=dict)
    task: str = "sentiment"
    num_classes: int = 4
    hidden_dim: int = 128
    expert_dim: int = 256
    scales: List[int] = field(default_factory=lambda: [1, 2, 4])
    ki_factor: float = 4.0
    sample_factor: float = 4.0
    kc: int = 16
    num_experts: int = 4
    ke: int = 2
    tau_u: float = 0.5
    tau_c: float = 1.0
    tau_e: float = 1.0
    tau_d: float = 2.0
    tau_t: float = 0.2
    tau_g: float = 1.0
    lambda_r: float = 1.0
    lambda_kappa: float = 1.0
    lambda_time: float = 0.1
    conv_kernel: int = 3
    cross_mode: str = "sparse"
    candidate_per_source: int = 32
    expert_training: str = "gumbel_st"
    missing_policy: str = "finite_prior"
    balance_convention: str = "raw"
    epsilon: float = 1e-8
    logvar_min: float = -8.0
    logvar_max: float = 8.0
    sparse_temporal: bool = True
    use_uncertainty: bool = True
    use_moe: bool = True
    expert_residual: bool = False
    text_message_pooling: bool = False
    sign_readout_pooling: str = "legacy"
    expert_reliability: bool = True
    separate_rho: bool = False
    contextual_reliability: bool = False
    reliability_context_content: bool = True
    reliability_context_hidden: int = 64
    reliability_context_bound: float = 2.0

    def validate(self):
        if self.task not in {"classification", "regression", "sentiment"}:
            raise ValueError("task must be classification, regression, or sentiment")
        if not self.modalities or len(set(self.modalities)) != len(self.modalities):
            raise ValueError("modalities must be unique and nonempty")
        if set(self.modalities) != set(self.input_dims) or any(
            v < 1 for v in self.input_dims.values()
        ):
            raise ValueError(
                "input_dims must specify each modality; CLI can infer numeric feature widths"
            )
        if self.hidden_dim < 1 or self.expert_dim < 1 or self.num_classes < 2:
            raise ValueError("Invalid hidden/expert dimension or number of classes")
        if not self.scales or any(s < 1 or int(s) != s for s in self.scales):
            raise ValueError("scales must contain positive integers")
        if not 1 <= self.ke <= self.num_experts:
            raise ValueError("Require 1 <= ke <= num_experts")
        if (
            min(self.ki_factor, self.sample_factor, self.kc, self.candidate_per_source)
            <= 0
        ):
            raise ValueError("Sparsity budgets must be positive")
        if (
            min(
                self.tau_u,
                self.tau_c,
                self.tau_e,
                self.tau_d,
                self.tau_t,
                self.tau_g,
                self.epsilon,
            )
            <= 0
        ):
            raise ValueError("Temperatures and epsilon must be positive")
        if min(self.lambda_r, self.lambda_kappa, self.lambda_time) < 0:
            raise ValueError("Reliability and temporal weights must be nonnegative")
        if self.conv_kernel < 1 or self.conv_kernel % 2 == 0:
            raise ValueError("conv_kernel must be positive and odd")
        if self.cross_mode not in {"dense", "sparse"}:
            raise ValueError("Unknown cross_mode")
        if self.expert_training not in {"gumbel_st", "selected"}:
            raise ValueError("Unknown expert_training")
        if self.missing_policy not in {"finite_prior", "exclude"}:
            raise ValueError("Unknown missing_policy")
        if self.balance_convention not in {"raw", "normalized"}:
            raise ValueError("Unknown balance convention")
        if self.logvar_min >= self.logvar_max:
            raise ValueError("Invalid log variance clamp")
        if self.reliability_context_hidden < 1 or self.reliability_context_bound <= 0:
            raise ValueError("Invalid contextual reliability dimensions or bound")
        if not isinstance(self.reliability_context_content, bool):
            raise ValueError("reliability_context_content must be boolean")
        if not isinstance(self.expert_residual, bool):
            raise ValueError("expert_residual must be boolean")
        if type(self.text_message_pooling) is not bool:
            raise ValueError("text_message_pooling must be boolean")
        if self.text_message_pooling and (
            "text" not in self.modalities or len(self.modalities) < 2
        ):
            raise ValueError("text_message_pooling requires text and another modality")
        if type(
            self.sign_readout_pooling
        ) is not str or self.sign_readout_pooling not in {
            "legacy",
            "full_pool",
            "text_message",
        }:
            raise ValueError(
                "sign_readout_pooling must be legacy, full_pool or text_message"
            )
        if self.sign_readout_pooling != "legacy":
            if self.text_message_pooling:
                raise ValueError(
                    "sign readout requires the original shared pooling state"
                )
            if (
                self.task == "classification"
                or "text" not in self.modalities
                or len(self.modalities) < 2
            ):
                raise ValueError(
                    "sign readout requires scalar sentiment/regression with text and another modality"
                )
        return self

    def to_dict(self):
        return asdict(self)


def load_config(path):
    with open(path, encoding="utf-8") as f:
        # JSON exponent notation such as 1e-08 is valid numeric JSON but can
        # become a string under PyYAML's YAML-1.1 scalar resolver.
        cfg = json.load(f) if str(path).lower().endswith(".json") else yaml.safe_load(f)
    if not isinstance(cfg, dict) or "model" not in cfg:
        raise ValueError("Configuration must contain a model mapping")
    ModelConfig(**cfg["model"])  # Reject misspelled model fields.
    return cfg


ABLATIONS = (
    "full",
    "no_multiscale",
    "no_sparse_temporal",
    "no_uncertainty",
    "no_reliability_coupling",
    "no_moe",
    "no_missing_consistency",
    "no_rho_consistency",
    "token_only",
    "expert_only",
    "separate_rho",
)


def apply_ablation(config, name):
    if name not in ABLATIONS:
        raise ValueError(f"Unknown ablation: {name}")
    c = copy.deepcopy(config)
    m, loss = c["model"], c.setdefault("loss", {})
    if name == "no_multiscale":
        m["scales"] = [1]
    if name == "no_sparse_temporal":
        m["sparse_temporal"] = False
    if name == "no_uncertainty":
        m["use_uncertainty"] = False
    if name in {"no_reliability_coupling", "expert_only"}:
        m["lambda_r"] = 0.0
    if name == "no_moe":
        m["use_moe"] = False
    if name == "token_only":
        m["expert_reliability"] = False
    if name == "separate_rho":
        m["separate_rho"] = True
    if name == "no_missing_consistency":
        c.setdefault("train", {})["missing_consistency"] = False
        loss.update(con=0.0, kd=0.0, rho=0.0)
    if name == "no_rho_consistency":
        loss["rho"] = 0.0
    c["ablation"] = name
    return c
