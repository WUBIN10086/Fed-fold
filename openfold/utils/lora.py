import math
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from openfold.model.primitives import Linear


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 0
    alpha: float = 16.0
    dropout: float = 0.0
    target: str = "structure_module"

    def __post_init__(self):
        if self.rank < 0:
            raise ValueError("LoRA rank must be nonnegative")
        if self.alpha < 0:
            raise ValueError("LoRA alpha must be nonnegative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if not self.target:
            raise ValueError("LoRA target must not be empty")

    @property
    def enabled(self) -> bool:
        return self.rank > 0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> "LoRAConfig":
        return cls(
            rank=int(values.get("rank", 0)),
            alpha=float(values.get("alpha", 16.0)),
            dropout=float(values.get("dropout", 0.0)),
            target=str(values.get("target", "structure_module")),
        )


class LoRALinear(Linear):
    """OpenFold Linear with a trainable low-rank residual."""

    def __init__(
        self,
        base: Linear,
        rank: int,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        if isinstance(base, LoRALinear):
            raise ValueError("Refusing to inject LoRA into an existing LoRALinear")
        if not isinstance(base, Linear):
            raise TypeError("LoRALinear can only wrap openfold.model.primitives.Linear")
        if rank <= 0:
            raise ValueError("LoRALinear rank must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")

        super().__init__(
            base.in_features,
            base.out_features,
            bias=base.bias is not None,
            precision=base.precision,
        )
        self.weight = base.weight
        self.bias = base.bias
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.lora_A = nn.Parameter(
            torch.empty(
                self.rank,
                self.in_features,
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                self.out_features,
                self.rank,
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def _adapter_dtype(self, inputs: torch.Tensor) -> torch.dtype:
        if self.precision is not None:
            return self.precision
        if inputs.dtype in (torch.float16, torch.bfloat16):
            return inputs.dtype
        return self.lora_A.dtype

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = super().forward(inputs)
        adapter_dtype = self._adapter_dtype(inputs)
        adapter_input = self.lora_dropout(inputs).to(dtype=adapter_dtype)
        adapter = F.linear(
            F.linear(adapter_input, self.lora_A.to(dtype=adapter_dtype)),
            self.lora_B.to(dtype=adapter_dtype),
        )
        return base_output + (adapter * self.scaling).to(dtype=base_output.dtype)

    def merged_weight(self) -> torch.Tensor:
        _validate_adapter_shapes(
            self.weight,
            self.lora_A,
            self.lora_B,
            expected_rank=self.rank,
        )
        delta = torch.matmul(
            self.lora_B.to(dtype=torch.float32),
            self.lora_A.to(dtype=torch.float32),
        )
        return self.weight.detach() + (delta * self.scaling).to(
            device=self.weight.device,
            dtype=self.weight.dtype,
        )

    def to_merged_linear(self) -> Linear:
        merged = Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            precision=self.precision,
        ).to(device=self.weight.device, dtype=self.weight.dtype)
        merged.weight = nn.Parameter(
            self.merged_weight(),
            requires_grad=self.weight.requires_grad,
        )
        if self.bias is not None:
            merged.bias = nn.Parameter(
                self.bias.detach().clone(),
                requires_grad=self.bias.requires_grad,
            )
        return merged


def _validate_adapter_shapes(
    weight: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    expected_rank: Optional[int] = None,
) -> int:
    if lora_A.ndim != 2 or lora_B.ndim != 2 or weight.ndim != 2:
        raise ValueError("LoRA A, B, and base weight must all be rank-2 tensors")
    rank = lora_A.shape[0]
    if expected_rank is not None and rank != expected_rank:
        raise ValueError(
            f"LoRA A rank {rank} does not match configured rank {expected_rank}"
        )
    if lora_B.shape[1] != rank:
        raise ValueError("LoRA A/B rank dimensions do not match")
    if lora_A.shape[1] != weight.shape[1]:
        raise ValueError("LoRA A input dimension does not match base weight")
    if lora_B.shape[0] != weight.shape[0]:
        raise ValueError("LoRA B output dimension does not match base weight")
    return rank


_EVOFORMER_ATTENTION_RE = re.compile(
    r"^evoformer\.blocks\.\d+\."
    r"(?:msa_att_row\.mha|msa_att_col\._msa_att\.mha|"
    r"pair_stack\.tri_att_(?:start|end)\.mha)\.linear_[qkvo]$"
)


def module_matches_target(name: str, target: str) -> bool:
    if target == "structure_module":
        return name.startswith("structure_module.")
    if target == "evoformer_attention":
        return _EVOFORMER_ATTENTION_RE.fullmatch(name) is not None
    if target == "all_linear":
        return True
    if target.startswith("re:"):
        pattern = target[len("re:"):]
        if not pattern:
            raise ValueError("Custom LoRA regex must not be empty")
        return re.search(pattern, name) is not None

    prefixes = [prefix.strip() for prefix in target.split(",") if prefix.strip()]
    if not prefixes:
        raise ValueError("Custom LoRA prefixes must not be empty")
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def _replace_submodule(model: nn.Module, name: str, replacement: nn.Module) -> None:
    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
    else:
        parent = model
        child_name = name
    setattr(parent, child_name, replacement)


def inject_lora(
    model: nn.Module,
    config: LoRAConfig,
) -> List[str]:
    if not config.enabled:
        return []
    existing = [name for name, module in model.named_modules()
                if isinstance(module, LoRALinear)]
    if existing:
        raise ValueError(
            "LoRA is already injected into the model: " + ", ".join(existing[:5])
        )

    candidates = [
        (name, module)
        for name, module in list(model.named_modules())
        if name
        and isinstance(module, Linear)
        and module_matches_target(name, config.target)
    ]
    if not candidates:
        raise ValueError(
            f"LoRA target {config.target!r} matched zero OpenFold Linear modules"
        )

    replaced = []
    for name, module in candidates:
        _replace_submodule(
            model,
            name,
            LoRALinear(
                module,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
            ),
        )
        replaced.append(name)
    return replaced


def freeze_non_lora_parameters(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)


def configure_lora(model: nn.Module, config: LoRAConfig) -> List[str]:
    replaced = inject_lora(model, config)
    if config.enabled:
        freeze_non_lora_parameters(model)
    return replaced


def parameter_counts(model: nn.Module) -> Dict[str, float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    ratio = (100.0 * trainable / total) if total else 0.0
    return {
        "total": total,
        "trainable": trainable,
        "trainable_percent": ratio,
    }


def merge_lora_modules(model: nn.Module) -> List[str]:
    modules = [
        (name, module)
        for name, module in list(model.named_modules())
        if isinstance(module, LoRALinear)
    ]
    for name, module in modules:
        _replace_submodule(model, name, module.to_merged_linear())
    return [name for name, _ in modules]


def merge_lora_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    alpha: float,
    rank: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    output = {key: value.detach().clone() for key, value in state_dict.items()}
    a_suffix = ".lora_A"
    b_suffix = ".lora_B"
    a_prefixes = {key[:-len(a_suffix)] for key in output if key.endswith(a_suffix)}
    b_prefixes = {key[:-len(b_suffix)] for key in output if key.endswith(b_suffix)}
    if not a_prefixes and not b_prefixes:
        raise ValueError("State dict contains no LoRA adapter parameters")
    if a_prefixes != b_prefixes:
        missing_a = sorted(b_prefixes - a_prefixes)
        missing_b = sorted(a_prefixes - b_prefixes)
        raise ValueError(
            f"Unpaired LoRA tensors; missing A={missing_a[:5]}, "
            f"missing B={missing_b[:5]}"
        )

    for prefix in sorted(a_prefixes):
        weight_key = prefix + ".weight"
        if weight_key not in output:
            raise ValueError(f"Missing base weight for LoRA module {prefix}")
        lora_A = output[prefix + a_suffix]
        lora_B = output[prefix + b_suffix]
        actual_rank = _validate_adapter_shapes(
            output[weight_key],
            lora_A,
            lora_B,
            expected_rank=rank,
        )
        scaling = float(alpha) / actual_rank
        delta = torch.matmul(
            lora_B.to(dtype=torch.float32),
            lora_A.to(dtype=torch.float32),
        )
        output[weight_key] = output[weight_key] + (delta * scaling).to(
            dtype=output[weight_key].dtype,
            device=output[weight_key].device,
        )
        del output[prefix + a_suffix]
        del output[prefix + b_suffix]

    remaining = [key for key in output if ".lora_" in key]
    if remaining:
        raise ValueError(f"Unexpected LoRA keys remain after merge: {remaining[:5]}")
    return output
