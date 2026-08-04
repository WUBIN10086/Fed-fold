import inspect
import logging
from argparse import Namespace
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch

from openfold.utils.exponential_moving_average import ExponentialMovingAverage
from openfold.utils.lora import LoRAConfig


TRAINER_ARG_NAMES = (
    "num_nodes",
    "precision",
    "max_epochs",
    "max_steps",
    "log_every_n_steps",
    "flush_logs_every_n_steps",
    "num_sanity_val_steps",
    "reload_dataloaders_every_n_epochs",
    "accumulate_grad_batches",
)


CHECKPOINT_WEIGHT_SOURCES = ("ema", "module", "state_dict", "auto")
DEFAULT_INIT_WEIGHTS_SOURCE = "ema"
TRAIN_INIT_AUTO_ORDER = ("module", "state_dict", "direct", "ema")
INFERENCE_AUTO_ORDER = ("ema", "state_dict", "module", "direct")


def _checkpoint_source_mapping(checkpoint, source):
    if not isinstance(checkpoint, Mapping):
        return None
    if source == "ema":
        ema = checkpoint.get("ema")
        if isinstance(ema, Mapping) and isinstance(ema.get("params"), Mapping):
            return ema["params"]
        return None
    if source in ("module", "state_dict"):
        value = checkpoint.get(source)
        return value if isinstance(value, Mapping) else None
    if source == "direct":
        tensors = {
            key: value
            for key, value in checkpoint.items()
            if isinstance(key, str) and isinstance(value, torch.Tensor)
        }
        return tensors or None
    raise ValueError(f"Unknown checkpoint weight source: {source}")


def extract_alphafold_weights(
    checkpoint,
    source: str = "auto",
    *,
    auto_order: Sequence[str] = TRAIN_INIT_AUTO_ORDER,
    return_source: bool = False,
) -> Union[
    Dict[str, torch.Tensor],
    Tuple[Dict[str, torch.Tensor], str],
]:
    if source not in CHECKPOINT_WEIGHT_SOURCES:
        raise ValueError(
            f"source must be one of {CHECKPOINT_WEIGHT_SOURCES}, got {source!r}"
        )

    selected_source = source
    state_dict = None
    if source == "auto":
        for candidate in auto_order:
            state_dict = _checkpoint_source_mapping(checkpoint, candidate)
            if state_dict is not None:
                selected_source = (
                    "state_dict" if candidate == "direct" else candidate
                )
                break
    else:
        state_dict = _checkpoint_source_mapping(checkpoint, source)
        if state_dict is None:
            raise ValueError(
                f"Checkpoint does not contain requested weight source {source!r}"
            )

    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint does not contain a usable model state dict")

    normalized = {}
    for key, value in state_dict.items():
        while key.startswith("module."):
            key = key[len("module."):]
        normalized[key] = value
    model_keys = {
        key[len("model."):]: value
        for key, value in normalized.items()
        if key.startswith("model.")
    }
    weights = model_keys if model_keys else normalized
    if not weights:
        raise ValueError(
            f"Checkpoint weight source {selected_source!r} contains no tensors"
        )
    if return_source:
        return weights, selected_source
    return weights


def build_trainer_kwargs(
    args: Namespace,
    trainer_init: Callable,
) -> Tuple[Dict[str, object], Tuple[str, ...]]:
    signature = inspect.signature(trainer_init)
    supported = set(signature.parameters)
    kwargs = {}
    ignored = []
    for name in TRAINER_ARG_NAMES:
        if not hasattr(args, name):
            continue
        if name in supported:
            kwargs[name] = getattr(args, name)
        else:
            ignored.append(name)
    return kwargs, tuple(ignored)


def log_ignored_trainer_args(ignored: Tuple[str, ...]) -> None:
    for name in ignored:
        logging.warning(
            "Trainer argument %s is not supported by this PyTorch Lightning "
            "version and will be ignored",
            name,
        )


def resolve_learning_rate(
    requested: Optional[float],
    lora_enabled: bool,
) -> float:
    learning_rate = (
        (1e-4 if lora_enabled else 1e-3)
        if requested is None
        else float(requested)
    )
    if learning_rate < 0:
        raise ValueError("learning_rate must be nonnegative")
    return learning_rate


def resolve_lora_config(
    rank: Optional[int],
    alpha: Optional[float],
    dropout: Optional[float],
    target: Optional[str],
    checkpoint_config: Optional[Mapping[str, object]] = None,
    full_checkpoint_resume: bool = False,
) -> LoRAConfig:
    cli_values = {
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "target": target,
    }
    defaults = {
        "rank": 0,
        "alpha": 16.0,
        "dropout": 0.0,
        "target": "structure_module",
    }

    if checkpoint_config is not None:
        restored = LoRAConfig.from_dict(checkpoint_config)
        restored_values = restored.to_dict()
        mismatches = {
            key: (value, restored_values[key])
            for key, value in cli_values.items()
            if value is not None and value != restored_values[key]
        }
        if mismatches:
            raise ValueError(
                "CLI LoRA configuration does not match checkpoint: "
                + ", ".join(
                    f"{key}=CLI {values[0]!r}, checkpoint {values[1]!r}"
                    for key, values in mismatches.items()
                )
            )
        return restored

    resolved = {
        key: defaults[key] if value is None else value
        for key, value in cli_values.items()
    }
    config = LoRAConfig.from_dict(resolved)
    if full_checkpoint_resume and checkpoint_config is None and config.enabled:
        raise ValueError(
            "Cannot resume a LoRA model from a checkpoint without lora_config"
        )
    return config


def validate_lora_runtime(config: LoRAConfig, script_modules: bool) -> None:
    if config.enabled and script_modules:
        raise ValueError(
            "--script_modules is not supported with LoRA; disable TorchScript "
            "or use --lora_rank 0"
        )


def should_reset_ema(full_checkpoint_resume: bool) -> bool:
    return not full_checkpoint_resume


def reset_ema_from_model(model_module, decay: float) -> None:
    model_module.ema = ExponentialMovingAverage(
        model=model_module.model,
        decay=decay,
    )
