import torch


def compute_alphafold_learning_rate(
    step_no: int,
    base_lr: float = 0.0,
    max_lr: float = 0.001,
    warmup_no_steps: int = 1000,
    start_decay_after_n_steps: int = 50000,
    decay_every_n_steps: int = 50000,
    decay_factor: float = 0.95,
) -> float:
    if step_no < 0:
        raise ValueError("step_no must be nonnegative")
    if warmup_no_steps < 0:
        raise ValueError("warmup_no_steps must be nonnegative")
    if start_decay_after_n_steps < 0:
        raise ValueError("start_decay_after_n_steps must be nonnegative")
    if decay_every_n_steps <= 0:
        raise ValueError("decay_every_n_steps must be positive")
    if warmup_no_steps > start_decay_after_n_steps:
        raise ValueError(
            "warmup_no_steps must not exceed start_decay_after_n_steps"
        )
    if not 0.0 < decay_factor <= 1.0:
        raise ValueError("decay_factor must be in (0, 1]")

    if warmup_no_steps > 0 and step_no <= warmup_no_steps:
        fraction = step_no / warmup_no_steps
        return base_lr + fraction * (max_lr - base_lr)
    if step_no > start_decay_after_n_steps:
        steps_since_decay = step_no - start_decay_after_n_steps
        exponent = (steps_since_decay // decay_every_n_steps) + 1
        return max_lr * (decay_factor ** exponent)
    return max_lr


class AlphaFoldLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """ Implements the learning rate schedule defined in the AlphaFold 2
        supplement. A linear warmup is followed by a plateau at the maximum
        learning rate and then exponential decay.
         
        Note that the initial learning rate of the optimizer in question is 
        ignored; use this class' base_lr parameter to specify the starting 
        point of the warmup.
    """
    def __init__(self, 
        optimizer, 
        last_epoch: int = -1, 
        verbose: bool = False,
        base_lr: float = 0.,
        max_lr: float = 0.001,
        warmup_no_steps: int = 1000,
        start_decay_after_n_steps: int = 50000,
        decay_every_n_steps: int = 50000,
        decay_factor: float = 0.95,
    ):
        compute_alphafold_learning_rate(
            step_no=max(last_epoch, 0),
            base_lr=base_lr,
            max_lr=max_lr,
            warmup_no_steps=warmup_no_steps,
            start_decay_after_n_steps=start_decay_after_n_steps,
            decay_every_n_steps=decay_every_n_steps,
            decay_factor=decay_factor,
        )

        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.verbose = verbose
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.warmup_no_steps = warmup_no_steps
        self.start_decay_after_n_steps = start_decay_after_n_steps
        self.decay_every_n_steps = decay_every_n_steps
        self.decay_factor = decay_factor

        # PyTorch >= 2.2 removed LRScheduler's `verbose` kwarg.
        super(AlphaFoldLRScheduler, self).__init__(
            optimizer,
            last_epoch=last_epoch,
        )

    def state_dict(self):
        state_dict = {
            k:v for k,v in self.__dict__.items() if k not in ["optimizer"]
        }

        return state_dict

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_lr(self):
        if(not self._get_lr_called_within_step):
            raise RuntimeError(
                "To get the last learning rate computed by the scheduler, use "
                "get_last_lr()"
            )

        lr = compute_alphafold_learning_rate(
            step_no=self.last_epoch,
            base_lr=self.base_lr,
            max_lr=self.max_lr,
            warmup_no_steps=self.warmup_no_steps,
            start_decay_after_n_steps=self.start_decay_after_n_steps,
            decay_every_n_steps=self.decay_every_n_steps,
            decay_factor=self.decay_factor,
        )

        return [lr for group in self.optimizer.param_groups]
