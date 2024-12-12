from dataclasses import dataclass
from typing import Any, Callable, List
import torch
from utils.loss_utils import ssim
import wandb


@dataclass
class GracePeriod:
    start: int
    end: int


def parse_grace_periods(arg: str | None) -> List[GracePeriod]:
    if arg is None:
        return []

    periods = []
    for pair in arg.split(","):
        frequency, grace_period = map(int, pair.split(":"))
        periods.append(GracePeriod(frequency, grace_period))
    return periods


class EarlyStoppingHandler:
    def __init__(
        self,
        use_early_stopping: bool,
        start_early_stopping_iteration: int,
        grace_periods: List[GracePeriod],
        early_stopping_check_interval: int,
        n_patience_epochs: int,
        device="cuda",
        use_wandb=True,
    ) -> None:
        self.use_early_stopping = use_early_stopping
        self.start_early_stopping_iteration = start_early_stopping_iteration
        self.grace_periods = grace_periods
        self.early_stopping_check_interval = early_stopping_check_interval
        self.best_ssim = -1.0
        self.n_epochs_without_improvement = 0
        self.n_patience_epochs = n_patience_epochs
        self.device = device
        self.use_wandb = use_wandb

    @torch.no_grad()
    def stop_early(
        self,
        step: int,
        test_cameras: List[Any],
        render_func: Callable,
    ) -> bool:
        if not self.use_early_stopping:
            return False

        if step % self.early_stopping_check_interval != 0:
            return False
        
        if step < self.start_early_stopping_iteration:
            return False

        ssims = []

        for camera in test_cameras:
            image = torch.clamp(render_func(camera), 0.0, 1.0)
            gt_image = torch.clamp(camera.original_image.to(self.device), 0.0, 1.0)

            ssims.append(ssim(image, gt_image))

        new_ssim = torch.tensor(ssims).mean().detach().cpu().item()

        if self.use_wandb:
            wandb.log({"early_stopping_test/ssim": new_ssim})

        is_in_grace_period = False

        for grace_period in self.grace_periods:
            if (grace_period.start <= step) and (step < grace_period.end):
                is_in_grace_period = True

        if is_in_grace_period:
            return False

        if new_ssim > (self.best_ssim + 0.0001):
            print(f"\nNew best SSIM {new_ssim} > {self.best_ssim}")
            self.best_ssim = new_ssim
            self.n_epochs_without_improvement = 0
        else:
            self.n_epochs_without_improvement = self.n_epochs_without_improvement + 1
            print(
                f"\nSSIM did not meaningfully improve for {self.n_epochs_without_improvement}: {new_ssim} < {self.best_ssim} + 0.0001"
            )

        if self.n_epochs_without_improvement > self.n_patience_epochs:
            print(
                f"\nNo improvement in SSIM for {self.n_epochs_without_improvement}, stopping training at step {step}"
            )
            return True

        return False
