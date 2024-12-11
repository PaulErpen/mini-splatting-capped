from typing import NamedTuple
import unittest
from early_stopping import EarlyStoppingHandler, GracePeriod
import torch


class TestCamera(NamedTuple):
    original_image: torch.Tensor


class EarlyStoppingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_cameras = [
            TestCamera(original_image=torch.zeros((128, 128, 3))),
            TestCamera(original_image=torch.zeros((128, 128, 3))),
            TestCamera(original_image=torch.zeros((128, 128, 3))),
        ]

    def test_given_valid_paramaters__when_initializing__then_do_not_throw_any_errors(
        self,
    ) -> None:
        EarlyStoppingHandler(
            use_early_stopping=True,
            start_early_stopping_iteration=10,
            grace_periods=[],
            early_stopping_check_interval=3,
            n_patience_epochs=3,
            device="cpu",
            use_wandb=False,
        )

    def test_given_valid_paramaters__when_checking_for_early_stop__then_do_not_throw_any_errors(
        self,
    ) -> None:
        handler = EarlyStoppingHandler(
            use_early_stopping=True,
            start_early_stopping_iteration=10,
            grace_periods=[],
            early_stopping_check_interval=3,
            n_patience_epochs=3,
            device="cpu",
            use_wandb=False,
        )

        handler.stop_early(
            step=1, test_cameras=[], render_func=lambda x: torch.rand_like(800, 800, 3)
        )

    def test_given_a_model_that_does_not_improve__when_checking_for_early_stop_50_times__then_return_true(
        self,
    ) -> None:
        handler = EarlyStoppingHandler(
            use_early_stopping=True,
            start_early_stopping_iteration=10,
            grace_periods=[],
            early_stopping_check_interval=2,
            n_patience_epochs=3,
            device="cpu",
            use_wandb=False,
        )

        for i in range(49):
            handler.stop_early(
                step=i,
                test_cameras=self.test_cameras,
                render_func=lambda x: torch.ones_like(x.original_image),
            )

        self.assertTrue(
            handler.stop_early(
                step=50,
                test_cameras=self.test_cameras,
                render_func=lambda x: torch.ones_like(x.original_image),
            )
        )

    def test_given_a_model_that_does_not_improve_but_a_forgiving_grace_period__when_checking_for_early_stop_28_times__then_return_false(
        self,
    ) -> None:
        handler = EarlyStoppingHandler(
            use_early_stopping=True,
            start_early_stopping_iteration=10,
            grace_periods=[GracePeriod(10, 8)],
            early_stopping_check_interval=2,
            n_patience_epochs=3,
            device="cpu",
            use_wandb=False,
        )

        for i in range(27):
            handler.stop_early(
                step=i,
                test_cameras=self.test_cameras,
                render_func=lambda x: torch.ones_like(x.original_image),
            )

        self.assertFalse(
            handler.stop_early(
                step=28,
                test_cameras=self.test_cameras,
                render_func=lambda x: torch.ones_like(x.original_image),
            )
        )

    def test_given_a_model_that_does_not_improve_but_a_strict_grace_period__when_checking_for_early_stop_28_times__then_return_True(
        self,
    ) -> None:
        handler = EarlyStoppingHandler(
            use_early_stopping=True,
            start_early_stopping_iteration=10,
            grace_periods=[GracePeriod(10, 3)],
            early_stopping_check_interval=2,
            n_patience_epochs=3,
            device="cpu",
            use_wandb=False,
        )

        for i in range(27):
            handler.stop_early(
                step=i,
                test_cameras=self.test_cameras,
                render_func=lambda x: torch.ones_like(x.original_image),
            )

        self.assertTrue(
            handler.stop_early(
                step=28,
                test_cameras=self.test_cameras,
                render_func=lambda x: torch.ones_like(x.original_image),
            )
        )

    def test_given_a_model_that_does_not_improve_but_a_strict_grace_period_and_a_late_start__when_checking_for_early_stop_28_times__then_return_false(
        self,
    ) -> None:
        handler = EarlyStoppingHandler(
            use_early_stopping=True,
            start_early_stopping_iteration=26,
            grace_periods=[GracePeriod(10, 3)],
            early_stopping_check_interval=2,
            n_patience_epochs=3,
            device="cpu",
            use_wandb=False,
        )

        for i in range(27):
            handler.stop_early(
                step=i,
                test_cameras=self.test_cameras,
                render_func=lambda x: torch.ones_like(x.original_image),
            )

        self.assertFalse(
            handler.stop_early(
                step=28,
                test_cameras=self.test_cameras,
                render_func=lambda x: torch.ones_like(x.original_image),
            )
        )


if __name__ == "__main__":
    unittest.main()
