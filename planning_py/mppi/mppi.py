from typing import Tuple, Callable

import torch


class MPPI:
    def __init__(
        self,
        horizon: int,
        num_samples: int,
        dim_state: int,
        dim_control: int,
        dynamics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        stage_cost: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        terminal_cost: Callable[[torch.Tensor], torch.Tensor],
        u_min: torch.Tensor,
        u_max: torch.Tensor,
        sigmas: torch.Tensor,
        lambda_: float,
        device=torch.device("cuda"),
        dtype=torch.float32,
        seed: int = 42,
    ) -> None:
        torch.manual_seed(seed)

        # * check dimensions
        assert u_min.shape == (dim_control,)
        assert u_max.shape == (dim_control,)
        assert sigmas.shape == (dim_control,)

        # * device
        if torch.cuda.is_available() and device == torch.device("cuda"):
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        # * dtype
        self._dtype = dtype

        # * set parameters
        self._horizon = horizon
        self._num_samples = num_samples
        self._dim_state = dim_state
        self._dim_control = dim_control
        self._dynamics = dynamics
        self._stage_cost = stage_cost
        self._terminal_cost = terminal_cost
        self._u_min = u_min.clone().detach().to(self._device, self._dtype)
        self._u_max = u_max.clone().detach().to(self._device, self._dtype)
        self._sigmas = sigmas.clone().detach().to(self._device, self._dtype)
        self._lambda = lambda_

        # * noise distribution
        zero_mean = torch.zeros(dim_control, device=self._device, dtype=self._dtype)
        initial_covariance = torch.diag(sigmas**2).to(self._device, self._dtype)
        self._inv_covarince = torch.inverse(initial_covariance).to(
            self._device, self._dtype
        )

        self._noise_distribution = torch.distributions.MultivariateNormal(
            loc=zero_mean, covariance_matrix=initial_covariance
        )

        self._sample_shape = torch.Size([self._num_samples, self._horizon])

        # * sampling with reparameting trick
        self._action_noises = self._noise_distribution.rsample(
            sample_shape=self._sample_shape
        )

        zero_mean_seq = torch.zeros(
            self._horizon, self._dim_control, device=self._device, dtype=self._dtype
        )
        self._perturbed_action_seqs = torch.clamp(
            zero_mean_seq + self._action_noises, self._u_min, self._u_max
        )

        self._previous_action_seq = zero_mean_seq

        # * inner variables
        self._state_seq_batch = torch.zeros(
            self._num_samples,
            self._horizon + 1,
            self._dim_state,
            device=self._device,
            dtype=self._dtype,
        )
        self._weights = torch.zeros(
            self._num_samples, device=self._device, dtype=self._dtype
        )

    def solve(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve the Optimal Control Problem
        Args:
            state (torch.Tensor): Current State
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of predictive control and state sequence
        """

        assert state.shape == (self._dim_state,)

        # check and convert to torch tensor if not
        if not torch.is_tensor(state):
            state = torch.tensor(state, device=self._device, dtype=self._dtype)
        else:
            if state.device != self._device or state.dtype != self._dtype:
                state = state.to(self._device, self._dtype)

        # * サンプリングの分布の平均は前回の最適解
        mean_action_seq = self._previous_action_seq.clone().detach()

        # * アクションに乗せるノイズをガウス分布からサンプリング
        self._action_noises = self._noise_distribution.rsample(
            sample_shape=self._sample_shape
        )

        # * アクションにノイズを加える + clamp
        self._perturbed_action_seqs = torch.clamp(
            mean_action_seq + self._action_noises, self._u_min, self._u_max
        )

        # * rollout samples in parallel
        self._state_seq_batch[:, 0, :] = state.repeat(self._num_samples, 1)

        for t in range(self._horizon):
            self._state_seq_batch[:, t + 1, :] = self._dynamics(
                self._state_seq_batch[:, t, :], self._perturbed_action_seqs[:, t, :]
            )

        # * サンプルのコストを計算
        stage_cost = torch.zeros(
            self._num_samples, self._horizon, device=self._device, dtype=self._dtype
        )
        action_cost = torch.zeros(
            self._num_samples, self._horizon, device=self._device, dtype=self._dtype
        )

        for t in range(self._horizon):
            stage_cost[:, t] = self._stage_cost(
                self._state_seq_batch[:, t, :], self._perturbed_action_seqs[:, t, :]
            )
            action_cost[:, t] = (
                mean_action_seq[t]
                @ self._inv_covarince
                @ self._perturbed_action_seqs[:, t].T
            )

        terminal_cost = self._terminal_cost(self._state_seq_batch[:, -1, :])

        cost = (
            torch.sum(stage_cost, dim=1)
            + terminal_cost
            + torch.sum(self._lambda * action_cost, dim=1)
        )

        # * 重みの計算
        self._weights = torch.softmax(-cost / self._lambda, dim=0)

        # * 最適解の計算（重み付き平均）
        optimal_action_seq = torch.sum(
            self._weights.view(self._num_samples, 1, 1) * self._perturbed_action_seqs,
            dim=0,
        )
        optimal_state_seq = torch.zeros(
            1,
            self._horizon + 1,
            self._dim_state,
            device=self._device,
            dtype=self._dtype,
        )

        optimal_state_seq[:, 0, :] = state
        expanded_optimal_action_seq = optimal_action_seq.repeat(1, 1, 1)
        for t in range(self._horizon):
            optimal_state_seq[:, t + 1, :] = self._dynamics(
                optimal_state_seq[:, t, :], expanded_optimal_action_seq[:, t, :]
            )

        self._previous_action_seq = optimal_action_seq

        return optimal_action_seq, optimal_state_seq

    def get_top_samples(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top samples
        Args:
            num_samples (int): Number of top samples
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of top samples of action and state sequence
        """
        assert num_samples <= self._num_samples

        top_indices = torch.topk(self._weights, num_samples).indices

        top_samples = self._state_seq_batch[top_indices]
        top_weights = self._weights[top_indices]

        top_samples = top_samples[torch.argsort(top_weights, descending=True)]
        top_weights = top_weights[torch.argsort(top_weights, descending=True)]

        return top_samples, top_weights
