from dataclasses import dataclass
from typing import Tuple, List, Callable
import math

import torch
import numpy as np
import matplotlib.pyplot as plt

import tqdm
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import os


@torch.jit.script
def angle_normalize(x):
    """
    convert the angle to the range of [-pi, pi]
    """
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi


class Map:
    @dataclass
    class CircleObstacle:
        center: np.ndarray
        radius: float

    @dataclass
    class RectangleObstacle:
        center: np.ndarray
        width: float
        height: float

    def __init__(
        self,
        map_size: Tuple[int, int] = (20, 20),
        cell_size: float = 0.01,
        device=torch.device("cuda"),
        dtype=torch.float32,
    ) -> None:
        """
        Args:
            map_size: (width, height) [m] of the map
            cell_size: (cell_size) [m]
        """
        # * device
        if torch.cuda.is_available() and device == torch.device("cuda"):
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        # * dtype
        self._dtype = dtype

        cell_map_dim = [0, 0]
        cell_map_dim[0] = math.ceil(map_size[0] / cell_size)
        cell_map_dim[1] = math.ceil(map_size[1] / cell_size)

        self._map = np.zeros(cell_map_dim)
        self._cell_size = cell_size

        # * cell map center
        self._cell_map_origin = np.zeros(2)
        self._cell_map_origin = np.array(
            [cell_map_dim[0] / 2, cell_map_dim[1] / 2]
        ).astype(int)

        self._torch_cell_map_origin = torch.from_numpy(self._cell_map_origin).to(
            self._device, self._dtype
        )

        # * limit of the map
        x_range = self._cell_size * self._map.shape[0]
        y_range = self._cell_size * self._map.shape[1]
        self.x_lim = [-x_range / 2, x_range / 2]  # [m]
        self.y_lim = [-y_range / 2, y_range / 2]  # [m]

        # * inner variables
        self._circle_obstacles: List[Map.CircleObstacle] = []
        self._rectangle_obstacles: List[Map.RectangleObstacle] = []

    def add_circle_obstacle(self, center: np.ndarray, radius: float) -> None:
        assert len(center) == 2, "center must be 2D"
        assert radius > 0, "radius must be positive"

        center_occ = (center / self._cell_size) + self._cell_map_origin
        center_occ = np.round(center_occ).astype(int)
        radius_occ = math.ceil(radius / self._cell_size)

        # * add to occ map
        for i in range(-radius_occ, radius_occ + 1):
            for j in range(-radius_occ, radius_occ + 1):
                if i**2 + j**2 <= radius_occ:
                    i_bounded = np.clip(center_occ[0] + i, 0, self._map.shape[0] - 1)
                    j_bounded = np.clip(center_occ[1] + j, 0, self._map.shape[1] - 1)
                    self._map[i_bounded, j_bounded] = 1

        # * add to circle obstacle list to visualize
        self._circle_obstacles.append(Map.CircleObstacle(center, radius))

    def add_rectangle_obstacle(
        self, center: np.ndarray, width: float, height: float
    ) -> None:
        assert len(center) == 2, "center must be 2D"
        assert width > 0, "width must be positive"
        assert height > 0, "height must be positive"

        # * convert to cell map
        center_occ = (center / self._cell_size) + self._cell_map_origin
        center_occ = np.ceil(center_occ).astype(int)
        width_occ = math.ceil(width / self._cell_size)
        height_occ = math.ceil(height / self._cell_size)

        # * add to occ map
        x_init = center_occ[0] - math.ceil(height_occ / 2)
        x_end = center_occ[0] + math.ceil(height_occ / 2)
        y_init = center_occ[1] - math.ceil(width_occ / 2)
        y_end = center_occ[1] + math.ceil(width_occ / 2)

        # * deal with out of bound
        x_init = np.clip(x_init, 0, self._map.shape[0] - 1)
        x_end = np.clip(x_end, 0, self._map.shape[0] - 1)
        y_init = np.clip(y_init, 0, self._map.shape[1] - 1)
        y_end = np.clip(y_end, 0, self._map.shape[1] - 1)

        self._map[x_init:x_end, y_init:y_end] = 1

        # * add to rectangle obstacle list to visualize
        self._rectangle_obstacles.append(Map.RectangleObstacle(center, width, height))

    def convert_to_torch(self):
        self._map_torch = torch.from_numpy(self._map).to(self._device, self._dtype)
        return self._map_torch

    def compute_cost(self, x: torch.Tensor) -> torch.Tensor:
        assert self._map_torch is not None

        if x.device != self._device or x.dtype != self._dtype:
            x = x.to(self._device, self._dtype)

        # print(x.shape)
        # print(self._torch_cell_map_origin.shape)
        x_occ = (x / self._cell_size) + self._torch_cell_map_origin
        x_occ = torch.round(x_occ).long().to(self._device)

        is_out_of_bound = torch.logical_or(
            torch.logical_or(
                x_occ[..., 0] < 0, x_occ[..., 0] >= self._map_torch.shape[0]
            ),
            torch.logical_or(
                x_occ[..., 1] < 0, x_occ[..., 1] >= self._map_torch.shape[1]
            ),
        )

        x_occ[..., 0] = torch.clamp(x_occ[..., 0], 0, self._map_torch.shape[0] - 1)
        x_occ[..., 1] = torch.clamp(x_occ[..., 1], 0, self._map_torch.shape[1] - 1)

        collisions = self._map_torch[x_occ[..., 0], x_occ[..., 1]]

        collisions[is_out_of_bound] = 1.0

        return collisions

    def render(self, ax, zorder: int = 0) -> None:
        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)
        ax.set_aspect("equal")

        # render circle obstacles
        for circle_obs in self._circle_obstacles:
            ax.add_patch(
                plt.Circle(
                    circle_obs.center, circle_obs.radius, color="gray", zorder=zorder
                )
            )

        # render rectangle obstacles
        for rectangle_obs in self._rectangle_obstacles:
            ax.add_patch(
                plt.Rectangle(
                    rectangle_obs.center
                    - np.array([rectangle_obs.width / 2, rectangle_obs.height / 2]),
                    rectangle_obs.width,
                    rectangle_obs.height,
                    color="gray",
                    zorder=zorder,
                )
            )


class Environment:
    def __init__(
        self, device=torch.device("cuda"), dtype=torch.float32, seed: int = 42
    ) -> None:
        # * device
        if torch.cuda.is_available() and device == torch.device("cuda"):
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        # * dtype
        self._dtype = dtype
        self._seed = seed

        # * start and goal position
        self._start_pos = torch.tensor(
            [-9.0, -9.0], device=self._device, dtype=self._dtype
        )
        self._goal_pos = torch.tensor(
            [9.0, 9.0], device=self._device, dtype=self._dtype
        )

        # * map
        self._map = Map(device=self._device, dtype=self._dtype)
        generate_random_obstacles(
            self._map,
            random_x_range=(-8.0, 8.0),
            random_y_range=(-8.0, 8.0),
            num_circle_obs=0,
            radius_range=(1, 1),
            num_rectangle_obs=10,
            width_range=(2, 2),
            height_range=(2, 2),
            max_iteration=1000,
            seed=self._seed,
        )
        self._map.convert_to_torch()

        # * Robot state
        self._robot_state = torch.zeros(3, device=self._device, dtype=self._dtype)
        self._robot_state[:2] = self._start_pos
        self._robot_state[2] = angle_normalize(
            torch.atan2(
                self._goal_pos[1] - self._robot_state[1],
                self._goal_pos[0] - self._robot_state[0],
            )
        )

        # * Control input: u = [v, omega] (m/s, rad/s)
        self.u_min = torch.tensor([0.0, -1.0], device=self._device, dtype=self._dtype)
        self.u_max = torch.tensor([2.0, 1.0], device=self._device, dtype=self._dtype)

    def reset(self):
        self._robot_state[:2] = self._start_pos
        self._robot_state[2] = angle_normalize(
            torch.atan2(
                self._goal_pos[1] - self._robot_state[1],
                self._goal_pos[0] - self._robot_state[0],
            )
        )

        self._fig = plt.figure(layout="tight")
        self._ax = self._fig.add_subplot()
        self._ax.set_xlim(-10, 10)
        self._ax.set_ylim(-10, 10)
        self._ax.set_aspect("equal")

        self._rendered_frames = []

        return self._robot_state

    def render(
        self,
        predicted_trajectory: torch.Tensor = None,
        is_collisions: torch.Tensor = None,
        top_samples: Tuple[torch.Tensor, torch.Tensor] = None,
        mode: str = "human",
    ) -> None:
        self._ax.set_xlabel("x [m]")
        self._ax.set_ylabel("y [m]")

        self._map.render(self._ax, zorder=10)

        self._ax.scatter(
            self._start_pos[0].item(),
            self._start_pos[1].item(),
            marker="o",
            color="red",
            zorder=10,
        )

        self._ax.scatter(
            self._goal_pos[0].item(),
            self._goal_pos[1].item(),
            marker="o",
            color="orange",
            zorder=10,
        )

        self._ax.scatter(
            self._robot_state[0].item(),
            self._robot_state[1].item(),
            marker="o",
            color="green",
            zorder=100,
        )

        if top_samples is not None:
            top_samples, top_weights = top_samples
            top_samples = top_samples.cpu().numpy()
            top_weights = top_weights.cpu().numpy()
            top_weights = 0.7 * top_weights / np.max(top_weights)
            top_weights = np.clip(top_weights, 0.1, 0.7)
            # print(top_samples.shape)
            for i in range(top_samples.shape[0]):
                self._ax.plot(
                    top_samples[i, :, 0],
                    top_samples[i, :, 1],
                    color="lightblue",
                    alpha=top_weights[i],
                    zorder=1,
                )

        if predicted_trajectory is not None:
            # if is collision color is red
            colors = np.array(["darkblue"] * predicted_trajectory.shape[1])
            if is_collisions is not None:
                is_collisions = is_collisions.cpu().numpy()
                is_collisions = np.any(is_collisions, axis=0)
                colors[is_collisions] = "red"

            self._ax.scatter(
                predicted_trajectory[0, :, 0].cpu().numpy(),
                predicted_trajectory[0, :, 1].cpu().numpy(),
                color=colors,
                marker="o",
                s=3,
                zorder=2,
            )

        if mode == "human":
            # online rendering
            plt.pause(0.001)
            plt.cla()
        elif mode == "rgb_array":
            # offline rendering for video
            # TODO: high resolution rendering
            self._fig.canvas.draw()
            data = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
            plt.cla()
            self._rendered_frames.append(data)

    def close(self, path: str = None) -> None:
        if path is None:
            # mkdir video if not exists
            if not os.path.exists("video"):
                os.mkdir("video")
            path = "video/" + "navigation_2d_" + str(self._seed) + ".gif"

        if len(self._rendered_frames) > 0:
            # save animation
            clip = ImageSequenceClip(self._rendered_frames, fps=10)
            # clip.write_videofile(path, fps=10)
            clip.write_gif(path, fps=10)

    def dynamics(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        delta_t: float = 0.1,
    ) -> torch.Tensor:
        x = state[:, 0].view(-1, 1)
        y = state[:, 1].view(-1, 1)
        theta = state[:, 2].view(-1, 1)

        v = torch.clamp(action[:, 0].view(-1, 1), self.u_min[0], self.u_max[0])
        omega = torch.clamp(action[:, 1].view(-1, 1), self.u_min[1], self.u_max[1])
        theta = angle_normalize(theta)

        new_x = x + v * torch.cos(theta) * delta_t
        new_y = y + v * torch.sin(theta) * delta_t
        new_theta = angle_normalize(theta + omega * delta_t)

        x_lim = torch.tensor(self._map.x_lim, device=self._device, dtype=self._dtype)
        y_lim = torch.tensor(self._map.y_lim, device=self._device, dtype=self._dtype)

        clamped_x = torch.clamp(new_x, x_lim[0], x_lim[1])
        clamped_y = torch.clamp(new_y, y_lim[0], y_lim[1])

        new_state = torch.cat([clamped_x, clamped_y, new_theta], dim=1)

        return new_state

    def stage_cost(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        goal_cost = torch.norm(state[:, :2] - self._goal_pos, dim=1)
        pos_batch = state[:, :2].unsqueeze(1)  # (batch_size, 1, 2)
        obstacle_cost = self._map.compute_cost(pos_batch).squeeze(1)

        cost = goal_cost + 10000 * obstacle_cost

        return cost

    def terminal_cost(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        zero_action = torch.zeros_like(state[:, 2])
        return self.stage_cost(state=state, action=torch.zeros_like(zero_action))

    def step(
        self,
        u: torch.Tensor,
    ) -> Tuple[torch.Tensor, bool]:
        u = torch.clamp(u, self.u_min, self.u_max)

        self._robot_state = self.dynamics(
            state=self._robot_state.unsqueeze(0), action=u.unsqueeze(0)
        ).squeeze(0)

        goal_threthold = 0.5
        is_goal_reached = (
            torch.norm(self._robot_state[:2] - self._goal_pos) < goal_threthold
        )
        return self._robot_state, is_goal_reached

    def collision_check(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        pos_batch = state[:, :, :2]
        is_collisions = self._map.compute_cost(pos_batch).squeeze(1)
        return is_collisions


def generate_random_obstacles(
    obstacle_map: Map,
    random_x_range: Tuple[float, float],
    random_y_range: Tuple[float, float],
    num_circle_obs: int,
    radius_range: Tuple[float, float],
    num_rectangle_obs: int,
    width_range: Tuple[float, float],
    height_range: Tuple[float, float],
    max_iteration: int,
    seed: int,
) -> None:
    """
    Generate random obstacles in the map
    """
    rng = np.random.default_rng(seed)

    # * if random range is larger than map size, use map size
    if random_x_range[0] < obstacle_map.x_lim[0]:
        random_x_range[0] = obstacle_map.x_lim[0]
    if random_x_range[1] > obstacle_map.x_lim[1]:
        random_x_range[1] = obstacle_map.x_lim[1]
    if random_y_range[0] < obstacle_map.y_lim[0]:
        random_y_range[0] = obstacle_map.y_lim[0]
    if random_y_range[1] > obstacle_map.y_lim[1]:
        random_y_range[1] = obstacle_map.y_lim[1]

    for _ in range(num_circle_obs):
        num_trial = 0
        while num_trial < max_iteration:
            center_x = rng.uniform(random_x_range[0], random_x_range[1])
            center_y = rng.uniform(random_y_range[0], random_y_range[1])
            center = np.array([center_x, center_y])
            radius = rng.uniform(radius_range[0], radius_range[1])

            # overlap check
            is_overlap = False
            for circle_obs in obstacle_map._circle_obstacles:
                if (
                    np.linalg.norm(circle_obs.center - center)
                    <= circle_obs.radius + radius
                ):
                    is_overlap = True

            for rectangle_obs in obstacle_map._rectangle_obstacles:
                if (
                    np.linalg.norm(rectangle_obs.center - center)
                    <= rectangle_obs.width / 2 + radius
                ):
                    if (
                        np.linalg.norm(rectangle_obs.center - center)
                        <= rectangle_obs.height / 2 + radius
                    ):
                        is_overlap = True

            if not is_overlap:
                break

            num_trial += 1

            if num_trial == max_iteration:
                raise RuntimeError(
                    "Cannot generate random obstacles due to reach max iteration."
                )

        obstacle_map.add_circle_obstacle(center, radius)

    for _ in range(num_rectangle_obs):
        num_trial = 0
        while num_trial < max_iteration:
            center_x = rng.uniform(random_x_range[0], random_x_range[1])
            center_y = rng.uniform(random_y_range[0], random_y_range[1])
            center = np.array([center_x, center_y])
            width = rng.uniform(width_range[0], width_range[1])
            height = rng.uniform(height_range[0], height_range[1])

            # overlap check
            is_overlap = False
            for circle_obs in obstacle_map._circle_obstacles:
                if (
                    np.linalg.norm(circle_obs.center - center)
                    <= circle_obs.radius + width / 2
                ):
                    if (
                        np.linalg.norm(circle_obs.center - center)
                        <= circle_obs.radius + height / 2
                    ):
                        is_overlap = True

            for rectangle_obs in obstacle_map._rectangle_obstacles:
                if (
                    np.linalg.norm(rectangle_obs.center - center)
                    <= rectangle_obs.width / 2 + width / 2
                ):
                    if (
                        np.linalg.norm(rectangle_obs.center - center)
                        <= rectangle_obs.height / 2 + height / 2
                    ):
                        is_overlap = True

            if not is_overlap:
                break

            num_trial += 1

            if num_trial == max_iteration:
                raise RuntimeError(
                    "Cannot generate random obstacles due to reach max iteration."
                )

        obstacle_map.add_rectangle_obstacle(center, width, height)


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


save_mode = True


def main():
    # * environment
    env = Environment()

    # * solver
    solver = MPPI(
        horizon=50,
        num_samples=1000,
        dim_state=3,
        dim_control=2,
        dynamics=env.dynamics,
        stage_cost=env.stage_cost,
        terminal_cost=env.terminal_cost,
        u_min=env.u_min,
        u_max=env.u_max,
        sigmas=torch.Tensor([0.5, 0.5]),
        lambda_=1.0,
    )

    state = env.reset()

    max_steps = 500
    # average_time = 0

    for i in range(max_steps):
        # start = time.time()
        with torch.no_grad():
            action_seq, state_seq = solver.solve(state)
        # end = time.time()
        # average_time += end - start

        state, is_goal_reached = env.step(action_seq[0, :])

        is_collisions = env.collision_check(state=state_seq)

        top_samples, top_weights = solver.get_top_samples(num_samples=100)

        if save_mode:
            env.render(
                predicted_trajectory=state_seq,
                is_collisions=is_collisions,
                top_samples=(top_samples, top_weights),
                mode="rgb_array",
            )
            if i == 0:
                pbar = tqdm.tqdm(total=max_steps, desc="recording video")
            pbar.update(1)

        else:
            env.render(
                predicted_trajectory=state_seq,
                is_collisions=is_collisions,
                top_samples=(top_samples, top_weights),
                mode="human",
            )

        if is_goal_reached:
            print("Goal Reached!")
            break

    env.close()


if __name__ == "__main__":
    main()
