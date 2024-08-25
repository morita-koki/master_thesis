import os
from typing import Tuple

import torch
import numpy as np
from matplotlib import pyplot as plt
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from map import Map


@torch.jit.script
def angle_normalize(x):
    """
    convert the angle to the range of [-pi, pi]
    """
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi


class Environment:
    def __init__(
        self,
        device=torch.device("cuda"),
        dtype=torch.float32,
        seed: int = 42,
        map_path: str = None,
        cell_size: float = 0.01,
    ) -> None:
        # * device
        if torch.cuda.is_available() and device == torch.device("cuda"):
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        # * dtype
        self._dtype = dtype
        self._seed = seed

        # * map
        self._map = Map(device=self._device, dtype=self._dtype)
        if map_path is None:
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
        else:
            self._map.load_from_png(path=map_path, cell_size=cell_size)
        self._map.convert_to_torch()

        # * start and goal position
        self._start_pos = torch.tensor(
            [self._map.x_lim[0] * 0.9, self._map.y_lim[0] * 0.9],
            device=self._device,
            dtype=self._dtype,
        )
        self._goal_pos = torch.tensor(
            [self._map.x_lim[1] * 0.9, self._map.y_lim[1] * 0.9],
            device=self._device,
            dtype=self._dtype,
        )

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
            path = "video/" + "output" + str(self._seed) + ".gif"
        else:
            path = os.path.join(os.path.dirname(__file__), path)

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
