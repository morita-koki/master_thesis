import os
import math
from dataclasses import dataclass
from typing import Tuple, List

import torch
import numpy as np
import matplotlib.pyplot as plt


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

    def load_from_png(self, path: str, cell_size: float = 0.01) -> None:
        """
        Load map from png file
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} does not exist.")

        if not path.endswith(".png"):
            raise ValueError(f"{path} is not a png file.")

        print(f"Load map from {path}")

        self._cell_size = cell_size

        img = plt.imread(path)
        self._origin_map = img.copy()
        # white is free space, black is obstacle
        img_r = np.fliplr(img[:, :, 0].T)  # only use one channel
        self._map = img_r.copy()
        self._map[img_r == 0] = 1
        self._map[img_r != 0] = 0
        self._map.astype(int)

        cell_map_dim = self._map.shape
        self._cell_map_origin = np.array(
            [cell_map_dim[0] / 2, cell_map_dim[1] / 2]
        ).astype(int)

        self._torch_cell_map_origin = torch.from_numpy(self._cell_map_origin).to(
            self._device, self._dtype
        )

        x_range = self._cell_size * self._map.shape[0]
        y_range = self._cell_size * self._map.shape[1]
        self.x_lim = [-x_range / 2, x_range / 2]  # [m]
        self.y_lim = [-y_range / 2, y_range / 2]  # [m]

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

        # render map
        ax.imshow(
            # (-1) * self._map.T,
            self._origin_map,
            cmap="gray",
            extent=[
                self.x_lim[0],
                self.x_lim[1],
                self.y_lim[0],
                self.y_lim[1],
            ],
            zorder=0,
        )

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
