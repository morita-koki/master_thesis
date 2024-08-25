import os

import torch
import yaml
import tqdm
import fire

from environment import Environment
from mppi import MPPI


def main(config: str = "conf.yaml"):
    with open(config, "r") as fp:
        conf = yaml.load(fp, Loader=yaml.SafeLoader)

    # * environment
    env = Environment(
        seed=conf["environment"]["seed"],
        map_path=os.path.join(
            os.path.dirname(__file__), conf["environment"]["map"]["path"]
        ),
        cell_size=conf["environment"]["map"]["cell_size"],
    )

    # * solver
    solver = MPPI(
        horizon=conf["mppi"]["horizon"],
        num_samples=conf["mppi"]["num_samples"],
        dim_state=3,
        dim_control=2,
        dynamics=env.dynamics,
        stage_cost=env.stage_cost,
        terminal_cost=env.terminal_cost,
        u_min=env.u_min,
        u_max=env.u_max,
        sigmas=torch.Tensor(conf["mppi"]["sigmas"]),
        lambda_=conf["mppi"]["lambda"],
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

        if conf["save"]:
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

    env.close(path=conf["output_gif"])


if __name__ == "__main__":
    fire.Fire(main)
