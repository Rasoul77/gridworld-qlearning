import argparse
from typing import Generator, Optional

import matplotlib.pyplot as plt

from actions import Action
from grid_world import Rewards, GridWorld, GridWorldConfig
from q_learning import QLearningAgent, QLearningParams
from visualization import GridWorldViewer, FuncAnimation, TrainFrame


def training_frames(
    env: GridWorld,
    agent: QLearningAgent,
    *,
    episodes: int,
    max_steps_per_episode: int,
) -> Generator[TrainFrame, None, None]:
    """
    Generator that performs Q-learning step-by-step and yields frames for animation.
    """
    s = env.reset()
    ep = 1
    step_in_ep = 0
    total_r = 0.0
    last_a: Optional[Action] = None
    last_r: Optional[float] = None

    # Yield an initial frame
    yield TrainFrame(ep, step_in_ep, total_r, agent.epsilon, env.agent_pos, last_a, last_r, done=False)

    while ep <= episodes:
        # Choose action
        a = agent.select_action(s, greedy=False)
        s2, r, done = env.step(a)

        # Update Q
        agent.update(s, a, r, s2, done)

        step_in_ep += 1
        total_r += r
        last_a = a
        last_r = r

        yield TrainFrame(ep, step_in_ep, total_r, agent.epsilon, env.agent_pos, last_a, last_r, done=done)

        if done or step_in_ep >= max_steps_per_episode:
            # Start next episode
            ep += 1
            if ep > episodes:
                break
            s = env.reset()
            step_in_ep = 0
            total_r = 0.0
            last_a = None
            last_r = None
            yield TrainFrame(ep, step_in_ep, total_r, agent.epsilon, env.agent_pos, last_a, last_r, done=False)
        else:
            s = s2


def main() -> None:
    parser = argparse.ArgumentParser(description="GridWorld Q-learning with interactive hover Q-values.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--interval-ms", type=int, default=25, help="Animation interval in milliseconds.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slip", type=float, default=0.00, help="Slip probability (0..1).")
    args = parser.parse_args()

    # Grid world definition
    ascii_map = [
        "S...#.......",
        ".##.#.###..P",
        "....#...#...",
        ".####.#.#.##",
        "......P#....",
        "###..#.#.###",
        ".....#...#..",
        ".###.#####..",
        ".#.........#",
        "..#.###..#G.",
    ]

    cfg = GridWorldConfig.from_ascii(
        ascii_map,
        slip_prob=args.slip,
        rewards=Rewards(step=-0.04, wall_bump=-0.20, goal=+1.00, pit=-1.00),
    )

    env = GridWorld(cfg, seed=args.seed)

    params = QLearningParams(
        alpha=0.20,
        gamma=0.98,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.9992,
    )
    agent = QLearningAgent(cfg.rows, cfg.cols, params, seed=args.seed)

    viewer = GridWorldViewer(env, agent)

    frames = training_frames(env, agent, episodes=args.episodes, max_steps_per_episode=args.max_steps)

    def _animate(frame: TrainFrame) -> None:
        # Pause handling (keeps UI responsive)
        if viewer.paused:
            return
        viewer.update(frame)

    animator = FuncAnimation(
        viewer.fig,
        _animate,
        frames=frames,
        interval=args.interval_ms,
        repeat=False,
        blit=False,
        cache_frame_data=False,
    )

    plt.show()


if __name__ == "__main__":
    main()
