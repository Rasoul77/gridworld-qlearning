import random

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from actions import *


@dataclass(frozen=True)
class Rewards:
    """
    Reward shaping:
    - step: small negative to encourage shortest paths
    - wall_bump: discourage banging into walls / borders
    - goal: positive terminal reward
    - pit: negative terminal reward
    """
    step: float = -0.04
    wall_bump: float = -0.20
    goal: float = +1.00
    pit: float = -1.00


@dataclass(frozen=True)
class GridWorldConfig:
    rows: int
    cols: int
    start: Pos
    goal: Pos
    walls: Tuple[Pos, ...]
    pits: Tuple[Pos, ...]
    rewards: Rewards = Rewards()
    slip_prob: float = 0.00  # probability of slipping to a random action (0 = deterministic)

    @staticmethod
    def from_ascii(ascii_map: Sequence[str], *, slip_prob: float = 0.0, rewards: Rewards = Rewards()) -> "GridWorldConfig":
        """
        Build a grid from an ASCII map.

        Legend:
          S = start
          G = goal
          # = wall
          P = pit
          . = empty

        Example:
          [
            "S...#....",
            ".##..P..G",
            "....#....",
          ]
        """
        if not ascii_map:
            raise ValueError("ascii_map must be a non-empty list of strings.")

        rows = len(ascii_map)
        cols = len(ascii_map[0])
        if any(len(r) != cols for r in ascii_map):
            raise ValueError("All rows in ascii_map must have the same length.")

        start: Optional[Pos] = None
        goal: Optional[Pos] = None
        walls: List[Pos] = []
        pits: List[Pos] = []

        for r, line in enumerate(ascii_map):
            for c, ch in enumerate(line):
                if ch == "S":
                    start = (r, c)
                elif ch == "G":
                    goal = (r, c)
                elif ch == "#":
                    walls.append((r, c))
                elif ch == "P":
                    pits.append((r, c))
                elif ch == ".":
                    pass
                else:
                    raise ValueError(f"Unknown char '{ch}' at row {r}, col {c}.")

        if start is None or goal is None:
            raise ValueError("Map must contain exactly one 'S' (start) and one 'G' (goal).")

        return GridWorldConfig(
            rows=rows,
            cols=cols,
            start=start,
            goal=goal,
            walls=tuple(walls),
            pits=tuple(pits),
            rewards=rewards,
            slip_prob=float(slip_prob),
        )


class GridWorld:
    """Simple episodic GridWorld with walls and terminal pits/goal."""
    def __init__(self, cfg: GridWorldConfig, *, seed: int = 0) -> None:
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.walls_set = set(cfg.walls)
        self.pits_set = set(cfg.pits)
        self.agent_pos: Pos = cfg.start

    def reset(self) -> Pos:
        self.agent_pos = self.cfg.start
        return self.agent_pos

    def in_bounds(self, pos: Pos) -> bool:
        r, c = pos
        return 0 <= r < self.cfg.rows and 0 <= c < self.cfg.cols

    def is_wall(self, pos: Pos) -> bool:
        return pos in self.walls_set

    def is_terminal(self, pos: Pos) -> bool:
        return pos == self.cfg.goal or pos in self.pits_set

    def step(self, action: Action) -> Tuple[Pos, float, bool]:
        """
        Apply action and return (next_state, reward, done).

        If slip_prob > 0, sometimes action becomes random (a "banana peel" effect).
        """
        # Slip (stochastic action)
        if self.cfg.slip_prob > 0 and self.rng.random() < self.cfg.slip_prob:
            action = self.rng.choice(list(Action.all()))

        dr, dc = ACTION_TO_DELTA[action]
        r, c = self.agent_pos
        nxt = (r + dr, c + dc)

        # Blocked by borders or wall -> stay and penalize
        if (not self.in_bounds(nxt)) or self.is_wall(nxt):
            reward = self.cfg.rewards.wall_bump
            done = False
            nxt = self.agent_pos
        else:
            # Move
            self.agent_pos = nxt
            if nxt == self.cfg.goal:
                reward = self.cfg.rewards.goal
                done = True
            elif nxt in self.pits_set:
                reward = self.cfg.rewards.pit
                done = True
            else:
                reward = self.cfg.rewards.step
                done = False

        return nxt, reward, done

    def valid_state_mask(self) -> np.ndarray:
        """Mask of non-wall cells."""
        mask = np.ones((self.cfg.rows, self.cfg.cols), dtype=bool)
        for (r, c) in self.cfg.walls:
            mask[r, c] = False
        return mask
