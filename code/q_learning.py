import random

import numpy as np
from dataclasses import dataclass

from actions import *


@dataclass
class QLearningParams:
    alpha: float = 0.20      # learning rate
    gamma: float = 0.98      # discount factor
    eps_start: float = 1.00  # exploration start
    eps_end: float = 0.05    # exploration floor
    eps_decay: float = 0.999 # per-step decay (close to 1 = slow decay)


class QLearningAgent:
    """Tabular Q-learning agent with epsilon-greedy policy."""
    def __init__(self, rows: int, cols: int, params: QLearningParams, *, seed: int = 0) -> None:
        self.rows = rows
        self.cols = cols
        self.params = params
        self.rng = random.Random(seed)

        # Q-table: (row, col, action)
        self.q = np.zeros((rows, cols, len(Action.all())), dtype=np.float32)
        self.epsilon = params.eps_start

    def select_action(self, state: Pos, *, greedy: bool = False) -> Action:
        r, c = state
        if (not greedy) and (self.rng.random() < self.epsilon):
            return self.rng.choice(list(Action.all()))
        # Greedy with random tie-breaking
        qvals = self.q[r, c, :]
        maxv = float(np.max(qvals))
        best = np.flatnonzero(np.isclose(qvals, maxv))
        return Action(int(self.rng.choice(best)))

    def update(self, s: Pos, a: Action, r: float, s2: Pos, done: bool) -> None:
        sr, sc = s
        nr, nc = s2
        a_i = int(a)

        q_sa = float(self.q[sr, sc, a_i])
        if done:
            target = r
        else:
            target = r + self.params.gamma * float(np.max(self.q[nr, nc, :]))

        self.q[sr, sc, a_i] = (1.0 - self.params.alpha) * q_sa + self.params.alpha * target

        # Decay epsilon per update (step)
        self.epsilon = max(self.params.eps_end, self.epsilon * self.params.eps_decay)

    def greedy_policy(self) -> np.ndarray:
        """Return (rows, cols) of best actions (ties broken arbitrarily)."""
        return np.argmax(self.q, axis=2).astype(np.int32)
