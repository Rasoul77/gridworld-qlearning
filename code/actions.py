# This file is covered by the LICENSE file in the root of this project.

from enum import IntEnum
from typing import Dict, Tuple

Pos = Tuple[int, int]  # (row, col)


class Action(IntEnum):
    """Four-neighborhood actions."""
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    @staticmethod
    def all() -> Tuple["Action", "Action", "Action", "Action"]:
        return (Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT)


ACTION_TO_DELTA: Dict[Action, Pos] = {
    Action.UP: (-1, 0),
    Action.RIGHT: (0, +1),
    Action.DOWN: (+1, 0),
    Action.LEFT: (0, -1),
}

ACTION_NAME: Dict[Action, str] = {
    Action.UP: "U",
    Action.RIGHT: "R",
    Action.DOWN: "D",
    Action.LEFT: "L",
}
