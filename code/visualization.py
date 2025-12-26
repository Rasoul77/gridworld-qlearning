import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrow, Rectangle
from matplotlib.animation import FuncAnimation

from actions import *
from grid_world import *
from q_learning import *


@dataclass
class TrainFrame:
    episode: int
    step_in_ep: int
    total_reward_ep: float
    epsilon: float
    agent_pos: Pos
    last_action: Optional[Action]
    last_reward: Optional[float]
    done: bool


class GridWorldViewer:
    """
    Matplotlib visualization with:
    - cell coloring (start/goal/walls/pits)
    - moving agent
    - optional greedy-policy arrows
    - hover popup displaying Q-values for actions at the hovered cell
    """
    def __init__(self, env: GridWorld, agent: QLearningAgent) -> None:
        self.env = env
        self.agent = agent

        self.show_policy_arrows: bool = True
        self.paused: bool = False

        self.fig: Figure
        self.ax: Axes
        self.fig, self.ax = plt.subplots(figsize=(8.0, 7.2))
        self.ax.set_aspect("equal")

        self._cell_patches: Dict[Pos, Rectangle] = {}
        self._agent_patch: Optional[Rectangle] = None
        self._policy_arrows: List[FancyArrow] = []

        # Hover annotation
        self._hover_annot = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.95),
            arrowprops=dict(arrowstyle="->", lw=1),
        )
        self._hover_annot.set_visible(False)

        self._text_status = self.ax.text(
            0.02, 1.02, "",
            transform=self.ax.transAxes,
            ha="left", va="bottom",
            fontsize=10
        )

        self._init_plot()
        self._connect_events()

    def _init_plot(self) -> None:
        cfg = self.env.cfg
        self.ax.set_xlim(0, cfg.cols)
        self.ax.set_ylim(cfg.rows, 0)  # invert y-axis so row 0 is top
        self.ax.set_xticks(range(cfg.cols + 1))
        self.ax.set_yticks(range(cfg.rows + 1))
        self.ax.grid(True, which="both", linewidth=0.6)
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])

        # Draw cells
        for r in range(cfg.rows):
            for c in range(cfg.cols):
                pos = (r, c)
                face = self._cell_facecolor(pos)
                rect = Rectangle((c, r), 1.0, 1.0, facecolor=face, edgecolor="black", linewidth=0.5)
                self.ax.add_patch(rect)
                self._cell_patches[pos] = rect

        # Agent patch (a smaller square inside the cell)
        ar, ac = self.env.agent_pos
        self._agent_patch = Rectangle(
            (ac + 0.15, ar + 0.15),
            0.70, 0.70,
            facecolor="#FFD54F", edgecolor="black", linewidth=1.2
        )
        self.ax.add_patch(self._agent_patch)

        # Legend-ish hints (minimal)
        self.ax.text(0.02, -0.04, "Space: pause/resume   |   P: toggle greedy policy arrows",
                     transform=self.ax.transAxes, ha="left", va="top", fontsize=9)

        self._draw_policy_arrows()

    def _cell_facecolor(self, pos: Pos) -> str:
        cfg = self.env.cfg
        if pos in set(cfg.walls):
            return "#2f2f2f"  # walls
        if pos == cfg.start:
            return "#90CAF9"  # start
        if pos == cfg.goal:
            return "#A5D6A7"  # goal
        if pos in set(cfg.pits):
            return "#EF9A9A"  # pit
        return "#FAFAFA"      # empty

    def _connect_events(self) -> None:
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_key(self, event) -> None:
        if event.key == " ":
            self.paused = not self.paused
        elif event.key and event.key.lower() == "p":
            self.show_policy_arrows = not self.show_policy_arrows
            self._draw_policy_arrows()

    def _on_hover(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            self._hover_annot.set_visible(False)
            self.fig.canvas.draw_idle()
            return

        col = int(math.floor(event.xdata))
        row = int(math.floor(event.ydata))
        cfg = self.env.cfg

        if not (0 <= row < cfg.rows and 0 <= col < cfg.cols):
            self._hover_annot.set_visible(False)
            self.fig.canvas.draw_idle()
            return

        pos = (row, col)
        # Don't bother showing Q-values for walls
        if pos in set(cfg.walls):
            self._hover_annot.set_visible(False)
            self.fig.canvas.draw_idle()
            return

        qvals = self.agent.q[row, col, :]
        best_a = int(np.argmax(qvals))

        txt = (
            f"Cell ({row},{col})\n"
            f"Q(U)={qvals[Action.UP]:+.3f}\n"
            f"Q(R)={qvals[Action.RIGHT]:+.3f}\n"
            f"Q(D)={qvals[Action.DOWN]:+.3f}\n"
            f"Q(L)={qvals[Action.LEFT]:+.3f}\n"
            f"Greedy: {ACTION_NAME[Action(best_a)]}"
        )

        self._hover_annot.xy = (col + 0.5, row + 0.5)
        self._hover_annot.set_text(txt)
        self._hover_annot.set_visible(True)
        self.fig.canvas.draw_idle()

    def _clear_policy_arrows(self) -> None:
        for a in self._policy_arrows:
            a.remove()
        self._policy_arrows.clear()

    def _draw_policy_arrows(self) -> None:
        self._clear_policy_arrows()

        if not self.show_policy_arrows:
            self.fig.canvas.draw_idle()
            return

        cfg = self.env.cfg
        policy = self.agent.greedy_policy()
        walls = set(cfg.walls)
        terminals = set(cfg.pits) | {cfg.goal}

        # Draw small arrows indicating the current greedy action
        for r in range(cfg.rows):
            for c in range(cfg.cols):
                pos = (r, c)
                if pos in walls or pos in terminals:
                    continue

                a = Action(int(policy[r, c]))
                dx, dy = 0.0, 0.0
                if a == Action.UP:
                    dx, dy = 0.0, -0.28
                elif a == Action.RIGHT:
                    dx, dy = +0.28, 0.0
                elif a == Action.DOWN:
                    dx, dy = 0.0, +0.28
                elif a == Action.LEFT:
                    dx, dy = -0.28, 0.0

                arrow = FancyArrow(
                    c + 0.5, r + 0.5, dx, dy,
                    width=0.05, head_width=0.18, head_length=0.14,
                    length_includes_head=True,
                    alpha=0.55,
                    edgecolor="black",
                    facecolor="black",
                    linewidth=0.8,
                )
                self.ax.add_patch(arrow)
                self._policy_arrows.append(arrow)

        self.fig.canvas.draw_idle()

    def update(self, frame: TrainFrame) -> None:
        # Move agent patch
        if self._agent_patch is not None:
            r, c = frame.agent_pos
            self._agent_patch.set_xy((c + 0.15, r + 0.15))

        # Status text
        la = "-" if frame.last_action is None else ACTION_NAME[frame.last_action]
        lr = "-" if frame.last_reward is None else f"{frame.last_reward:+.2f}"
        done = "DONE" if frame.done else ""
        self._text_status.set_text(
            f"Episode: {frame.episode:5d}   Step: {frame.step_in_ep:3d}   "
            f"EpReward: {frame.total_reward_ep:+.2f}   "
            f"Epsilon: {frame.epsilon:.3f}   Last: {la}/{lr}   {done}"
        )

        # Update policy arrows occasionally (cheap heuristic)
        # Redraw at episode boundaries so it feels responsive.
        if frame.done:
            self._draw_policy_arrows()
