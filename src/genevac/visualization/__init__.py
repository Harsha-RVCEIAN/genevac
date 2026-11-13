"""Live simulation visualizer, trajectory charts, and heatmaps."""

from typing import List, Tuple
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np


def plot_trajectories(trajectories: List[List[Tuple[float, float]]], title: str = "Pedestrian Trajectories") -> Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    for traj in trajectories:
        xs = [pt[0] for pt in traj]
        ys = [pt[1] for pt in traj]
        ax.plot(xs, ys, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    return fig


def plot_heatmap(density_matrix: np.ndarray, title: str = "Congestion Heatmap") -> Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(density_matrix, cmap="hot", interpolation="nearest", origin="lower")
    fig.colorbar(cax, ax=ax)
    ax.set_title(title)
    return fig
