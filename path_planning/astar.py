"""
A* (A-Star) Algorithm – Path Planning Visualization

Finds the shortest path on a 2-D grid using a heuristic search.
Uses the Euclidean distance as the heuristic (admissible for 8-connected
grids with diagonal cost √2).
"""

import heapq
import numpy as np
import matplotlib.pyplot as plt


def build_grid(rows=25, cols=25, seed=5):
    """Return a binary occupancy grid matching the Dijkstra example."""
    rng = np.random.default_rng(seed)
    grid = (rng.random((rows, cols)) < 0.25).astype(int)
    grid[0, 0] = 0
    grid[rows - 1, cols - 1] = 0
    return grid


def heuristic(a, b):
    """Euclidean distance heuristic."""
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def astar(grid, start, goal):
    """
    Run A* on *grid*.

    Returns
    -------
    path        : list of (row, col) from start to goal, or [] if no path
    visited_map : 2-D array with the step at which each cell was closed
    """
    rows, cols = grid.shape
    g_cost = np.full((rows, cols), np.inf)
    g_cost[start] = 0
    prev = np.full((rows, cols, 2), -1, dtype=int)
    visited_map = np.zeros((rows, cols), dtype=int)

    # Priority queue entries: (f_cost, g_cost, (r, c))
    pq = [(heuristic(start, goal), 0, start)]
    step = 0

    while pq:
        f, g, (r, c) = heapq.heappop(pq)
        if g > g_cost[r, c]:
            continue
        step += 1
        visited_map[r, c] = step

        if (r, c) == goal:
            break

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                move_cost = 1.0 if abs(dr) + abs(dc) == 1 else np.sqrt(2)
                new_g = g + move_cost
                if new_g < g_cost[nr, nc]:
                    g_cost[nr, nc] = new_g
                    prev[nr, nc] = [r, c]
                    f_new = new_g + heuristic((nr, nc), goal)
                    heapq.heappush(pq, (f_new, new_g, (nr, nc)))

    # Reconstruct path
    path = []
    cur = list(goal)
    while cur != [-1, -1]:
        path.append(tuple(cur))
        pr, pc = prev[cur[0], cur[1]]
        if pr == -1:
            break
        cur = [pr, pc]
    path.reverse()

    if path and path[0] != start:
        return [], visited_map
    return path, visited_map


def visualize(grid, path, visited_map, title="A* Algorithm"):
    rows, cols = grid.shape
    img = np.ones((rows, cols, 3))

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 1:
                img[r, c] = [0.25, 0.25, 0.25]

    max_step = visited_map.max() or 1
    for r in range(rows):
        for c in range(cols):
            s = visited_map[r, c]
            if s > 0 and grid[r, c] == 0:
                t = s / max_step
                # Yellow-green gradient for A* explored cells
                img[r, c] = [0.8 + 0.2 * (1 - t), 1.0, 0.6 * (1 - t)]

    for (r, c) in path:
        img[r, c] = [1.0, 0.55, 0.0]

    sr, sc = (0, 0)
    gr, gc = (rows - 1, cols - 1)
    img[sr, sc] = [0.0, 0.75, 0.0]
    img[gr, gc] = [0.85, 0.1, 0.1]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, origin="upper")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=[0.25, 0.25, 0.25], label="Obstacle"),
        Patch(facecolor=[0.8, 1.0, 0.6], label="Explored"),
        Patch(facecolor=[1.0, 0.55, 0.0], label="Path"),
        Patch(facecolor=[0.0, 0.75, 0.0], label="Start"),
        Patch(facecolor=[0.85, 0.1, 0.1], label="Goal"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    path_len = sum(
        np.sqrt((path[i + 1][0] - path[i][0]) ** 2 + (path[i + 1][1] - path[i][1]) ** 2)
        for i in range(len(path) - 1)
    ) if len(path) > 1 else 0

    ax.text(
        0.02, 0.02,
        f"Path length: {path_len:.2f}  |  Cells explored: {(visited_map > 0).sum()}",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

    plt.tight_layout()
    return fig


def main():
    grid = build_grid(rows=25, cols=25, seed=5)
    start = (0, 0)
    goal = (24, 24)

    path, visited_map = astar(grid, start, goal)

    fig = visualize(grid, path, visited_map, title="A* Algorithm – Path Planning")
    plt.savefig("astar_result.png", dpi=120, bbox_inches="tight")
    plt.show()

    if path:
        print(f"Path found! Length: {len(path)} cells, "
              f"Explored: {(visited_map > 0).sum()} cells.")
    else:
        print("No path found.")


if __name__ == "__main__":
    main()
