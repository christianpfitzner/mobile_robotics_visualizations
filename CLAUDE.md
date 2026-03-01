# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Visualizations

Each module is fully self-contained. Run any script directly:

```bash
python icp/icp.py
python ransac/ransac.py
python least_squares/least_squares.py
python path_planning/dijkstra.py
python path_planning/astar.py
python kalman_filter/kalman_filter.py
```

Each script opens a matplotlib window and saves a `*_result.png` in the **script's own directory** (the save path is relative to the working directory, so run from the repo root or the module's directory accordingly).

## Dependencies

```bash
pip install numpy matplotlib
```

No other dependencies. No build step, no test framework, no package manifest.

## Architecture

The repo is a flat collection of independent algorithm demonstrations — there is no shared library, no imports between modules, and no central entry point.

```
icp/                  — Iterative Closest Point (2-D point cloud alignment)
ransac/               — RANSAC robust line fitting
least_squares/        — Polynomial / plane / hyperplane LS regression (1-D, 2-D, 3-D)
path_planning/        — A* and Dijkstra on a 25×25 occupancy grid
kalman_filter/        — 1-D linear Kalman filter (position + velocity state)
```

### Per-module structure

Every module follows the same pattern:

1. **Pure algorithm functions** — stateless, NumPy-only, no matplotlib dependency.
2. **`main()`** — generates synthetic data, calls the algorithm, builds and saves the figure.
3. `if __name__ == "__main__": main()` guard.

The path-planning pair (`astar.py` / `dijkstra.py`) share the same `build_grid(seed=5)` call so they operate on an identical grid, making the comparison meaningful.

### Key algorithmic details

- **ICP** (`icp/icp.py`): SVD-based rigid transform; brute-force nearest-neighbour (`O(n²)`); convergence tracked by mean point-to-point distance.
- **RANSAC** (`ransac/ransac.py`): Fits the line as `ax + by + c = 0` (normalised implicit form); refits final model using OLS over all inliers.
- **Least squares** (`least_squares/least_squares.py`): Uses `np.linalg.lstsq` (not `np.polyfit`) for the 2-D and 3-D cases; Vandermonde matrix for the 1-D polynomial case.
- **Path planning** (`path_planning/`): 8-connected grid; diagonal moves cost `√2`; A* uses Euclidean distance heuristic.
- **Kalman filter** (`kalman_filter/kalman_filter.py`): Discrete white-noise acceleration process model; state `[position, velocity]`; position-only measurement.
