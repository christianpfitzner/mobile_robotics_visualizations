"""
1-D Kalman Filter Visualization

Models a 1-D object moving with nearly constant velocity.
The state vector is [position, velocity].  Only the position is measured
(noisy GPS-like sensor).

The Kalman filter fuses the motion model with measurements to produce an
optimal (minimum-variance) state estimate.
"""

import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter1D:
    """
    Linear Kalman filter for 1-D position + velocity tracking.

    State:  x = [position, velocity]ᵀ
    Motion: x_k = F·x_{k-1} + w,   w ~ N(0, Q)
    Measurement: z_k = H·x_k + v,  v ~ N(0, R)
    """

    def __init__(self, dt, process_var, measurement_var):
        """
        Parameters
        ----------
        dt              : time step (seconds)
        process_var     : variance of the process noise (acceleration noise)
        measurement_var : variance of the measurement noise
        """
        self.dt = dt

        # State transition matrix
        self.F = np.array([[1, dt],
                           [0,  1]])

        # Observation matrix (we only measure position)
        self.H = np.array([[1, 0]])

        # Process noise covariance (modelled as discrete white-noise acceleration)
        q = process_var
        self.Q = q * np.array([[dt ** 4 / 4, dt ** 3 / 2],
                                [dt ** 3 / 2, dt ** 2]])

        # Measurement noise covariance
        self.R = np.array([[measurement_var]])

        # Initial state and covariance
        self.x = np.zeros((2, 1))      # [position; velocity]
        self.P = np.eye(2) * 500       # large initial uncertainty

    def predict(self):
        """Prediction step."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy(), self.P.copy()

    def update(self, z):
        """
        Measurement update step.

        Parameters
        ----------
        z : scalar measurement (position)
        """
        z = np.array([[z]])
        S = self.H @ self.P @ self.H.T + self.R          # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)         # Kalman gain
        y = z - self.H @ self.x                           # innovation
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return self.x.copy(), self.P.copy()


def simulate(n_steps=80, dt=0.5, seed=42):
    """
    Simulate a 1-D object with slowly varying velocity and
    return true positions, noisy measurements, and Kalman estimates.
    """
    rng = np.random.default_rng(seed)

    # True trajectory: sinusoidal acceleration → smooth velocity changes
    t = np.arange(n_steps) * dt
    true_velocity = np.sin(0.3 * t) * 2.0        # m/s
    true_position = np.cumsum(true_velocity * dt)  # m

    measurement_std = 3.0    # sensor noise (m)
    process_var = 0.5        # acceleration variance

    measurements = true_position + rng.normal(0, measurement_std, n_steps)

    kf = KalmanFilter1D(dt=dt, process_var=process_var,
                        measurement_var=measurement_std ** 2)
    kf.x = np.array([[measurements[0]], [0.0]])  # initialise near first measurement

    est_positions = []
    est_velocities = []
    covariances = []

    for z in measurements:
        kf.predict()
        x, P = kf.update(z)
        est_positions.append(x[0, 0])
        est_velocities.append(x[1, 0])
        covariances.append(P[0, 0])

    return (t, true_position, true_velocity,
            measurements, np.array(est_positions),
            np.array(est_velocities), np.array(covariances))


def main():
    (t, true_pos, true_vel,
     measurements, est_pos, est_vel, cov) = simulate()

    pos_std = np.sqrt(cov)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle("1-D Kalman Filter", fontsize=14, fontweight="bold")

    # ── Position ─────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(t, true_pos, "k-", linewidth=2, label="True position")
    ax.scatter(t, measurements, c="tomato", s=15, alpha=0.6, label="Measurements", zorder=3)
    ax.plot(t, est_pos, "steelblue", linewidth=2, label="Kalman estimate")
    ax.fill_between(
        t,
        est_pos - 2 * pos_std,
        est_pos + 2 * pos_std,
        alpha=0.2, color="steelblue", label="±2σ confidence",
    )
    ax.set_ylabel("Position (m)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # ── Velocity ─────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(t, true_vel, "k-", linewidth=2, label="True velocity")
    ax.plot(t, est_vel, "orange", linewidth=2, label="Kalman velocity estimate")
    ax.set_ylabel("Velocity (m/s)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # ── Position error & covariance ───────────────────────────────────────
    ax = axes[2]
    error = true_pos - est_pos
    ax.plot(t, error, "purple", linewidth=1.5, label="Position error")
    ax.fill_between(t, -2 * pos_std, 2 * pos_std, alpha=0.2, color="steelblue",
                    label="±2σ bound")
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (m)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("kalman_filter_result.png", dpi=120, bbox_inches="tight")
    plt.show()

    rmse = np.sqrt(np.mean(error ** 2))
    meas_rmse = np.sqrt(np.mean((true_pos - measurements) ** 2))
    print(f"Measurement RMSE: {meas_rmse:.3f} m")
    print(f"Kalman Filter RMSE: {rmse:.3f} m")


if __name__ == "__main__":
    main()
