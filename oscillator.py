#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np


def rhs(state: np.ndarray, gamma: float, omega_0: float) -> np.ndarray:
    x, v = state
    # x in m, v in m/s, gamma in s^-1, omega_0 in rad/s.
    # Writing x'' + gamma x' + omega_0^2 x = 0 as a first-order pair.
    return np.array([v, -gamma * v - omega_0**2 * x], dtype=float)


def rk4_step(state: np.ndarray, dt: float, gamma: float, omega_0: float) -> np.ndarray:
    k1 = rhs(state, gamma, omega_0)
    k2 = rhs(state + 0.5 * dt * k1, gamma, omega_0)
    k3 = rhs(state + 0.5 * dt * k2, gamma, omega_0)
    k4 = rhs(state + dt * k3, gamma, omega_0)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_trajectory(t: np.ndarray, x0: float, v0: float, gamma: float, omega_0: float) -> tuple[np.ndarray, np.ndarray]:
    states = np.empty((t.size, 2), dtype=float)
    states[0] = np.array([x0, v0], dtype=float)
    dt = t[1] - t[0]

    for i in range(t.size - 1):
        states[i + 1] = rk4_step(states[i], dt, gamma, omega_0)

    return states[:, 0], states[:, 1]


def underdamped_solution(t: np.ndarray, x0: float, v0: float, gamma: float, omega_0: float) -> tuple[np.ndarray, float, float, float]:
    discriminant = omega_0**2 - (gamma**2) / 4.0
    if discriminant <= 0.0:
        raise ValueError(
            "This closed-form comparison only works in the underdamped regime: gamma < 2 * omega_0."
        )

    omega = np.sqrt(discriminant)

    # Determining A and phi from initial displacement and velocity.
    # Start from x(t) = exp(-gamma t / 2) [C cos(omega t) + D sin(omega t)]
    # and solve C = x0, D = -(v0 + gamma x0 / 2) / omega at t = 0.
    c_cos = x0
    c_sin = -(v0 + 0.5 * gamma * x0) / omega
    amplitude = np.hypot(c_cos, c_sin)
    phase = np.arctan2(-c_sin, c_cos)

    x_analytic = amplitude * np.exp(-0.5 * gamma * t) * np.cos(omega * t + phase)
    return x_analytic, omega, amplitude, phase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Damped oscillator: RK4 against the underdamped analytic curve."
    )
    parser.add_argument("--gamma", type=float, default=0.2, help="Damping rate gamma [s^-1].")
    parser.add_argument("--omega0", type=float, default=1.5, help="Natural angular frequency omega_0 [rad/s].")
    parser.add_argument("--x0", type=float, default=1.0, help="Initial displacement [m].")
    parser.add_argument("--v0", type=float, default=0.0, help="Initial velocity [m/s].")
    parser.add_argument("--tmax", type=float, default=30.0, help="Stop time [s].")
    parser.add_argument("--steps", type=int, default=4000, help="Number of RK4 steps.")
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("oscillator.png"),
        help="Where to write the figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the plot window too.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.steps < 2:
        raise ValueError("steps must be at least 2.")
    if args.tmax <= 0.0:
        raise ValueError("tmax must be positive.")

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    # Uniform time grid keeps the RK4 bookkeeping clean for quick parameter sweeps.
    t = np.linspace(0.0, args.tmax, args.steps + 1)
    x_num, _ = integrate_trajectory(t, args.x0, args.v0, args.gamma, args.omega0)
    x_analytic, omega, amplitude, phase = underdamped_solution(
        t, args.x0, args.v0, args.gamma, args.omega0
    )
    residuals = x_num - x_analytic

    rms_error = np.sqrt(np.mean(residuals**2))
    max_error = np.max(np.abs(residuals))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    axes[0].plot(t, x_num, label="RK4 numerical", linewidth=2.0)
    axes[0].plot(t, x_analytic, "--", label="Analytical underdamped", linewidth=2.0)
    axes[0].set_ylabel("Position x(t) [m]")
    axes[0].set_title(
        "Damped Harmonic Oscillator\n"
        f"$\\gamma={args.gamma:.3f}$, $\\omega_0={args.omega0:.3f}$, "
        f"$\\omega={omega:.3f}$, $A={amplitude:.3f}$, $\\phi={phase:.3f}$"
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, residuals, color="crimson", linewidth=1.5)
    axes[1].axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("Time t [s]")
    axes[1].set_ylabel("Residual [m]")
    axes[1].set_title(f"Residuals: numerical - analytical | RMS = {rms_error:.3e}, max = {max_error:.3e}")
    axes[1].grid(True, alpha=0.3)

    args.save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=200)

    print("Damped harmonic oscillator simulation complete.")
    print(f"Underdamped angular frequency omega = {omega:.8f}")
    print(f"Amplitude A = {amplitude:.8f}")
    print(f"Phase phi = {phase:.8f}")
    print(f"RMS residual = {rms_error:.8e}")
    print(f"Max residual = {max_error:.8e}")
    print(f"Figure saved to: {args.save.resolve()}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
