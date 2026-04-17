#!/usr/bin/env python3
"""Simulate a damped harmonic oscillator with RK4 and compare to the underdamped analytical solution.

Equation of motion:
    d^2x/dt^2 + gamma * dx/dt + omega_0^2 * x = 0

Converted to a first-order system with y = [x, v]:
    dx/dt = v
    dv/dt = -gamma * v - omega_0^2 * x
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def oscillator_rhs(state: np.ndarray, gamma: float, omega_0: float) -> np.ndarray:
    """Return the time derivative [dx/dt, dv/dt] for the damped oscillator."""
    x, v = state
    return np.array([v, -gamma * v - omega_0**2 * x], dtype=float)


def rk4_step(state: np.ndarray, dt: float, gamma: float, omega_0: float) -> np.ndarray:
    """Advance the state by one RK4 step."""
    k1 = oscillator_rhs(state, gamma, omega_0)
    k2 = oscillator_rhs(state + 0.5 * dt * k1, gamma, omega_0)
    k3 = oscillator_rhs(state + 0.5 * dt * k2, gamma, omega_0)
    k4 = oscillator_rhs(state + dt * k3, gamma, omega_0)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def solve_rk4(
    t: np.ndarray,
    x0: float,
    v0: float,
    gamma: float,
    omega_0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the oscillator numerically with RK4 on a uniform time grid."""
    states = np.zeros((t.size, 2), dtype=float)
    states[0] = np.array([x0, v0], dtype=float)
    dt = t[1] - t[0]

    for i in range(t.size - 1):
        states[i + 1] = rk4_step(states[i], dt, gamma, omega_0)

    return states[:, 0], states[:, 1]


def analytical_underdamped(
    t: np.ndarray,
    x0: float,
    v0: float,
    gamma: float,
    omega_0: float,
) -> tuple[np.ndarray, float, float, float]:
    """Return x(t) = A exp(-gamma t / 2) cos(omega t + phi) for the underdamped regime."""
    discriminant = omega_0**2 - (gamma**2) / 4.0
    if discriminant <= 0.0:
        raise ValueError(
            "The analytical form used here requires an underdamped oscillator: gamma < 2 * omega_0."
        )

    omega = np.sqrt(discriminant)

    # Match the analytical solution to the initial conditions.
    c_cos = x0
    c_sin = -(v0 + 0.5 * gamma * x0) / omega
    amplitude = np.hypot(c_cos, c_sin)
    phase = np.arctan2(-c_sin, c_cos)

    x_analytic = amplitude * np.exp(-0.5 * gamma * t) * np.cos(omega * t + phase)
    return x_analytic, omega, amplitude, phase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Damped harmonic oscillator solved with RK4 and compared to the analytical solution."
    )
    parser.add_argument("--gamma", type=float, default=0.2, help="Damping coefficient.")
    parser.add_argument("--omega0", type=float, default=1.5, help="Natural angular frequency.")
    parser.add_argument("--x0", type=float, default=1.0, help="Initial position.")
    parser.add_argument("--v0", type=float, default=0.0, help="Initial velocity.")
    parser.add_argument("--tmax", type=float, default=30.0, help="Final simulation time.")
    parser.add_argument("--steps", type=int, default=4000, help="Number of time steps.")
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("oscillator.png"),
        help="Path for the saved figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively instead of only saving it.",
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

    t = np.linspace(0.0, args.tmax, args.steps + 1)
    x_num, _ = solve_rk4(t, args.x0, args.v0, args.gamma, args.omega0)
    x_analytic, omega, amplitude, phase = analytical_underdamped(
        t, args.x0, args.v0, args.gamma, args.omega0
    )
    residuals = x_num - x_analytic

    rms_error = np.sqrt(np.mean(residuals**2))
    max_error = np.max(np.abs(residuals))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    axes[0].plot(t, x_num, label="RK4 numerical", linewidth=2.0)
    axes[0].plot(t, x_analytic, "--", label="Analytical underdamped", linewidth=2.0)
    axes[0].set_ylabel("Position x(t)")
    axes[0].set_title(
        "Damped Harmonic Oscillator\n"
        f"$\\gamma={args.gamma:.3f}$, $\\omega_0={args.omega0:.3f}$, "
        f"$\\omega={omega:.3f}$, $A={amplitude:.3f}$, $\\phi={phase:.3f}$"
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, residuals, color="crimson", linewidth=1.5)
    axes[1].axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("Time t")
    axes[1].set_ylabel("Residual")
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
