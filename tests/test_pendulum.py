"""Energy sanity checks for the pendulum, where angle 0 is upright.

With E = KE + PE, the dynamics give dE/dt = omega * (torque - damping * omega).
"""

import numpy as np

from integrators import rk4
from models import pendulum

TIMESTEP = 0.01
N_STEPS = 100


def simulate(params, initial_state):
    """Return the state trajectory, shape (2, N_STEPS + 1)."""
    state_traj = np.zeros((2, N_STEPS + 1))
    state_traj[:, 0] = initial_state
    for step in range(N_STEPS):
        state_traj[:, step + 1] = rk4(
            pendulum.dynamics, step * TIMESTEP, state_traj[:, step], TIMESTEP, params
        )
    return state_traj


def total_energy(state_traj, params):
    kinetic, potential = pendulum.calculate_energy(state_traj, params)
    return kinetic + potential


def test_energy_conserved_without_damping_or_torque():
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.0
    params["torque"] = 0.0
    # Start away from the equilibria at 0 (upright) and pi (hanging).
    state_traj = simulate(params, np.array([1.0, 0.0]))
    energy = total_energy(state_traj, params)

    assert np.all(np.isclose(np.diff(energy), 0.0, atol=1e-6))
    assert np.isclose(energy[-1] - energy[0], 0.0, atol=1e-6)


def test_damping_dissipates_energy():
    # dE/dt = -b * omega^2, so energy never increases and the total loss
    # equals the work done by damping.
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.5
    params["torque"] = 0.0
    state_traj = simulate(params, np.array([1.0, 0.0]))
    energy = total_energy(state_traj, params)
    damping_work = np.trapezoid(
        params["damping_coeff"] * state_traj[1] ** 2, dx=TIMESTEP
    )

    assert np.all(np.diff(energy) <= 1e-9)
    assert np.isclose(energy[0] - energy[-1], damping_work, rtol=1e-3)


def test_torque_does_work():
    # Without damping, dE/dt = torque * omega, so a constant torque adds
    # torque * (change in angle) to the energy.
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.0
    params["torque"] = 2.0
    state_traj = simulate(params, np.array([1.0, 0.0]))
    energy = total_energy(state_traj, params)
    torque_work = params["torque"] * (state_traj[0] - state_traj[0, 0])

    assert np.all(np.isclose(energy - energy[0], torque_work, atol=1e-6))
