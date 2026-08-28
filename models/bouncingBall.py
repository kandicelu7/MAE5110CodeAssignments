import numpy as np


def dynamics(t, state, params):
    gravity = params["gravity"]
    elastic_coeff = params["elastic_coeff"]
    mass = params["mass"]
    drag_coeff = params["drag_coeff"]

    height = state[0]
    velocity = state[1]

    acceleration = (
        - gravity
        - drag_coeff * abs(velocity) * velocity  # <-- DAMPING TERM
    )

    state_derivative = np.array([velocity, acceleration])
    return state_derivative

def generate_params():
    params = {
        "gravity": 9.81,  # gravity m/s^2)
        "elastic_coeff": 0.95,  # rod length (m)
        "mass": 1,  # point mass at end of rod (kg)
        "drag_coeff": 0.1,  # damping coefficient (kg*m^2/s)
    }
    return params


def calculate_energy(state, params):
    """Compute energies for a state ``(2,)`` or trajectory ``(2, N)``."""
    gravity = params["gravity"]
    mass = params["mass"]

    height = state[0]  # indexes entire row "vectorized" if state is (2, N)
    velocity = state[1]

    kinetic_energy = 0.5 * mass * (velocity) ** 2
    potential_energy = mass * gravity * height
    return kinetic_energy, potential_energy

def check(state, params):
    #reflects ball if at ground
    elastic_coeff = params["elastic_coeff"]
    if state[0] < 0:
        state[0] = 0.0
        state[1] = elastic_coeff * abs(state[1])
    return state
