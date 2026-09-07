import numpy as np

def dynamics(t, state, params):
    gravity = params["gravity"]
    drag_coeff = params["drag_coeff"]

    velocity = state[1]

    acceleration = (-gravity - drag_coeff * abs(velocity)*velocity)

    state_derivative = np.array([velocity, acceleration])
    return state_derivative

def generate_params():
    params = {
        "gravity": 9.81,  # gravity m/s^2)
        "elastic_coeff": 0.95,  # rod length (m)
        "mass": 1,  # point mass at end of rod (kg)
        "drag_coeff": 0.1,  # drag coefficient (kg*m^2/s)
        }
    return params

def calculate_energy(state, params):
    gravity = params["gravity"]
    mass = params["mass"]

    height = state[0]
    velocity = state[1]

    kinetic_energy = 0.5 * mass * (velocity)**2
    potential_energy = mass * gravity * height
    return kinetic_energy, potential_energy

def detect_event(state, params):
    elastic_coeff = params["elastic_coeff"]

    height = state[0]
    velocity = state[1]

    if height < 0:        #reflects ball if at ground
        state[0] = 0.0
        state[1] = elastic_coeff * abs(velocity)
    return state
