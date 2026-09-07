import numpy as np
from models import rimlessWheel as model
from integrators import rk4 as integrator

def find_fixed_point(params, timestep, sim_time, initial_velocity):
    initial_state = np.array([0, initial_velocity, 0])
    tracker = []
    _, state_traj, poincare_tracker = integrator.integrate(model, initial_state, timestep, sim_time, params, tracker)
    fixed_point_ang_velocity = np.mean(poincare_tracker[-4:])

    if len(poincare_tracker) < 5:
        return np.nan
    if state_traj[1, -1] < 0.1:
        return np.nan

    return fixed_point_ang_velocity

def estimate_floquet_multiplier(params, timestep, fixed_point_velocity, perturbance):
    spoke_number = params["spoke_number"]
    inclination_angle = params["inclination_angle"]
    alpha = np.pi / spoke_number

    post_impact_angle = inclination_angle - alpha

    initial_state_minus = [post_impact_angle, fixed_point_velocity - perturbance, 0]
    initial_state_plus = [post_impact_angle, fixed_point_velocity + perturbance, 0]
    _, _, poincare_tracker_minus = integrator.integrate(model, initial_state_minus, timestep, 3, params, [])
    _, _, poincare_tracker_plus = integrator.integrate(model, initial_state_plus, timestep, 3, params, [])

    if len(poincare_tracker_minus) == 0 or len(poincare_tracker_plus) == 0:
            return np.nan  # perturbation pushed trajectory out of the rolling basin

    floquet_multiplier = (poincare_tracker_plus[0]-poincare_tracker_minus[0]) / (2*perturbance)

    return floquet_multiplier

def estimate_RoA_fraction(params, timestep, sim_time, grid_points=20):

    spoke_number = params["spoke_number"]
    inclination_angle = params["inclination_angle"]
    alpha = np.pi / spoke_number

    angles = np.linspace(inclination_angle - alpha, inclination_angle + alpha, grid_points)
    velocities = np.linspace(-np.pi/10, np.pi/10, grid_points)

    rolls = 0
    total = 0
    for v0 in velocities:
        for a0 in angles:
            _, state_traj, _ = integrator.integrate(model, np.array([a0, v0, 0]), timestep, sim_time, params, [])
            if abs(state_traj[1, -1]) > 0.1:
                rolls += 1
            total += 1
    return rolls / total
