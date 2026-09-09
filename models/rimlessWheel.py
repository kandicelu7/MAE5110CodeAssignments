import numpy as np
from integrators import rk4 as integrator

def dynamics(t, state, params):
    gravity = params["gravity"]
    length = params["length"]

    angle = state[0]
    angular_velocity = state[1]

    angular_acceleration = (gravity * np.sin(angle)) / length

    return np.array([angular_velocity, angular_acceleration])


def calculate_energy(state, params):
    mass = params["mass"]
    length = params["length"]
    gravity = params["gravity"]
    spoke_number = params["spoke_number"]
    inclination_angle = params["inclination_angle"]

    angle = state[0]
    angular_velocity = state[1]
    step_count = state[2]

    alpha = np.pi/spoke_number
    height_drop_per_step = 2*length * np.sin(alpha) * np.sin(inclination_angle)

    kinetic_energy = 0.5 * mass * (length * angular_velocity)**2
    hub_height = (length*np.cos(angle) - height_drop_per_step*step_count)
    potential_energy = mass * gravity * hub_height

    return kinetic_energy, potential_energy, hub_height


def detect_event(state, params, poincare_tracker):
    spoke_number = params["spoke_number"]
    inclination_angle = params["inclination_angle"]

    angle = state[0]
    angular_velocity = state[1]

    alpha = np.pi/spoke_number

    impact_angle_pos = inclination_angle + alpha
    impact_angle_neg = inclination_angle - alpha

    if angle > impact_angle_pos:
        state[0] = inclination_angle - alpha
        state[1] = angular_velocity * np.cos(2*alpha)
        state[2] += 1

        poincare_tracker = np.append(poincare_tracker, [state[1]])

    if angle < impact_angle_neg:
        state[0] = inclination_angle + alpha
        state[1] = angular_velocity * np.cos(2*alpha)
        state[2] -= 1

        poincare_tracker = np.append(poincare_tracker, [state[1]])

    return state, poincare_tracker


def break_condition(state_history, poincare_tracker, t):
    if t >= 1.0 and np.all(abs(state_history[-100:]) < 0.05):
        return True
    if len(poincare_tracker) >= 3 and np.max(np.abs(poincare_tracker[-1] - poincare_tracker[-2]) < 1e-3):
        return True
    else:
        return False


# ---------------- simulation / analysis ----------------

def full_integration(model, initial_state, coarse_timestep, fine_timestep, impact_threshold, sim_time, params, keep_history):
    spoke_number = params["spoke_number"]
    inclination_angle = params["inclination_angle"]
    alpha = np.pi / spoke_number

    angle_pos = inclination_angle + alpha
    angle_neg = inclination_angle - alpha

    state = initial_state.copy().astype(float)
    tracker = []
    t = 0.0

    velocity_history = [state[1]]

    if keep_history:
        time_list = [t]
        state_list = [state.copy()]

    t_limit = sim_time if sim_time is not None else 5
    while t < t_limit: #
        dist_to_impact = min(abs(angle_pos - state[0]), abs(state[0] - angle_neg))
        timestep = fine_timestep if dist_to_impact < impact_threshold and state[1] > 0.05 else coarse_timestep

        state_main = integrator.integrate_step(model, state[:2], t, timestep, params)
        state = np.concatenate([state_main, state[2:]])
        state, tracker = model.detect_event(state, params, tracker)

        t += timestep
        velocity_history.append(state[1])

        if keep_history:
            time_list.append(t)
            state_list.append(state.copy())

        if len(velocity_history) % 75 == 0 and sim_time is None:
            if model.break_condition(np.array(velocity_history), tracker, t):
                break

    if keep_history:
        time_traj = np.array(time_list)
        state_traj = np.array(state_list).T
        return time_traj, state_traj, tracker
    else:
        return state, tracker

def find_fixed_point(params, coarse_timestep, fine_timestep, impact_threshold, sim_time, initial_velocity):
    import sys
    model = sys.modules[__name__]

    initial_state = np.array([0, initial_velocity, 0], dtype=float)
    _, state_traj, poincare_tracker = full_integration(
        model, initial_state, coarse_timestep, fine_timestep, impact_threshold, sim_time, params, keep_history=True
    )

    if len(poincare_tracker) < 5:
        return np.nan
    if state_traj[1, -1] < 0.1:
        return np.nan

    return np.mean(poincare_tracker[-4:])


def estimate_floquet_multiplier(params, coarse_timestep, fine_timestep, impact_threshold, fixed_point_velocity, perturbance):
    import sys
    model = sys.modules[__name__]

    spoke_number = params["spoke_number"]
    inclination_angle = params["inclination_angle"]
    alpha = np.pi / spoke_number

    post_impact_angle = inclination_angle - alpha
    sim_time = 3

    initial_state_minus = np.array([post_impact_angle, fixed_point_velocity - perturbance, 0], dtype=float)
    initial_state_plus = np.array([post_impact_angle, fixed_point_velocity + perturbance, 0], dtype=float)

    _, _, poincare_tracker_minus = full_integration(
        model, initial_state_minus, coarse_timestep, fine_timestep, impact_threshold, sim_time, params, keep_history=True
    )
    _, _, poincare_tracker_plus = full_integration(
        model, initial_state_plus, coarse_timestep, fine_timestep, impact_threshold, sim_time, params, keep_history=True
    )

    if len(poincare_tracker_minus) == 0 or len(poincare_tracker_plus) == 0:
        return np.nan  # perturbation pushed trajectory out of the rolling basin

    return (poincare_tracker_plus[0] - poincare_tracker_minus[0]) / (2 * perturbance)


def estimate_RoA_fraction(params, coarse_timestep, fine_timestep, impact_threshold, sim_time, grid_points):
    import sys
    model = sys.modules[__name__]

    spoke_number = params["spoke_number"]
    inclination_angle = params["inclination_angle"]
    alpha = np.pi / spoke_number

    angles = np.linspace(inclination_angle - alpha, inclination_angle + alpha, grid_points)
    velocities = np.linspace(-4, 4, grid_points)

    rolls = 0
    total = 0
    for v0 in velocities:
        for a0 in angles:
            initial_state = np.array([a0, v0, 0], dtype=float)
            final_state, _ = full_integration(
                model, initial_state, coarse_timestep, fine_timestep, impact_threshold, sim_time, params, keep_history=False
            )
            if abs(final_state[1]) > 0.1:
                rolls += 1
            total += 1
    return rolls / total