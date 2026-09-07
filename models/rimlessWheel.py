import numpy as np

def generate_params():
    params = {
        "mass": 1, # mass of hub (kg)
        "length": 1,  # length of spoke (m)
        "gravity": 9.81,  # gravity (m/s^2)
        "spoke_number": 6, #number of wheel spokes
        "inclination_angle": np.pi/10,  #slope of ground (rad)

    }
    return params

def dynamics(t, state, params):
    gravity = params["gravity"]
    length = params["length"]

    angle = state[0]
    angular_velocity = state[1]

    angular_acceleration = (gravity * np.sin(angle)) / length

    state_derivative = np.array([angular_velocity, angular_acceleration])
    return state_derivative

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

def detect_event(state, params, poincare_tracker): #impact with ground
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
        state[2] += 1 #increment step count

        poincare_tracker = np.append(poincare_tracker, [state[1]])

    if angle < impact_angle_neg:
        state[0] = inclination_angle + alpha
        state[1] = angular_velocity * np.cos(2*alpha)
        state[2] -= 1

        poincare_tracker = np.append(poincare_tracker, [state[1]])

    return state, poincare_tracker

def break_condition(state_history):
    if len(state_history) >= 10000 and np.all(abs(state_history[-1000:]) < 0.05):
        break_integration = True
    else:
        break_integration = False

    return break_integration


