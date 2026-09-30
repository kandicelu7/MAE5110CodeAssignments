#integrates one step with rk4
import numpy as np

def integrate_step(dynamics, t, state, timestep, params):

    k1 = dynamics(t,              state,                  params)
    k2 = dynamics(t + timestep/2, state+ timestep * k1/2, params)
    k3 = dynamics(t + timestep/2, state+ timestep * k2/2, params)
    k4 = dynamics(t + timestep,   state+ timestep * k3,   params)

    state_new = state + (timestep / 6) * (k1 + 2*k2 + 2*k3 + k4)

    return state_new