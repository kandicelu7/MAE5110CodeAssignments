#integrates one step with rk4
import numpy as np

def integrate_step(model, state, t, timestep, params):

    k1 = model.dynamics(t,              state,                  params)
    k2 = model.dynamics(t + timestep/2, state+ timestep * k1/2, params)
    k3 = model.dynamics(t + timestep/2, state+ timestep * k2/2, params)
    k4 = model.dynamics(t + timestep,   state+ timestep * k3,   params)

    state_new = state + (timestep / 6) * (k1 + 2*k2 + 2*k3 + k4)

    return state_new