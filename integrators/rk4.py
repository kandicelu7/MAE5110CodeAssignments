import numpy as np

def integrate(model, initial_state, timestep, sim_time, params):
    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state

    # simulation loop
    for step, t in enumerate(time_traj[:-1]):

        state = state_traj[:, step]# current x

        k1 = model.dynamics(t, state, params)

        k2 = model.dynamics(
            t + timestep / 2, # t+h/2
            state + timestep * k1 / 2, # x+k1*h/2
            params
        )

        k3 = model.dynamics(
            t + timestep / 2, # t+h/2
            state + timestep * k2 / 2, # x+k2*h/2
            params
        )

        k4 = model.dynamics(
            t + timestep, # t+h
            state + timestep * k3, # x+k3*h
            params
        )
        blah = state + (timestep / 6) * (k1 + 2*k2 + 2*k3 + k4) # weighted sum

        if hasattr(model, "check"):
            blah = model.check(blah, params)

        state_traj[:, step + 1] = blah

    return time_traj, state_traj

