import numpy as np

def integrate(model, initial_state, timestep, sim_time, params):
    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((len(initial_state), n_timesteps))
    state_traj[:, 0] = initial_state

    for step, t in enumerate(time_traj[:-1]):
        state_main = state_traj[:2, step] # states to be integrated
        state_additional = state_traj[2:, step] #states not integrated over

        state_new = state_main + timestep * model.dynamics(t, state_main, params)

        state_new_full = np.concatenate([state_new, state_additional])

        if hasattr(model, "detect_event"): #check for impact
            state_new_full = model.check(state_new_full, params)

        state_traj[:2, step + 1] = state_new_full

    return time_traj, state_traj

