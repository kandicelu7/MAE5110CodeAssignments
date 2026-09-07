import numpy as np

def integrate(model, initial_state, timestep, sim_time, params, tracker):
    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((len(initial_state), n_timesteps))
    state_traj[:, 0] = initial_state

    for step, t in enumerate(time_traj[:-1]):
        state_main = state_traj[:2, step] # states to be integrated
        state_additional = state_traj[2:, step] #states not integrated over

        k1 = model.dynamics(t,              state_main,                  params)
        k2 = model.dynamics(t + timestep/2, state_main+ timestep * k1/2, params)
        k3 = model.dynamics(t + timestep/2, state_main+ timestep * k2/2, params)
        k4 = model.dynamics(t + timestep,   state_main+ timestep * k3,   params)

        state_new = state_main + (timestep / 6) * (k1 + 2*k2 + 2*k3 + k4)

        state_new_full = np.concatenate([state_new, state_additional])

        if hasattr(model, "detect_event"): #impact check
            state_new_full, tracker = model.detect_event(state_new_full, params, tracker)

        state_traj[:, step + 1] = state_new_full

        if hasattr(model, "break_condition"):
            break_integration = model.break_condition(state_traj[1, :step + 2])
            if break_integration == True:
                time_traj = time_traj[:step + 2]
                state_traj = state_traj[:, :step + 2]
                break

    return time_traj, state_traj, tracker

