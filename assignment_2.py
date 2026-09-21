from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from models import inverted_pendulum_walker as model
from integrators import rk4 as integrator

output = Path("output/assignment_2")
output.mkdir(parents=True, exist_ok=True)

fine_timestep = 1e-4
coarse_timestep = 1e-2
sim_time = 15
desired_number_of_steps = 6

simulate_RoA_sweep = 1 #roa map
return_map = 1
state_action_table = 1
validate_state_action_table = 1 #checks how accurate predictions are

run_control_simulation = 1 #requires state_action_table to be run once before
create_animation = 1

params = model.generate_params()
#params["angle_of_attack"] = np.pi/8
initial_state = np.array([0, 3])

def full_integration(initial_state, coarse_timestep, fine_timestep, sim_time, desired_number_of_steps, params, torque_activation_limit, alpha_control):

    time_traj = [0.0]
    state_traj = [initial_state]
    torque_traj = [params["ankle_torque"]]
    velocity_tracker = []

    inclination = params["incline"]

    t = 0.0
    completed_steps = 0
    final_state = "time_limit_reached"

    while t < sim_time:
        state = state_traj[-1]

        alpha = params["angle_of_attack"]
        impact_angle = inclination + alpha

        #refines timestep when needed
        timestep = fine_timestep if state[0]-impact_angle > 0.01 or abs(state[0]) < 0.01 else coarse_timestep

        next_state = integrator.integrate_step(model, state, t, timestep, params)
        t += timestep

        step_impact = model.impact_guard(next_state, params) #check if impact or failure happened
        if step_impact:
            next_state[0] = inclination - alpha
            next_state[1] = next_state[1] * np.cos(2*alpha)
            completed_steps += 1

        if model.zero_crossing_guard(state, next_state, step_impact): #check for theta = 0 crossing
            velocity_tracker.append(next_state[1])

            if alpha_control is not None:
                params["angle_of_attack"] = model.alpha_feedback(next_state, alpha_control)


        params["ankle_torque"] = model.torque_feedback(next_state, params, torque_activation_limit)

        state_traj.append(next_state)
        time_traj.append(t)
        torque_traj.append(params["ankle_torque"])

        end_integration, final_state = model.break_condition(state, state_traj, params, completed_steps, desired_number_of_steps)
        if end_integration:
            break

    time_traj = np.array(time_traj)
    state_traj = np.array(state_traj).T
    torque_traj = np.array(torque_traj)

    return time_traj, state_traj, torque_traj, final_state, completed_steps, velocity_tracker

def find_velocity_index(velocity, velocity_range):
    if not np.isfinite(velocity):
        return -1

    dv = velocity_range[1] - velocity_range[0]
    if velocity < velocity_range[0] - dv / 2 or velocity > velocity_range[-1] + dv / 2:
        return -1                                   # outside the table

    idx = int(np.searchsorted(velocity_range, velocity, side="left"))

    if idx <= 0:
        return 0
    if idx >= len(velocity_range):
        return len(velocity_range) - 1

    if abs(velocity_range[idx] - velocity) < abs(velocity - velocity_range[idx - 1]):
        return idx
    else:
        return idx - 1

if return_map:
    inclination = params["incline"]
    velocity_magnitudes = np.linspace(-0.5, 1.75, 100)

    poincare_x = []  #v_k
    poincare_y = []  #v_k+1

    initial_conditions = []
    print("")
    for v0 in velocity_magnitudes:
        initial_state_sweep = np.array([-0.001, v0])
        sweep_params = model.generate_params()

        _, _, _, final_state, _, velocity_tracker = full_integration(initial_state_sweep, coarse_timestep, fine_timestep, sim_time, None, sweep_params, True, None)

        if final_state == "stabilized_upright" and len(velocity_tracker) == 1:
            poincare_x.append(velocity_tracker[0])
            poincare_y.append(0)
            continue

        poincare_x.extend(velocity_tracker[:-1])
        poincare_y.extend(velocity_tracker[1:])

    plt.figure(4)
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(True)
    plt.scatter(poincare_x, poincare_y, s=15)

    lims = [min(poincare_x, default=-1), max(poincare_x, default=1)]
    plt.plot(lims, lims, linewidth=0.5, label="identity line")
    plt.xlabel(r"$\dot\theta_k$ (incoming)")
    plt.ylabel(r"$\dot\theta_{k+1}$ (next crossing)")
    plt.title(f"Poincare section (alpha={np.degrees(params['angle_of_attack']):.1f} deg, "f"{np.degrees(inclination):.1f} deg incline)\n")
    plt.tight_layout()
    plt.legend()
    plt.savefig("output/assignment_2/return_map_plot.png")

if simulate_RoA_sweep:
    RoA_grid_points = 100
    angle_range = np.linspace(-0.3, 0.3, RoA_grid_points)
    velocity_range = np.linspace(-.75, 0.75, RoA_grid_points)

    result_categories = [
        "stabilized_upright",
        "fell_over",
        "step_limit_reached"]
    category_codes = {name: idx for idx, name in enumerate(result_categories)}
    sweep_results = np.zeros((RoA_grid_points, RoA_grid_points))

    print("sweeping RoA")
    for i, angle0 in enumerate(angle_range):
        for j, velocity0 in enumerate(velocity_range):
            sweep_params = model.generate_params()
            initial_state_sweep = [angle0, velocity0]

            _, state_traj_sweep, _, final_state, _, _ = full_integration(initial_state_sweep, coarse_timestep, fine_timestep, sim_time, 1, sweep_params, True, None)

            sweep_results[j, i] = category_codes[final_state]

        print(f"  row {i + 1}/{RoA_grid_points} done")

    cmap = plt.get_cmap("Blues", len(result_categories))
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    mesh = ax.pcolormesh(angle_range, velocity_range, sweep_results, cmap=cmap, vmin=-0.5, vmax=len(result_categories) - 0.5, shading="nearest")
    cbar = fig.colorbar(mesh, ax=ax, ticks=range(len(result_categories)))
    cbar.ax.set_yticklabels(["Stabilizes upright", "Falls backwards", "Takes another step"])
    ax.set( xlabel=r"$\theta_0$ (rad)", ylabel=r"$\dot\theta_0$ (rad/s)")
    ax.set_title("Ankle controller region of attraction")

    fig.savefig(output / "roa_sweep.png", dpi=150)

if state_action_table:
    params_copy = model.generate_params()

    mass = params_copy["mass"]
    length = params_copy["length"]
    gravity = params_copy["gravity"]

    grid_points = 75

    velocity_range = np.linspace(0,np.sqrt(2*gravity/length), grid_points)
    alpha_range = np.linspace(np.pi/8, np.pi/7, grid_points)

    policy_path = output / "state_action_policy.npz"

    steps_to_stabilize = np.full((grid_points), np.nan)
    alpha_to_apply = np.full((grid_points, grid_points), -1)

    def state_action_sweep(velocity_range, alpha_range, coarse_timestep, fine_timestep, sim_time):
        steps_grid = np.zeros((grid_points, grid_points), dtype=bool)
        velocity_next = np.full((grid_points, grid_points), np.nan)

        for v,  v0 in enumerate(velocity_range):
            print("Sweeping state action grid. Current velocity:", v0)
            for a,  alpha in enumerate(alpha_range):
                initial_state = [0, v0]
                params_copy = model.generate_params()
                params_copy["angle_of_attack"] = alpha

                _,_,_, final_state, completed_steps, velocity_tracker = full_integration(initial_state, coarse_timestep, fine_timestep, sim_time, 2, params_copy, True, None)

                if final_state == "stabilized_upright":
                    if completed_steps == 0:
                        steps_to_stabilize[v] = 0
                    elif completed_steps == 1:
                        steps_grid[v,a] = 1

                if velocity_tracker:
                    velocity_next[v,a] = velocity_tracker[0]
        return steps_grid, velocity_next

    def safest_alpha(indices): #alpha in middle of potential angles
        indices = sorted(indices)
        best_start = best_len = 0
        run_start = 0
        for i in range(1, len(indices) + 1):
            if i == len(indices) or indices[i] != indices[i - 1] + 1:
                run_len = i - run_start
                if run_len > best_len:
                    best_len = run_len
                    best_start = run_start
                run_start = i
        run = indices[best_start:best_start + best_len]
        return run[len(run) // 2]

    steps_grid, velocity_next = state_action_sweep(velocity_range, alpha_range, coarse_timestep, fine_timestep, sim_time)

    #mark velocities than can stabilize within a step
    for v in range(grid_points):
        if np.isnan(steps_to_stabilize[v]) and np.any(steps_grid[v] == 1):
            steps_to_stabilize[v] = 1
            working_a = np.where(steps_grid[v])[0]
            alpha_to_apply[v] = safest_alpha(working_a)

    #map next velocity points to grid
    next_index = np.full((grid_points, grid_points), -1) #initialize with -1 being and invalid index
    for v in range(grid_points):
        for a in range(grid_points):
            next_index[v, a] = find_velocity_index(velocity_next[v, a], velocity_range) #maps each output velocity to an input velocity

    steps = 2
    while True:
        prev_steps_to_stabilize = steps_to_stabilize
        states_added = False
        updates = {}

        for v in range(grid_points):
            if not np.isnan(steps_to_stabilize[v]):
                continue

            working_alphas = []
            for a in range(grid_points):
                if next_index[v,a] >= 0:
                    new_velocity_index = next_index[v,a]
                    if not np.isnan(prev_steps_to_stabilize[new_velocity_index]):
                        working_alphas.append(a)

            if working_alphas:
                downstream_steps = [prev_steps_to_stabilize[next_index[v, a2]] for a2 in working_alphas]
                min_downstream = min(downstream_steps)
                best_tier = [a2 for a2, s in zip(working_alphas, downstream_steps) if s == min_downstream]
                best_a = safest_alpha(best_tier)
                updates[v] = (min_downstream + 1, best_a)
                states_added = True

        for v, (s, a) in updates.items():
            steps_to_stabilize[v] = s
            alpha_to_apply[v] = a

        if not states_added: #no new changes
            break
        steps += 1

    np.savez(policy_path, velocity_range=velocity_range, alpha_range=alpha_range,
                steps_to_stabilize=steps_to_stabilize,
                alpha_to_apply=alpha_to_apply, steps_grid=steps_grid, next_index=next_index)

    reachable = ~np.isnan(steps_to_stabilize)

    unique_steps, counts = np.unique(steps_to_stabilize[reachable], return_counts=True)
    total_states = grid_points
    unreachable_count = total_states - int(np.sum(counts))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, layout="constrained")

    sc = ax1.scatter(velocity_range[reachable], steps_to_stabilize[reachable],
                    c=steps_to_stabilize[reachable], cmap="viridis", s=30)

    if np.any(~reachable):
        ax1.scatter(velocity_range[~reachable], np.zeros(np.sum(~reachable)) - 1,
                    marker="x", color="red", label="never reaches standing")

    fig.colorbar(sc, ax=ax1, label="steps to stabilize")
    ax1.set_ylabel("Steps count")
    ax1.set_title("Steps needed to reach standstill vs. initial velocity")
    ax1.xaxis.set_major_locator(MultipleLocator(0.5))
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax1.minorticks_on()
    ax1.grid(True, which="major")
    ax1.grid(True, which="minor", alpha=0.5)

    if np.any(~reachable):
        ax1.legend()

    chosen_alpha_idx = alpha_to_apply[:, 0].astype(int)
    chosen_alpha = np.full(grid_points, np.nan)
    chosen_alpha[reachable] = alpha_range[chosen_alpha_idx[reachable]]

    sc2 = ax2.scatter(velocity_range[reachable], chosen_alpha[reachable],
                    c=steps_to_stabilize[reachable], cmap="viridis", s=30)

    ax2.set_xlabel(r"$\dot\theta_0$ (initial velocity)")
    ax2.set_ylabel(r"New angle of attack $\alpha$ (rad)")
    ax2.xaxis.set_major_locator(MultipleLocator(0.5))
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax2.minorticks_on()
    ax2.grid(True, which="major")
    ax2.grid(True, which="minor", alpha=0.5)

    plt.savefig(output / "state_action_lookup.png")

    per_action_steps = np.full((grid_points, grid_points), np.nan)

    for v in range(grid_points):
        for a in range(grid_points):
            if steps_grid[v, a]:
                per_action_steps[v, a] = 1
            else:
                k = next_index[v, a]
                if k >= 0 and not np.isnan(steps_to_stabilize[k]):
                    per_action_steps[v, a] = steps_to_stabilize[k] + 1


    reachable = ~np.isnan(steps_to_stabilize)
    total_states = grid_points

    unique_steps, counts = np.unique(
        steps_to_stabilize[reachable].astype(int),
        return_counts=True)

    print(f"\n--- minimum stabilization steps by state ({total_states} states) ---")

    for step_val, count in zip(unique_steps, counts):
        pct = 100 * count / total_states
        print(f"  {step_val}-step: {pct:.2f}% ({count}/{total_states})")

    # States for which no action sequence was found
    unreachable_count = total_states - int(np.sum(counts))
    pct_unreachable = 100 * unreachable_count / total_states

    print(f"  unreachable: {pct_unreachable:.2f}% "
          f"({unreachable_count}/{total_states})")

    masked_steps = np.ma.masked_invalid(per_action_steps)

    max_steps = int(np.nanmax(per_action_steps))
    heat_cmap = plt.get_cmap("viridis", max_steps).copy()
    heat_cmap.set_bad(color="white")
    heat_norm = BoundaryNorm(np.arange(0.5, max_steps + 1.5), heat_cmap.N)

    fig3, ax3 = plt.subplots(figsize=(8, 6), layout="constrained")
    mesh = ax3.pcolormesh(velocity_range, alpha_range, masked_steps.T, cmap=heat_cmap, norm=heat_norm, shading="nearest")
    cbar3 = fig3.colorbar(mesh, ax=ax3, label="steps to stabilize")
    cbar3.set_ticks(range(1, max_steps + 1))

    dv = velocity_range[1] - velocity_range[0]
    da = alpha_range[1] - alpha_range[0]
    for v in range(grid_points):
        if reachable[v]:
            m_star = chosen_alpha_idx[v]
            ax3.add_patch(plt.Rectangle(
                (velocity_range[v] - dv / 2, alpha_range[m_star] - da / 2),
                dv, da, facecolor="black", alpha=0.25, edgecolor="none"
            ))
    ax3.set_ylim(alpha_range[0], alpha_range[-1])
    ax3.set_xlabel(r"$\dot\theta$ at $\theta=0$")
    ax3.set_ylabel(r"angle of attack $\alpha$ (rad)")
    ax3.set_title("Step count by (state, action)")

    plt.savefig(output / "state_action_heatmap.png")

if run_control_simulation:
    policy_sim_params = model.generate_params()
    policy_path = output / "state_action_policy.npz"

    cached = np.load(policy_path)
    alpha_control = {
        "velocity_range": cached["velocity_range"],
        "alpha_range": cached["alpha_range"],
        "steps_to_stabilize": cached["steps_to_stabilize"],
        "alpha_to_apply": cached["alpha_to_apply"],
    }

    v0 = initial_state[1]
    idx = int(np.argmin(np.abs(alpha_control["velocity_range"] - v0)))
    chosen_m = int(alpha_control["alpha_to_apply"][idx, 0])
    policy_sim_params["angle_of_attack"] = alpha_control["alpha_range"][chosen_m]

    (time_traj, state_traj, torque_traj, final_state, completed_steps, velocity_tracker) = full_integration(initial_state, coarse_timestep, fine_timestep, sim_time, None, policy_sim_params, True, alpha_control)

    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 6), sharex=True, layout="constrained")
    ax1.plot(time_traj, state_traj[0], label=r"$\theta$")
    ax1.plot(time_traj, state_traj[1], label=r"$\dot\theta$")
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylabel("state (rad or rad/s)")
    ax1.set_ylabel("time (s)")
    ax1.set_title(f"Simulation with Alpha Control (step count={completed_steps})")

    plt.savefig(output / "simulation.png")
else:
    (time_traj, state_traj, torque_traj, final_state, completed_steps, velocity_tracker) = full_integration(initial_state, coarse_timestep, fine_timestep, sim_time, None, params, True, None)
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 6), sharex=True, layout="constrained")
    ax1.plot(time_traj, state_traj[0], label=r"$\theta$")
    ax1.plot(time_traj, state_traj[1], label=r"$\dot\theta$")
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylabel("state (rad or rad/s)")
    ax1.set_ylabel("time (s)")
    ax1.set_title(f"Simulation without Alpha Control (step count={completed_steps})")

    plt.savefig(output / "simulation.png")

print(f"final_state={final_state}, footstrikes={completed_steps}")

if validate_state_action_table:

    policy_path = output / "state_action_policy.npz"

    cached = np.load(policy_path)
    velocity_range   = cached["velocity_range"]
    alpha_range      = cached["alpha_range"]
    steps_to_stabilize = cached["steps_to_stabilize"]
    alpha_to_apply     = cached["alpha_to_apply"]

    alpha_control = {
        "velocity_range": velocity_range,
        "alpha_range": alpha_range,
        "steps_to_stabilize": steps_to_stabilize,
        "alpha_to_apply": alpha_to_apply,
    }

    params_copy = model.generate_params()
    mass, length, gravity = params_copy["mass"], params_copy["length"], params_copy["gravity"]

    n_test = 17
    test_velocity_range = np.linspace(0.25, 4.25, 17)

    predicted_steps = np.full(n_test, np.nan)
    actual_steps    = np.full(n_test, np.nan)
    actual_final    = np.empty(n_test, dtype=object)
    match           = np.zeros(n_test, dtype=bool)

    for i, v0 in enumerate(test_velocity_range):
        # look up what the table predicts for this velocity
        table_idx = find_velocity_index(v0, velocity_range)
        predicted = steps_to_stabilize[table_idx] if table_idx >= 0 else np.nan
        predicted_steps[i] = predicted

        # actually simulate this velocity under the policy
        initial_state = [0, v0]
        sim_params = model.generate_params()

        if table_idx >= 0 and not np.isnan(alpha_to_apply[table_idx, 0]):
            chosen_m = int(alpha_to_apply[table_idx, 0])
            sim_params["angle_of_attack"] = alpha_range[chosen_m]

        (_, _, _, final_state, completed_steps, velocity_tracker) = full_integration(
            initial_state, coarse_timestep, fine_timestep, sim_time,
            None, sim_params, True, alpha_control
        )
        actual_final[i] = final_state

        if final_state == "stabilized_upright":
            actual_steps[i] = completed_steps
            match[i] = (not np.isnan(predicted)) and (completed_steps == predicted)
        else:
            match[i] = np.isnan(predicted)  # both agree: never stabilizes

        pred_str = "nan" if np.isnan(predicted) else f"{predicted:.0f}"
        act_str  = "nan" if np.isnan(actual_steps[i]) else f"{actual_steps[i]:.0f}"

    n_match = int(np.sum(match))
    print(f"Correct step counts: {n_match}/{n_test} ({100 * n_match / n_test:.2f}%)")

if create_animation:
    print("animating")

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")

    def draw_frame(index):
        # The massless swing leg is repositioned instantaneously at each impact.
        params["ankle_torque"] = torque_traj[index]
        model.visualize(state_traj[:, index], params, ax=ax)
        ax.set_title(f"t = {time_traj[index]:.2f} s")


    # Simulate at a small timestep, but render only 25 frames per second.
    fps = 20
    target_times = np.arange(0, time_traj[-1], 1 / fps)
    frame_indices = np.searchsorted(time_traj, target_times)
    if frame_indices[-1] != time_traj.size - 1:
        frame_indices = np.append(frame_indices, time_traj.size - 1)

    animation = FuncAnimation(
        fig, draw_frame, frames=frame_indices, interval=1000 / fps, repeat=False
    )

    animation.save(output / "walker.gif", writer=PillowWriter(fps=fps))

    # To save an MP4 instead, install FFmpeg and use:
    # animation.save(output / "walker.mp4", writer="ffmpeg", fps=fps)
    print(f"Saved {output / 'walker.gif'} ({completed_steps} footstrikes).")

plt.show()
