from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from models import inverted_pendulum_walker as model
from integrators import rk4 as integrator

output = Path("output/assignment_2")
output.mkdir(parents=True, exist_ok=True)

params = model.generate_params()


fine_timestep = 1e-4
coarse_timestep = 1e-2
sim_time = 15.0
desired_number_of_steps = 6

simulate_RoA_sweep = 0
create_animation = 0
return_map = 1

def full_integration(initial_state, coarse_timestep, fine_timestep, sim_time, desired_number_of_steps, params):
    time_traj = [0.0]
    state_traj = [initial_state]
    torque_traj = [params["ankle_torque"]]
    velocity_tracker = []

    inclination = params["incline"]
    alpha = params["angle_of_attack"]

    t = 0.0
    completed_steps = 0
    final_state = "time_limit_reached"

    while t < sim_time:
        state = state_traj[-1]

        impact_angle = inclination + alpha

        timestep = fine_timestep if state[0]-impact_angle > 0.1 else coarse_timestep

        next_state = integrator.integrate_step(model, state, t, timestep, params)
        t += timestep

        step_impact, failure = model.impact_guard(next_state, params)
        if step_impact:
            next_state[0] = inclination - alpha
            next_state[1] = next_state[1] * np.cos(2*alpha)
            completed_steps += 1

        if model.zero_crossing_guard(state, next_state, step_impact):
            velocity_tracker.append(next_state[1])

        state_traj.append(next_state)
        time_traj.append(t)

        params["ankle_torque"] = model.feedback_guard(next_state, params)
        torque_traj.append(params["ankle_torque"])

        if len(state_traj) % 50 == 0 and np.all(np.abs(np.array(state_traj)[-100:, 1]) < 0.005):
            # if np.all(np.abs(np.array(state_traj)[-20:, 0]) < 0.01):
            final_state = "stabilized_upright"

            # else:
            #     final_state = "stabilized_elsewhere"
            break
        elif failure:
            final_state = "fell_over"
            break
        elif desired_number_of_steps is not None and completed_steps == desired_number_of_steps:
            final_state = "step_limit_reached"
            break


    #print(final_state)

    time_traj = np.array(time_traj)
    state_traj = np.array(state_traj).T
    torque_traj = np.array(torque_traj)

    return time_traj, state_traj, torque_traj, final_state, completed_steps, velocity_tracker

params = model.generate_params()
initial_state = np.array([-0.21, 0.84])

coarse_timestep = 1e-3
time_traj, state_traj, torque_traj, final_state, completed_steps, velocity_tracker = full_integration(initial_state, coarse_timestep, fine_timestep, sim_time, 2, params)

if return_map:
    inclination = params["incline"]
    velocity_magnitudes = np.linspace(0.05, 0.5, 40)

    poincare_x = []  # v_k  (incoming velocity, swept)
    poincare_y = []  # v_k+1 (velocity at the very next crossing)

    initial_conditions = []
    for v_mag in velocity_magnitudes:
        initial_conditions.append((-0.1, v_mag))    # approaching theta=0 from below
        initial_conditions.append((0.1, -v_mag))    # approaching theta=0 from above

    for theta0, v0 in initial_conditions:
        initial_state_v = np.array([theta0, v0])
        sweep_params = model.generate_params()

        _, _, _, final_state, _, velocity_tracker = full_integration(
            initial_state_v, coarse_timestep, fine_timestep, sim_time, None, sweep_params
        )

        if len(velocity_tracker) < 1:
            continue  # never even crossed theta=0

        poincare_x.append(v0)
        poincare_y.append(velocity_tracker[0])  # first crossing = one map iteration

    plt.figure(4)
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(True)
    plt.scatter(poincare_x, poincare_y, s=20)

    lims = [min(poincare_x, default=-1), max(poincare_x, default=1)]
    plt.plot(lims, lims, linewidth=0.5, label="identity line")
    plt.xlabel(r"$\dot\theta_k$ (incoming)")
    plt.ylabel(r"$\dot\theta_{k+1}$ (next crossing)")
    plt.title(
        f"Poincare section (alpha={np.degrees(params['angle_of_attack']):.1f} deg, "
        f"{np.degrees(inclination):.1f} deg incline)\n"
    )
    plt.tight_layout()
    plt.legend()
    plt.savefig("figures/return_map_plot.png")
    plt.show()

if simulate_RoA_sweep:
    RoA_grid_points = 60
    angle_range = np.linspace(-0.3, 0.3, RoA_grid_points)
    velocity_range = np.linspace(-0.3, 0.3, RoA_grid_points)

    categories = ["stabilized_upright", "fell_over", "step_limit_reached"]
    category_codes = {name: idx for idx, name in enumerate(categories)}
    sweep_results = np.zeros((RoA_grid_points, RoA_grid_points))

    print("sweeping")
    for i, angle0 in enumerate(angle_range):
        for j, velocity0 in enumerate(velocity_range):

            sweep_params = model.generate_params()
            initial_state_sweep = [angle0, velocity0]
            _, state_traj_sweep, _, final_state, _,_= full_integration(initial_state_sweep, coarse_timestep, fine_timestep, sim_time, 1, sweep_params)

            sweep_results[j, i] = category_codes[final_state]

        print(f"  row {i + 1}/{RoA_grid_points} done")

    cmap = plt.get_cmap("Blues", len(categories))
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    mesh = ax.pcolormesh(
    angle_range, velocity_range, sweep_results,
    cmap=cmap, vmin=-0.5, vmax=len(categories) - 0.5, shading="nearest",
    )
    cbar = fig.colorbar(mesh, ax=ax, ticks=range(len(categories)))
    cbar.ax.set_yticklabels(["Stablizes upright","Falls backwards","Takes another step"])
    ax.set(
    xlabel=r"$\theta_0$ (rad)",
    ylabel=r"$\dot\theta_0$ (rad/s)",
    )
    x = np.linspace(-0.3, 0.3, 10)
    y = x*-3.075 -0.10
    y2 = x*-3.075 +0.1
    ax.plot(x,y)
    ax.plot(x,y2)

    title="Ankle controller region of attraction",

    #fig.savefig(output / "roa_sweep.png", dpi=150)
    plt.show()





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
