import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from models import rimlessWheel as model

#what to compute/plot. Can plot RoA stuff from existing data w/o running
plot_energy = 1                                 #energy over time
simulate_single_RoA = 0;  plot_single_RoA = 1   #state space RoA plot for given slope/spoke #
plot_return_map = 1                             #1D return map plot
estimate_floquet_multiplier = 1                 #rolling cycle floquet multiplier

sweep_floquet_inclination = 0                   #sweep floquet mult. over inclinations
sweep_floquet_spokes = 0                        #sweep floquet mult. over spoke #

sweep_RoA_inclination = 0;  plot_RoA_inclination = 1     #sweep RoA over inclinations
sweep_RoA_spokes = 0;       plot_RoA_spokes = 1          #sweep RoA over spoke #

#Setup/parameters
impact_threshold = 0.05 #when to switch to finer timestep
coarse_timestep = 1e-2; fine_timestep = 1e-4; floquet_timestep = 5e-6

mass = 1; length = 1; gravity = 9.81 #wheel parameters
spoke_number = 6
inclination_angle = np.pi/18

sim_time = 5.0
initial_state = np.array([0, 1.5*np.pi, 0]) #[initial angle, initial velocity, impact counter = 0]



params = {"mass": mass, "length": length, "gravity": gravity, "spoke_number": spoke_number, "inclination_angle": inclination_angle}
alpha = np.pi/spoke_number

if plot_energy:
    time_traj, state_traj, _ = model.full_integration(
        model, initial_state, coarse_timestep, fine_timestep, impact_threshold, sim_time, params, keep_history=True)

    kinetic_energy, potential_energy, _ = model.calculate_energy(state_traj, params)

    plt.figure(1)
    plt.plot(time_traj, potential_energy, label="Potential energy")
    plt.plot(time_traj, kinetic_energy, label="Kinetic energy")
    plt.plot(time_traj, potential_energy + kinetic_energy, label="Total energy")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (J)")
    plt.title(f"Rimless Wheel Energy ({spoke_number} spokes, {np.degrees(inclination_angle):.1f} deg incline)\n"
              f"IC: θ={np.degrees(initial_state[0]):.1f} deg, θ̇={np.degrees(initial_state[1]):.1f} deg/s")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/energy_plot.png")

if simulate_single_RoA:
    grid_points_RoA = 75 #adjust resolution
    
    alpha = np.pi/spoke_number
    angle_pos = inclination_angle + alpha
    angle_neg = inclination_angle - alpha

    #adjust state space grid bounds to simulate within
    inital_angles = np.linspace(angle_neg, angle_pos, grid_points_RoA)
    inital_velocities = np.linspace(-4, 4, grid_points_RoA)

    steady_state = np.zeros([grid_points_RoA, grid_points_RoA])

    loop_count = 0

    print("Looping for single RoA plot")
    for i in range(grid_points_RoA):
        theta_dot0 = inital_velocities[i]

        for j in range(grid_points_RoA):
            theta0 = inital_angles[j]

            initial_state_RoA = np.array([theta0, theta_dot0, 0])

            final_state, _ = model.full_integration(
                model, initial_state_RoA, coarse_timestep, fine_timestep, impact_threshold, None, params, keep_history=False)

            if abs(final_state[1]) > 0.1:
                steady_state[i, j] = 1

            loop_count += 1
        print(loop_count, " out of ", grid_points_RoA**2, "states simulated")

    np.savez("data/roa_plot.npz",
             steady_state=steady_state,
             inital_angles=inital_angles,
             inital_velocities=inital_velocities)
if plot_single_RoA:
    data = np.load("data/roa_plot.npz")
    steady_state = data["steady_state"]
    inital_angles = data["inital_angles"]
    inital_velocities = data["inital_velocities"]

    cmap = ListedColormap(["#000000", "#ffffff"])

    plt.figure(3)
    plt.imshow(steady_state, origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=1,
        extent=[inital_angles.min(), inital_angles.max(), inital_velocities.min(), inital_velocities.max()])
    plt.xlabel('Angle (rad)')
    plt.ylabel('Angular velocity (rad/s)')
    plt.title(f'Steady state map ({spoke_number} spokes, {np.degrees(inclination_angle):.1f} deg incline)')

    cbar = plt.colorbar(ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(['Stops', "Rolls Forever"])
    plt.savefig("figures/single_roa_plot.png")

if plot_return_map:
    _, _, poincare_tracker = model.full_integration(
        model, initial_state, coarse_timestep, floquet_timestep, impact_threshold, sim_time, params, keep_history=True
    )
    poincare_tracker = np.array(poincare_tracker)

    poincare_x = poincare_tracker[:-1]
    poincare_y = poincare_tracker[1:]

    print("Steady state post-impact velocity:", np.mean(poincare_tracker[-5:]), "rad/s")

    plt.figure(4)
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(True)
    plt.scatter(poincare_x, poincare_y)
    plt.scatter(np.mean(poincare_tracker[-5:]), np.mean(poincare_tracker[-5:]), label="fixed point")
    plt.plot(poincare_x, poincare_x, linewidth=0.5, label="identity line")
    plt.xlabel("x_k")
    plt.ylabel("x_k+1")
    plt.title(f"Poincare Section ({spoke_number} spokes, {np.degrees(inclination_angle):.1f} deg incline)\n"
              f"IC: θ={np.degrees(initial_state[0]):.1f} deg, θ̇={np.degrees(initial_state[1]):.1f} deg/s")
    plt.tight_layout()
    plt.legend()
    plt.savefig("figures/return_map_plot.png")

if estimate_floquet_multiplier:
    fixed_point_velocity = model.find_fixed_point(params, coarse_timestep, floquet_timestep, impact_threshold, sim_time, initial_velocity=0.5)
    perturbance = 0.01
    floquet_multiplier = model.estimate_floquet_multiplier(params, coarse_timestep, floquet_timestep, impact_threshold, fixed_point_velocity, perturbance)
    print("Fixed point velocity:", fixed_point_velocity)
    print("Floquet Multiplier:", floquet_multiplier)

if sweep_floquet_inclination:
    inclination_values = np.linspace(0, np.pi/2, 21) #toggle sweep range/resolution
    
    floquet_sweep_inclination = []
    perturbance = 0.01
    
    print("Sweeping inclinations for Floquet multiplier")

    for k, inclination in enumerate(inclination_values):
        params["inclination_angle"] = inclination

        fixed_point_velocity = model.find_fixed_point(params, coarse_timestep, floquet_timestep, impact_threshold, sim_time = None, initial_velocity=np.pi)

        if np.isnan(fixed_point_velocity):
            multiplier_inclination = np.nan
        else:
            multiplier_inclination = model.estimate_floquet_multiplier(params, coarse_timestep, floquet_timestep, impact_threshold, fixed_point_velocity, perturbance)

        floquet_sweep_inclination.append(multiplier_inclination)
        print(f"Inclination={np.degrees(inclination):.1f} deg, v*={fixed_point_velocity:.4f} rad/s, Floquet multiplier={multiplier_inclination:.4f}")

    floquet_sweep_inclination = np.array(floquet_sweep_inclination)

    plt.figure(6)
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(True)
    plt.scatter(np.degrees(inclination_values), floquet_sweep_inclination)
    plt.xlabel("Inclination")
    plt.ylabel("Floquet Multiplier")
    plt.title(f"Floquet Multiplier vs Inclination ({spoke_number} spokes)")
    plt.ylim(0.1, 0.8)
    plt.tight_layout()
    plt.savefig("figures/slope_sweep_floquet_map_plot.png")

if sweep_floquet_spokes:
    spoke_values = [6, 7, 8, 9, 10, 11, 12] #adjust spokes counts to sweep over

    floquet_spoke_sweep = []
    perturbance = 0.01

    for l, spoke in enumerate(spoke_values):
        params["spoke_number"] = spoke

        fixed_point_velocity = model.find_fixed_point(params, coarse_timestep, floquet_timestep, impact_threshold, None, initial_velocity=0.3)

        if np.isnan(fixed_point_velocity):
            print(f"l={l}, spoke={spoke:.1f}: no steady rolling found")
            floquet_spoke_sweep.append(np.nan)
            multiplier_spokes = np.nan
        else:
            multiplier_spokes = model.estimate_floquet_multiplier(params, coarse_timestep, floquet_timestep, impact_threshold, fixed_point_velocity, perturbance)
            floquet_spoke_sweep.append(multiplier_spokes)

        print(f"Spokes={spoke:.0f}, v*={fixed_point_velocity:.4f} rad/s, Floquet multiplier={multiplier_spokes:.4f}")

    floquet_spoke_sweep = np.array(floquet_spoke_sweep)

    plt.figure(7)
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(True)
    plt.scatter(spoke_values, floquet_spoke_sweep)
    plt.xlabel("Spokes")
    plt.ylabel("Floquet Multiplier")
    plt.title(f"Floquet Multiplier vs Spoke Count ({np.degrees(inclination_angle):.1f} deg incline)")
    plt.ylim(0.1, 0.8)
    plt.tight_layout()
    plt.savefig("figures/spoke_sweep_floquet_plot.png")

if sweep_RoA_inclination:
    inclination_values = np.linspace(0, np.pi/5, 19) #adjust slope bounds/resolution of sweep

    RoA_percent_rolling = []

    for k, inclination in enumerate(inclination_values):
        print("Simulating for inclination =", np.degrees(inclination_values[k]), "degrees")

        params["inclination_angle"] = inclination

        RoA_percent_rolling.append(
            model.estimate_RoA_fraction(params, coarse_timestep, fine_timestep, impact_threshold, None, grid_points=25)
        )

    RoA_percent_rolling = np.array(RoA_percent_rolling)

    np.savez("data/inclination_roa_sweep.npz", inclination_values=inclination_values, RoA_percent_rolling=RoA_percent_rolling)
if plot_RoA_inclination:
    data = np.load("data/inclination_roa_sweep.npz")
    inclination_values = data["inclination_values"]
    RoA_percent_rolling = data["RoA_percent_rolling"]

    plt.figure(9)
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(True)
    plt.scatter(np.degrees(inclination_values), RoA_percent_rolling * 100)
    plt.xlabel("Inclination (deg)")
    plt.ylabel("Percentage")
    plt.title(f"Percentage of states reaching steady state rolling ({spoke_number} spokes)")
    plt.tight_layout()
    plt.savefig("figures/inclination_roa_sweep.png")

if sweep_RoA_spokes:
    spoke_values = [6, 7, 8, 9, 10, 11, 12] #adjust spokes counts to sweep over

    RoA_percent_rolling_spokes = []

    for m, spoke in enumerate(spoke_values):
        print("Simulating for spoke count =", spoke)

        params["spoke_number"] = spoke

        RoA_percent_rolling_spokes.append(
            model.estimate_RoA_fraction(params, coarse_timestep, fine_timestep, impact_threshold, sim_time, grid_points=25)
        )

    RoA_percent_rolling_spokes = np.array(RoA_percent_rolling_spokes)

    np.savez("data/spoke_roa_sweep.npz", spoke_values=np.array(spoke_values),
             RoA_percent_rolling_spokes=RoA_percent_rolling_spokes)
if plot_RoA_spokes:
    data = np.load("data/spoke_roa_sweep.npz")
    spoke_values = data["spoke_values"]
    RoA_percent_rolling_spokes = data["RoA_percent_rolling_spokes"]

    plt.figure(50)
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(True)
    plt.scatter(spoke_values, RoA_percent_rolling_spokes * 100)
    plt.xlabel("Number of spokes")
    plt.ylabel("Percentage")
    plt.title(f"Percentage of states reaching steady state rolling ({np.degrees(inclination_angle):.1f} deg incline)")
    plt.tight_layout()
    plt.savefig("figures/spoke_roa_sweep.png")
plt.show()