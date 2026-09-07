import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from models import rimlessWheel as model
from integrators import rk4 as integrator
import rimlessWheelAnalysis as analysis

#what plots/outputs to create
energy_plot = 0
hub_height_plot = 0
angular_velocity_plot = 0
RoA_plot = 0                    #takes a while
poincare_plot = 0
floquet_estimate = 0
inclination_floquet_sweep = 1
inclination_RoA_sweep = 0
spoke_floquet_sweep = 1
spoke_RoA_sweep = 0


timestep = 5e-4
sim_time = 5.0

params = model.generate_params()
initial_state = np.array([np.pi/6, 0, 0])
poincare_tracker = []

#time_traj, state_traj, poincare_tracker = integrator.integrate(model, initial_state, timestep, sim_time, params, poincare_tracker)

if energy_plot == True:
    kinetic_energy, potential_energy, _ = model.calculate_energy(state_traj, params)
    total_energy = potential_energy + kinetic_energy

    plt.figure(1)
    plt.plot(time_traj, potential_energy, label="Potential energy")
    plt.plot(time_traj, kinetic_energy, label="Kinetic energy")
    plt.plot(time_traj, potential_energy + kinetic_energy, label="Total energy")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (J)")
    plt.title("Rimless Wheel Energy")
    plt.legend()
    plt.tight_layout()

if hub_height_plot == True:
    _, _, hub_height = model.calculate_energy(state_traj, params)

    plt.figure(2)
    plt.plot(time_traj, hub_height, label="Height")
    plt.xlabel("Time (s)")
    plt.ylabel("Height (m)")
    plt.title("Height over time")
    plt.legend()
    plt.tight_layout()

if angular_velocity_plot == True:
    plt.figure(5)
    plt.plot(time_traj, state_traj[1])
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Velocity (rad/s)")
    plt.title("Angular Velocity over time")
    plt.tight_layout()

if RoA_plot == True:
    spoke_number = params["spoke_number"]
    inclination_angle = params["inclination_angle"]

    alpha = np.pi/spoke_number

    angle_pos = inclination_angle + alpha
    angle_neg = inclination_angle - alpha

    grid_points_RoA = 20

    inital_angles = np.linspace(angle_neg, angle_pos, grid_points_RoA)
    inital_velocities = np.linspace(-np.pi/10, np.pi/10, grid_points_RoA)

    steady_state = np.zeros([grid_points_RoA, grid_points_RoA])

    loop_count = 0

    print("looping for RoA plot")
    for i in range(len(inital_velocities)):
        theta_dot0 = inital_velocities[i]
        for j in range(len(inital_angles)):
            theta0 = inital_angles[j]

            initial_state_RoA = np.array([theta0, theta_dot0, 0])  # 0 = impact_count

            time_traj_RoA, state_traj_RoA, _ = integrator.integrate(model, initial_state_RoA, timestep, sim_time, params, [])

            if state_traj_RoA[1, -1:] > 0.1:
                steady_state[i, j] = 1 #rolls forever

            loop_count +=1
        print(loop_count)

    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#000000","#ffffff"])  # color for 0, color for 1

    plt.figure(3)
    plt.imshow(steady_state, origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=1,
        extent=[inital_angles.min(), inital_angles.max(), inital_velocities.min(), inital_velocities.max()])
    plt.xlabel('Angle (rad)')
    plt.ylabel('Angular velocity (rad/s)')
    plt.title('Steady state map')

    cbar = plt.colorbar(ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(['Stops', "Rolls Forever"])

if poincare_plot == True:
    poincare_x = poincare_tracker[:-1] #vel at crossing
    poincare_y = poincare_tracker[1:]   #vel at next crossing

    print("Steady state post-impact velocity:", np.mean(poincare_tracker[-5:]), "rad/s")

    plt.figure(4)
    plt.scatter(poincare_x, poincare_y)
    plt.plot(poincare_x,poincare_x)
    plt.xlabel("x_k")
    plt.ylabel("x_k+1")
    plt.title("Poincare Section")
    plt.tight_layout()

if floquet_estimate == True:
    fixed_point_velocity = analysis.find_fixed_point(params, timestep, sim_time=5.0, initial_velocity=0.3)
    perturbance = 0.01
    floquet_multiplier = analysis.estimate_floquet_multiplier(params, timestep, fixed_point_velocity, perturbance)
    print("Floquet Multiplier:", floquet_multiplier)

if inclination_floquet_sweep == True:
    inclination_values = np.linspace(0, np.pi/2, 11)
    floquet_sweep_inclination = []
    perturbance = 0.01
    print("sweeping")

    for k, inclination in enumerate(inclination_values):
        params = model.generate_params()
        params["inclination_angle"] = inclination

        fixed_point_velocity = analysis.find_fixed_point(params, timestep, sim_time=5.0, initial_velocity=0.3)

        if np.isnan(fixed_point_velocity):
            multiplier_inclination = np.nan
        else:
            floquet_timestep = 1e-5
            multiplier_inclination = analysis.estimate_floquet_multiplier(params, floquet_timestep, fixed_point_velocity, perturbance)

        floquet_sweep_inclination.append(multiplier_inclination)
        print(f"Inclination={np.degrees(inclination):.1f} deg, " f"v*={fixed_point_velocity:.4f}, Floquet multiplier={multiplier_inclination}")
        
    floquet_sweep_inclination = np.array(floquet_sweep_inclination)

    plt.figure(6)
    plt.grid()
    plt.scatter(np.degrees(inclination_values), floquet_sweep_inclination)
    plt.xlabel("Inclination")
    plt.ylabel("Floquet Multiplier")
    plt.title("Floquet Multiplier vs Inclination")
    plt.ylim(0, 0.8)
    plt.tight_layout()

if inclination_RoA_sweep == True:
    inclination_values = np.linspace(0, np.pi/5, 21)
    RoA_percent_rolling = []

    for k, inclination in enumerate(inclination_values):
        print("RoA k =", k)
        params = model.generate_params()
        params["inclination_angle"] = inclination
        spoke_number = params["spoke_number"]
        alpha = np.pi / spoke_number

        angle_pos = inclination + alpha
        angle_neg = inclination - alpha

        grid_points_RoA = 20

        inital_angles = np.linspace(angle_neg, angle_pos, grid_points_RoA)
        inital_velocities = np.linspace(-np.pi/10, np.pi/10, grid_points_RoA)

        steady_state = np.zeros([grid_points_RoA, grid_points_RoA])

        for i in range(grid_points_RoA):
            theta_dot0 = inital_velocities[i]
            for j in range(grid_points_RoA):
                theta0 = inital_angles[j]

                initial_state_RoA = np.array([theta0, theta_dot0, 0])
                _, state_traj_RoA, _ = integrator.integrate(model, initial_state_RoA, timestep, sim_time, params, [])

                if abs(state_traj_RoA[1, -1]) > 0.1:
                    steady_state[i, j] = 1

        RoA_percent_rolling.append(np.sum(steady_state == 1) / (grid_points_RoA**2))

        cmap = ListedColormap(["#000000", "#ffffff"])

        plt.figure(10 + k)
        plt.imshow(steady_state, origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=1, extent=[inital_angles.min(), inital_angles.max(), inital_velocities.min(), inital_velocities.max()])
        plt.xlabel('Angle (rad)')
        plt.ylabel('Angular velocity (rad/s)')
        plt.title(f'Steady state map for inclination {np.degrees(inclination):.1f} deg')

        cbar = plt.colorbar(ticks=[0.25, 0.75])
        cbar.ax.set_yticklabels(['Stops', "Rolls Forever"])

    RoA_percent_rolling = np.array(RoA_percent_rolling)

    plt.figure(9)
    plt.scatter(np.degrees(inclination_values), RoA_percent_rolling * 100)
    plt.xlabel("Inclination (deg)")
    plt.ylabel("Percentage")
    plt.title("Percentage of states reaching steady state rolling")
    plt.tight_layout()


if spoke_floquet_sweep == True:
    spoke_values = [6, 7, 8, 9, 10, 11, 12]
    floquet_spoke_sweep = []
    perturbance = 0.01

    for l, spoke in enumerate(spoke_values):
        params = model.generate_params()
        params["spoke_number"] = spoke

        fixed_point_velocity = analysis.find_fixed_point(params, timestep = 5e-4, sim_time=5.0, initial_velocity=0.3)

        if fixed_point_velocity == np.nan:
            print(f"l={l}, spoke={spoke:.1f}: no steady rolling found")
            floquet_spoke_sweep.append(np.nan)
        else:
            timestep = 1e-5
            multiplier_spokes = analysis.estimate_floquet_multiplier(params, timestep, fixed_point_velocity, perturbance)
            floquet_spoke_sweep.append(multiplier_spokes)

        print(f"Spokes={spoke:.1f}, " f"v*={fixed_point_velocity:.4f}, Floquet multiplier={multiplier_spokes}")

    floquet_spoke_sweep = np.array(floquet_spoke_sweep)

    plt.figure(7)
    plt.grid()
    plt.scatter(spoke_values, floquet_spoke_sweep)
    plt.xlabel("Spokes")
    plt.ylabel("Floquet Multiplier")
    plt.title("Floquet Multiplier vs Spoke Count")
    plt.ylim(0, 0.8)
    plt.tight_layout()

if spoke_RoA_sweep == True:
    spoke_values = [6, 7, 8, 9, 10, 11, 12]
    RoA_percent_rolling_spokes = []

    for m, spoke in enumerate(spoke_values):
        params = model.generate_params()
        params["spoke_number"] = spoke
        inclination_angle = params["inclination_angle"]

        alpha = np.pi / spoke

        angle_pos = inclination_angle + alpha
        angle_neg = inclination_angle - alpha

        grid_points_RoA = 20

        inital_angles = np.linspace(angle_neg, angle_pos, grid_points_RoA)
        inital_velocities = np.linspace(-np.pi/10, np.pi/10, grid_points_RoA)

        steady_state = np.zeros([grid_points_RoA, grid_points_RoA])

        for i in range(grid_points_RoA):
            theta_dot0 = inital_velocities[i]
            for j in range(grid_points_RoA):
                theta0 = inital_angles[j]

                initial_state_RoA = np.array([theta0, theta_dot0, 0])
                _, state_traj_RoA, _ = integrator.integrate(model, initial_state_RoA, timestep, sim_time, params, [])

                if abs(state_traj_RoA[1, -1]) > 0.1:
                    steady_state[i, j] = 1

        RoA_percent_rolling_spokes.append(np.sum(steady_state == 1) / (grid_points_RoA**2))

        cmap = ListedColormap(["#000000", "#ffffff"])

        plt.figure(40 + m)
        plt.imshow(steady_state, origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=1,
                   extent=[inital_angles.min(), inital_angles.max(), inital_velocities.min(), inital_velocities.max()])
        plt.xlabel('Angle (rad)')
        plt.ylabel('Angular velocity (rad/s)')
        plt.title(f'Steady state map for {spoke} spokes')

        cbar = plt.colorbar(ticks=[0.25, 0.75])
        cbar.ax.set_yticklabels(['Stops', "Rolls Forever"])

    RoA_percent_rolling_spokes = np.array(RoA_percent_rolling_spokes)

    plt.figure(50)
    plt.grid()
    plt.scatter(spoke_values, RoA_percent_rolling_spokes * 100)
    plt.xlabel("Number of spokes")
    plt.ylabel("Percentage")
    plt.title("Percentage of states reaching steady state rolling")
    plt.tight_layout()

plt.show()