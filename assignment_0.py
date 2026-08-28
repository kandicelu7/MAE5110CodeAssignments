import numpy as np
import matplotlib.pyplot as plt
import timeit

#from models import pendulum as model
from models import bouncingBall as model

#from integrators import explicit_euler as integrator
from integrators import rk4 as integrator
# Basic simulation of the pendulum


# params = {
#     "gravity": 9.81,  # gravity m/s^2)
#     "length": 1,  # rod length (m)
#     "mass": 0.2,  # point mass at end of rod (kg)
#     "damping_coeff": 0.0,  # damping coefficient (kg*m^2/s)
# }
params = model.generate_params()

# some set-up
initial_state = np.array([np.pi / 4, 0.0])

timestep = 1e-4
sim_time = 10.0


###n_timesteps = int(sim_time / timestep) + 1
#time_traj = np.arange(n_timesteps) * timestep
#state_traj = np.zeros((2, n_timesteps))
#state_traj[:, 0] = initial_state

# simulation loop
#for step, t in enumerate(time_traj[:-1]):
#    state_traj[:, step + 1] = state_traj[:, step] + timestep * model.dynamics(
#        t, state_traj[:, step], params
#    )
time_traj, state_traj = integrator.integrate(model, initial_state, timestep, sim_time, params)

#Timing block
# blah = timeit.timeit(
#     lambda: integrator.integrate(initial_state, timestep, sim_time, params),
#     number=1
# )
# print("Time: ",blah)


# sanity check the energies: since there is no actuation, and no damping, total energy should stay
# constant. If we turn on the damping coefficient, it should slowly bleed out energy until it comes to
# a stand-still.

kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)
total_energy = potential_energy + kinetic_energy

print("Relative change in total energy: " + str((total_energy[-1]-total_energy[0])/total_energy[-1]))


plt.figure(1)
plt.plot(time_traj, potential_energy, label="Potential energy")
plt.plot(time_traj, kinetic_energy, label="Kinetic energy")
plt.plot(time_traj, potential_energy + kinetic_energy, label="Total energy")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Pendulum/Ball energy")
plt.legend()
plt.tight_layout()

plt.figure(2)
plt.plot(time_traj, state_traj[0], label="Height")
plt.xlabel("Time (s)")
plt.ylabel("Height (M)")
plt.title("Height over time")
plt.legend()
plt.tight_layout()


# TODO: make a phase portrait plot
plt.figure(3)
plt.plot(state_traj[0], state_traj[1])
plt.xlabel("x")
plt.ylabel("x dot")
plt.title("Phase Portait")
plt.legend()
plt.tight_layout()

plt.show()