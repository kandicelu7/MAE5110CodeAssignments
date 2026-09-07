import numpy as np
import matplotlib.pyplot as plt

from models import rimlessWheel as model
from integrators import rk4 as integrator

params = model.generate_params()

initial_state = np.array([0.3, 0, 0])
timestep = 1e-4
sim_time = 10.0

time_traj, state_traj = integrator.integrate(model, initial_state, timestep, sim_time, params)

kinetic_energy, potential_energy, hub_height = model.calculate_energy(state_traj, params)
total_energy = potential_energy + kinetic_energy

if abs(state_traj[1, -1]) < 0.001:
    print("Stopped")
else:
    print("Rolling")

plt.figure(1)
plt.plot(time_traj, potential_energy, label="Potential energy")
plt.plot(time_traj, kinetic_energy, label="Kinetic energy")
plt.plot(time_traj, potential_energy + kinetic_energy, label="Total energy")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Rimless Wheel Energy")
plt.legend()
plt.tight_layout()

# plt.show()

plt.figure(2)
plt.plot(time_traj, state_traj[1], label="Height")
plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (m)")
plt.title("Angular Velocity over time")
#plt.legend()
plt.tight_layout()

# plt.figure(3)
# plt.plot(state_traj[0], state_traj[1])
# plt.xlabel("x")
# plt.ylabel("x dot")
# plt.title("Phase Portait")
# #plt.legend()
# plt.tight_layout()

plt.show()