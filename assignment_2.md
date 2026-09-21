**Sketches of system:**
![](output/assignment_2/Markdown/HW2sketch.jpg)

**Region of attraction for ankle controller:**
My torque controller was the following:

    if abs(state[1]) < 0.01:
        applied_torque = - (gravity * (angle)) / length - 2.5*angle - 1*velocity
    else:
        applied_torque = - (gravity * (angle)) / length - 3*velocity

When angular velocity was high, it focused on reducing speed, but once the system largely stabilized, it used torque based off out the angle error to converge on theta = 0. It had the following region of attraction:

![](output/assignment_2/Markdown/roa_sweep.png)

**Poincare Section:**
Since the impact angle depended on alpha, I chose the Poincare section to be at theta = 0. This choice of section was transverse to the flow, and easily understandable as it also represented the desired equilibrium of the system. The resulting return map is shown below:

![](output/assignment_2/Markdown/return_map_plot.png)

The fixed point is located at (0,0). A semi-linear decrease in velocity can be observed for theta_k of roughly 1.25rad/s and greater. Below this point, there is a sudden jump to significantly lower theta_k+1's: this likely indicates the range of velocities that will converge in the next step, with the small but non-zero velocities between ~1 and 1.25rad/s indication conditions where the pendulum swing crosses 0 again, but is stablized instead of taking another step.

**Grid Resolution**

For my state action table, I verified the grid resolution by simulating 17 velocities (0.25, 0.5, 0.75, ... 4.25 rad/s), and checking the number of steps needed to stabilize against the predicted number of steps. The results showed that a 75x75 grid was needed to obtain the correct result across all velocities. For coarser grids, the

| Grid Resolution | Correctly Predicted Step Counts | Accuracy |
| --------------: | ------------------------------: | -------: |
|              20 |                           11/17 |   64.71% |
|              30 |                           10/17 |   58.82% |
|              45 |                           12/17 |   70.59% |
|              60 |                           13/17 |   76.47% |
|              75 |                           17/17 |  100.00% |

This following plot shows the optimal angle to be taken for each velocity state and the resulting step count. The controller aims for the middle angle out of all suitable angles leading to a minimum step count. This was done to avoid choosing angles on the boundary of resulting step counts, which may result in an extra step being taken.

![](output/assignment_2/Markdown/state_action_lookup.png)

An initial condition that requires at least 3 steps to come to a standstill is [0, 3]. With angle of attack control, this stabilizes within 3 steps, as predicted by the above plot. Without angle of attack control, and alpha fixed at pi/8 for the entire simulation, the system instead can continue for 6 steps before stabilizing. The two trajectories are show below

![](output/assignment_2/Markdown/sim_with_control.png)

![](output/assignment_2/Markdown/sim_without_control.png)