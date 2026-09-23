## Sketches of System

![](output/assignment_2/Markdown/sketch.jpg)

## Region of Attraction for Ankle Controller

My torque controller was the following:

```python
if abs(state[1]) < 0.01:
    applied_torque = -(gravity * np.sin(angle)) / length - 2.5 * angle - 1 * velocity
else:
    applied_torque = -(gravity * np.sin(angle)) / length - 3 * velocity
```

When angular velocity was high, the controller focused on reducing the speed. Once the system was largely stabilized, it used torque based on the angle error to converge to $\theta = 0$. It had the following region of attraction:

![](output/assignment_2/Markdown/roa_sweep.png)

## Poincaré Section

Since the impact angle depended on $\alpha$, I chose the Poincaré section to be at $\theta = 0$. This choice of section was transverse to the flow and easily understandable, as it also represented the desired equilibrium of the system. The resulting return map is shown below:

![](output/assignment_2/Markdown/return_map_plot.png)

The fixed point is located at $(0,0)$. A semi-linear decrease in velocity can be observed for $\theta_k$ of roughly $1.25$ rad/s and greater. Below this point, there is a sudden jump to significantly lower $\theta_{k+1}$ values. This likely indicates the range of velocities that will converge in the next step, with the small but non-zero velocities between approximately $1$ and $1.25$ rad/s indicating conditions where the pendulum swing crosses $0$ again but is stabilized instead of taking another step.

## Grid Resolution

For my state-action table, I verified the grid resolution by simulating 17 velocities ($0.25, 0.5, 0.75, \ldots, 4.25$ rad/s) and checking the number of steps needed to stabilize against the predicted number of steps. The results showed that a $75 \times 75$ grid was needed to obtain the correct result across all velocities.

| Grid Resolution | Correctly Predicted Step Counts | Accuracy |
| --------------: | ------------------------------: | -------: |
|              20 |                           14/17 |   82.35% |
|              30 |                           14/17 |   82.35% |
|              45 |                           14/17 |   82.35% |
|              60 |                           16/17 |   94.12% |
|              75 |                           17/17 |  100.00% |

The following plot shows the optimal angle to be taken for each velocity state and the resulting step count. The controller aims for the middle angle out of all suitable angles leading to a minimum step count. This was done to avoid choosing angles on the boundary of resulting step counts, which may result in an extra step being taken.

![](output/assignment_2/Markdown/state_action_lookup.png)

An initial condition that requires at least 3 steps to come to a standstill is $[0,4]$. With angle-of-attack control, this stabilizes within 3 steps, as predicted by the above plot. Without angle-of-attack control, and with $\alpha$ fixed at the minimum value of $\pi/8$ for the entire simulation (this should result in the most steps possible due to minimal energy loss), the system instead takes 5 steps before stabilizing.

The two trajectories are shown below.

![](output/assignment_2/Markdown/sim_with_control.png)

![](output/assignment_2/Markdown/sim_without_control.png)
