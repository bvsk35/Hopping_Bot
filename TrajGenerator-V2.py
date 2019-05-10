#!/usr/bin/env python3
# coding: utf-8

# ### Import required libraries




import numpy as np
import matplotlib.pyplot as plt

from CartPole import CartPole
# from CartPole_GPS import CartPole_GPS

from ilqr.dynamics import constrain
from copy import deepcopy

from EstimateDynamics import local_estimate
from GMM import Estimated_Dynamics_Prior

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

from mujoco_py import load_model_from_path, MjSim, MjViewer
import mujoco_py

import time


# ### Formulate the iLQR problem



'''
1 - dt = time step
2 - N = Number of control points in the trajectory
3 - x0 = Initial state
4 - x_goal = Final state
5 - Q = State cost
6 - R = Control cost
7 - Q_terminal = Cost at the final step
8 - x_dynamics array stores the information regarding system. 
    x_dynamics[0] = m = mass of the pendulum bob 
    x_dynamics[1] = M = mass of the cart 
    x_dynamics[2] = L = length of the massles|s rod 
    x_dynamics[3] = g = gravity 
    x_dynamics[4] = d = damping in the system
'''
dt = 0.05
N = 600  # Number of time steps in trajectory.
x_dynamics = np.array([0.1, 1, 1, 9.80665, 0]) # m=1, M=5, L=2, g=9.80665, d=1
x0 = np.array([0.0, 0.0, 3.14, 0.0])  # Initial state
x_goal = np.array([0.0, 0.0, 0.0, 0.0])
# Instantenous state cost.
Q = np.eye(5)
Q[0, 0] = 1
Q[1, 1] = 100
Q[2, 2] = 10
Q[3, 3] = 10
Q[4, 4] = 1
# Terminal state cost.
Q_terminal = 100  * np.eye(5)
# Q_terminal[2, 2] = 100
# Q_terminal[3, 3] = 100
# Instantaneous control cost.
R = np.array([[1.0]])


# ### iLQR on Cart Pole



cartpole_prob = CartPole(dt, N, x_dynamics, x0, x_goal, Q, R, Q_terminal)
xs, us, K, k = cartpole_prob.run_IterLinQuadReg()




# State matrix split into individual states. For plotting and analysing purposes.
t = np.arange(N + 1) * dt
x = xs[:, 0] # Position
x_dot = xs[:, 1] # Velocity
theta = np.unwrap(cartpole_prob.deaugment_state(xs)[:, 2])  # Theta, makes for smoother plots.
theta_dot = xs[:, 3] # Angular velocity


# ### Simulate the real system and generate the data
# Cost matrices, initial position and goal position will remain same as the above problem. As it indicates one policy. 
# But still the initial positions and goal positions must be passed explicitly to the function. 
# But you don't need to pass cost matrices 
# (assume penalty on the system is same), this is just used to use to calculate the cost of the trajectory. 
# Correct control action must be passed. Parameter gamma indicates how much of original data you want to keep
# 
# Variance of the Gaussian noise will be taken as input from a Unif(0, var_range) uniform distribution. 
# Inputs: x_initial, x_goal, u, n_rollouts, pattern='Normal', pattern_rand=False, var_range=10, gamma=0.2, percent=20
# 
# Pattern controls how the control sequence will be modified after applying white Guassian noise (zero mean).
# - Normal: based on the correction/mixing parameter gamma generate control (gamma controls how much noise we want).
# - MissingValue: based on the given percentage, set those many values to zero (it is implicitly it uses "Normal" generated control is used). 
# - Shuffle: shuffles the entire "Normal" generated control sequence.
# - TimeDelay: takes the "Normal" generated control and shifts it by 1 index i.e. one unit time delay.
# - Extreme: sets gamma as zeros and generates control based on only noise.
# 
# If 'pattern_rand' is 'True' then we don't need to send the explicitly, it will chose one randomly for every rollout (default is 'False'). If you want to chose specific pattern then send it explicitly. 



x_rollout, u_rollout, local_policy, cost = cartpole_prob.gen_rollouts(x0, x_goal, us, n_rollouts=10, pattern_rand=True, var_range=10, gamma=0.2, percent=20)


# ### Local system dynamics/model estimate
# loca_estimate: function takes the states (arranged in a special format, [x(t), u(t), x(t+1)]), no. of gaussian mixtures and no.of states.



model = Estimated_Dynamics_Prior(init_sequential=False, eigreg=False, warmstart=True, 
                 min_samples_per_cluster=20, max_clusters=50, max_samples=20, strength=1.0)
model.update_prior(x_rollout, u_rollout)
A, B, C = model.fit(x_rollout, u_rollout)



print(A.shape)
print(B.shape)
print(C.shape)



'''
Model = "mujoco/cartpole.xml"
model_loaded = load_model_from_path(Model)
sim = MjSim(model_loaded)




#viewer = mujoco_py.MjViewer(sim)
t = 0
sim.data.qpos[0] = 0.0
sim.data.qpos[1] = 0.14
sim.data.qvel[0] = 0
sim.data.qvel[1] = 0
final = 0
for i in range(2000):
    start_time = time.time()
    state = np.c_[sim.data.qpos[0],sim.data.qvel[0],np.sin(sim.data.qpos[1]),
                  np.cos(sim.data.qpos[1]),sim.data.qvel[1]].T
    control = np.dot(k[i,:],(xs[i].reshape(5,1) - state ))  + K[i].T + us[i]
    sim.data.ctrl[0] = control
    sim.step()
    #viewer.render()
    #print(control, "this is the control")
    if (sim.data.qpos[0] == 1.0 and sim.data.qpos[1] == 0):
        print('states reached')
        break
print(sim.get_state())




import time
time.sleep(5)




from Simulator import Mujoco_sim
Model = "mujoco/cartpole.xml"
cart_pole_simulator = Mujoco_sim(Model,True)
cart_pole_simulator.load(xs,us,k,K,x0,initial=False)
cart_pole_simulator.runSimulation()




cart_pole_simulator.runSimulation()
'''