#!/usr/bin/env python3
import numpy as np
from scipy.stats.mstats import gmean
import theano.tensor as T

from ilqr import iLQR
from ilqr.cost import QRCost
from ilqr.dynamics import constrain
from ilqr.dynamics import AutoDiffDynamics

class CartPole(object):
    '''
    __init__: in this method we store all the variables required for formulating the Cart Pole iLQR problem.
    '''
    def __init__(self, delta_t, traj_steps, state_dynamics, initial_state, goal, cost_state, cost_action, terminal_cost):
        self.dt = delta_t
        self.N = traj_steps
        self.xd = state_dynamics
        self.x0 = initial_state
        self.x_goal = goal
        self.Q = cost_state
        self.R = cost_action
        self.Q_terminal = terminal_cost
        self.J_hist = []

    '''
    state_inputs: in this method we generate the state and control vector. State for the give system are
                    1 - Position == x
                    2 - Linear velocity (of the cart) == x_dot
                    3 - Sine component of the angular position == sin_theta
                    4 - Cosine component of the angular position == cos_theta
                    5 - Angular velocity == theta_dot.
                  We use sine and cosine component of the angular position instead of the angular position because 
                  we can having issue regarding angular wrap around issues, when solving the optimization problem.
                  Control for the system is given by
                    1 - Horizontal force on the cart == F_x
    '''
    def state_inputs(self):
        x_input = [
            T.dscalar("x"),
            T.dscalar("x_dot"),
            T.dscalar("sin_theta"),
            T.dscalar("cos_theta"),
            T.dscalar("theta_dot")
        ]
        u_input = [
            T.dscalar("F_x")
        ]
        return x_input, u_input
    
    '''
    augment_state: in this method we change the state vector which is of the form [pos, vel, angular_pos, angular_vel] 
                    to [pos, vel, sin(angular_pos), cos(angular_pos), angular_vel].
    '''
    def augment_state(self, X):
        if X.ndim == 1:
            x, x_dot, theta, theta_dot = X
        else:
            x = X[..., 0].reshape(-1, 1)
            x_dot = X[..., 1].reshape(-1, 1)
            theta = X[..., 2].reshape(-1, 1)
            theta_dot = X[..., 3].reshape(-1, 1)
        return np.hstack([x, x_dot, np.sin(theta), np.cos(theta), theta_dot])
    
    '''
    deaugment_state: in this method we change the state vector which is of the form 
                        [pos, vel, sin(angular_pos), cos(angular_pos), angular_vel] to [pos, vel, angular_pos, angular_vel].
    '''
    def deaugment_state(self, X):
        if X.ndim == 1:
            x, x_dot, sin_theta, cos_theta, theta_dot = X
        else: 
            x = X[..., 0].reshape(-1, 1)
            x_dot = X[..., 1].reshape(-1, 1)
            sin_theta = X[..., 2].reshape(-1, 1)
            cos_theta = X[..., 3].reshape(-1, 1)
            theta_dot = X[..., 4].reshape(-1, 1)
        theta = np.arctan2(sin_theta, cos_theta)
        return np.hstack([x, x_dot, theta, theta_dot])
    
    '''
    accel: this method is used for generating the linear acceleration, angular acceleration and next angle. All these values 
            are used for calculating the next states. Accelerations will be used to calculate the next time step velocities,
            linear velocitiy will be used to calculate the next time step positions and angular velocity will be used to 
            calculate the next theta. 
            For calculating the accelerations we follow the non-linear system dynamics of the cart-pole problem. 
    Equation are given here:
    https://pdfs.semanticscholar.org/f665/cfa1143e6414cdcb459c5d84796908af4727.pdf?_ga=2.177173158.203278840.1548098392562587694.1548098392
    '''
    def accel(self, X, u, Xd):
        temp = (u[0] + Xd[0] * Xd[2] * X[4]**2 * X[2])/(Xd[1] + Xd[0])
        num = Xd[3] * X[2] - X[3] * temp
        den = Xd[2] * (4/3 - (Xd[0] * X[3]**2)/(Xd[1] + Xd[0]))
        ang_acc = num/den
        lin_acc = temp - (Xd[0] * Xd[2] * ang_acc * X[3])/(Xd[1] + Xd[0])
        theta = T.arctan2(X[2], X[3])
        next_theta = theta + X[4] * self.dt
        return lin_acc, ang_acc, next_theta
    
    '''
    next_states: this method calculates the next time step states based on accelerations as described above. 
    '''
    def next_states(self, X, lin_acc, ang_acc, next_theta):
        f = T.stack([
            X[0] + X[1] * self.dt,
            X[1] + lin_acc * self.dt,
            T.sin(next_theta),
            T.cos(next_theta),
            X[4] + ang_acc * self.dt,
        ])
        return f
    
    '''
    on_iteration: this method will help print useful information that is required so that we can keep track
                    about what is happening.
    '''
    def on_iteration(self, iteration_count, xs, us, J_opt, accepted, converged):
        self.J_hist.append(J_opt)
        info = "converged" if converged else ("accepted" if accepted else "failed")
        final_state = self.deaugment_state(xs[-1])
        print("iteration", iteration_count, info, J_opt, final_state)
    
    '''
    run_IterLinQuadReg: this method is the main function which will run the iLQR on the Cart Pole problem. Steps that are happening
                        1 - Calculate the state vectors (Symbolic vector)
                        2 - Calculate the accelerations (Symbolic vector)
                        3 - Generate the next state (Symbolic vector)
                        4 - Differentiate the states and generate the dynamics
                        5 - Set the goal
                        6 - Set cost based on if the termial cost is provided or not
                        7 - Set the initial state
                        8 - Guess some control action
                        9 - Run iLQR
    '''
    def run_IterLinQuadReg(self, us_init=None):
        x_input, u_input = self.state_inputs()
        x_dot_dot, theta_dot_dot, theta_prime = self.accel(x_input, u_input, self.xd)
        f = self.next_states(x_input, x_dot_dot, theta_dot_dot, theta_prime)
        dynamics = AutoDiffDynamics(f, x_input, u_input,hessians=True)
        x_goal = self.augment_state(self.x_goal)
        if self.Q_terminal.all() == None:
            cost = QRCost(self.Q, self.R)
        else:
            cost = QRCost(self.Q, self.R, Q_terminal=self.Q_terminal, x_goal=x_goal)
        x0 = self.augment_state(self.x0)
        if us_init == None:
            us_init = np.random.uniform(-1, 1, (self.N, dynamics.action_size))
        ilqr = iLQR(dynamics, cost, self.N,hessians=True)
        xs, us = ilqr.fit(x0, us_init, on_iteration=self.on_iteration, tol=1e-12)
        # print(ilqr._K,'this is capital K')
        return xs, us, ilqr._k, ilqr._K
   
    '''
    next_states_matrix: this method will work same as next_state except that it will take A, B state-space matrices as input
    '''
    def next_states_matrix(self, X, U, A, B, C):
        theta = T.arctan2(X[2], X[3])
        next_theta = theta + X[4] * self.dt
        f = T.stack([
            X[0] * A[0][0] + X[1] * A[0][1] + X[2] * A[0][2] + X[3] * A[0][3] + X[4] * A[0][4] + U[0] * A[0][5] + B[0][0],
            X[0] * A[1][0] + X[1] * A[1][1] + X[2] * A[1][2] + X[3] * A[1][3] + X[4] * A[1][4] + U[0] * A[1][5] + B[1][0],
            T.sin(next_theta),
            T.cos(next_theta),
            X[0] * A[4][0] + X[1] * A[4][1] + X[2] * A[4][2] + X[3] * A[4][3] + X[4] * A[4][4] + U[0] * A[4][5] + B[4][0]
        ])
        return f
    
    '''
    run_IterLinQuadReg_matrix: this method will run iLQR when we are given A, B, C state-space matrices
                        X(k+1) = A * [X(k) U(k)].T + B ---- Evolution of state over time is governed by this equation
    '''
    def run_IterLinQuadReg_matrix(self, A, B, C, dist_info_sharing='GM', us_init=None):
        x_input, u_input = self.state_inputs()
        if np.ndim(A) != 2:
            if dist_info_sharing == 'GM':
                A = gmean(A, axis=0)
                B = gmean(B, axis=0)
                C = gmean(C, axis=0)
            elif dist_info_sharing == 'AM':
                A = np.sum(A, axis=0)/A.shape[0]
                B = np.sum(B, axis=0)/B.shape[0]
                C = np.sum(C, axis=0)/C.shape[0]
        else:
            pass
        f = self.next_states_matrix(x_input, u_input, A, B, C)
        dynamics = AutoDiffDynamics(f, x_input, u_input)
        x_goal = self.augment_state(self.x_goal)
        if self.Q_terminal.all() == None:
            cost = QRCost(self.Q, self.R)
        else:
            cost = QRCost(self.Q, self.R, Q_terminal=self.Q_terminal, x_goal=x_goal)
        x0 = self.augment_state(self.x0)
        if us_init == None:
            us_init = np.random.uniform(-1, 1, (self.N, dynamics.action_size))
        ilqr = iLQR(dynamics, cost, self.N)
        xs, us = ilqr.fit(x0, us_init, on_iteration=self.on_iteration)
        return xs, us
        
    '''
    control_pattern: This modify the control actions based on the user input.
                        1 - Normal: based on the correction/mixing parameter gamma generate control 
                                (gamma controls how much noise we want).
                        2 - MissingValue: based on the given percentage, set those many values to zero 
                                (it is implicitly it uses "Normal" generated control is used). 
                        3 - Shuffle: shuffles the entire "Normal" generated control sequence.
                        4 - TimeDelay: takes the "Normal" generated control and shifts it by 1 index i.e. one unit time delay.
                        5 - Extreme: sets gamma as zeros and generates control based on only noise.
    '''
    def control_pattern(self, u, pattern, mean, var, gamma, percent):
        if pattern == 'Normal':
            u = gamma * u + (1 - gamma) * np.random.normal(mean, var, u.shape)
        elif pattern == 'MissingValue':
            n = int(u.shape[0] * percent * 0.01)
            index = np.random.randint(0, u.shape[0], n)
            u = gamma * u + (1 - gamma) * np.random.normal(mean, var, u.shape)
            u[index, :] = 0
        elif pattern == 'Shuffle':
            u = gamma * u + (1 - gamma) * np.random.normal(mean, var, u.shape)
            np.random.shuffle(u)
        elif pattern == 'TimeDelay':
            u = gamma * u + (1 - gamma) * np.random.normal(mean, var, u.shape)
            u = np.roll(u, 1, axis=0)
            u[0, :] = 0
        elif pattern == 'Extreme':
            u = np.random.normal(mean, var, u.shape)
        return u
    
    '''
    noise_traj_generator: In this method we generate trajectories based on some inital condition and noisy control action
                            which is generated from doing noise free ilQR roll-out and then adding noise to it.
                            Noise is added to the control action in special way. I use parameter gamma which indicates how much
                            percentage of original control action sequence must be mixed some percentage of the Gaussian noise.
                            Parameter: 1 - Mean (mean): Gaussian mean and 2 - Variance (var): Gaussian var.
                            Takes deaugmented states as input. Input will be the initial point and perfect control action.
                            Also, based on the control pattern it again remodifies the enitre control action. Look above func.
    '''
    def noise_traj_generator(self, x, u, pattern, mean, var, gamma, percent):
        u = self.control_pattern(u, pattern, mean, var, gamma, percent)
        x_new = self.augment_state(x)
        x_new = x_new.reshape(1,x_new.shape[0])
        for i in range(self.N):
            temp = (u[i][0] + self.xd[0] * self.xd[2] * x_new[-1][4]**2 * x_new[-1][2])/(self.xd[1] + self.xd[0])
            num = self.xd[3] * x_new[-1][2] - x_new[-1][3] * temp
            den = self.xd[2] * (4/3 - (self.xd[0] * x_new[-1][3]**2)/(self.xd[1] + self.xd[0]))
            ang_acc = num/den
            lin_acc = temp - (self.xd[0] * self.xd[2] * ang_acc * x_new[-1][3])/(self.xd[1] + self.xd[0])
            theta = np.arctan2(x_new[-1][2], x_new[-1][3])
            next_theta = theta + x_new[-1][4] * self.dt
            temp_1 = x_new[-1][0] + x_new[-1][1] * self.dt
            temp_2 = x_new[-1][1] + lin_acc * self.dt
            temp_3 = np.sin(next_theta)
            temp_4 = np.cos(next_theta)
            temp_5 = x_new[-1][4] + ang_acc * self.dt
            x_new = np.concatenate((x_new, [[temp_1, temp_2, temp_3, temp_4, temp_5]]), axis=0)
        return x_new, u
    
    '''
    eval_traj_cost: This method calculates the trajectory cost based on Q, R and Q_terminal cost matrices.
                        Takes the deaugmented states as input. Entire trajectory states including goal and control actions
                        are sent as inputs.
    '''
    def eval_traj_cost(self, x, y, u):
        x = self.augment_state(x)
        y = self.augment_state(y)
        J = 0
        if self.Q_terminal.all() == None:
            for i in range(self.N+1):
                J = J + np.matmul((x[i,:]-y), np.matmul(self.Q, (x[i,:]-y).T)) + np.matmul((u[i]-u[-1]).T, np.matmul(self.R, u[i]-u[-1]))
            J = J + np.matmul((x[-1]-y), np.matmul(self.Q, (x[-1]-y).T))
        else:
            for i in range(self.N):
                J = J + np.matmul((x[i,:]-y), np.matmul(self.Q, (x[i,:]-y).T)) + np.matmul((u[i]-u[-1]).T, np.matmul(self.R, u[i]-u[-1]))
            J = J + np.matmul((x[-1]-y), np.matmul(self.Q_terminal, (x[-1]-y).T))
        return J
    
    '''
    gen_rollouts: Generates specified no. of rollouts and outputs states, control action and cost of all rollouts
                    Variance of the Gaussian noise will be taken as input from a Unif(0, var_range) uniform distribution.
    '''
    def gen_rollouts(self, x_initial, x_goal, u, n_rollouts, pattern='Normal', pattern_rand=False, var_range=10, gamma=0.2, percent=20):
        x_rollout = []
        u_rollout = []
        x_gmm = []
        cost = []
        local_policy = self.control_pattern(u, 'Normal', 0, np.random.uniform(0, 5, 1), 0.2, 20)
        for i in range(n_rollouts):
            if pattern_rand == True:
                pattern_seq = np.array(['Normal', 'MissingValue', 'Shuffle', 'TimeDelay', 'Extreme'])
                pattern = pattern_seq[np.random.randint(0, 5, 1)][0]
            x_new, u_new = self.noise_traj_generator(x_initial, local_policy, pattern, 0, np.random.uniform(0, var_range, 1), gamma, percent)
            x_new_temp = self.deaugment_state(x_new)
            cost.append(self.eval_traj_cost(x_new_temp, x_goal, u_new))
            x_rollout.append(x_new)
            u_new = np.append(u_new, [[0]], axis=0)
            u_rollout.append(u_new)
#             temp = np.append(x_new[:-1,:], u_new, axis=1)
#             temp = np.append(temp, x_new[1:,:], axis=1)
#             x_gmm.append(temp)
#         x_rollout = np.array(x_rollout).reshape(len(x_rollout)*len(x_rollout[0]), len(x_rollout[0][0,:]))
#         u_rollout = np.array(u_rollout).reshape(len(u_rollout)*len(u_rollout[0]), len(u_rollout[0][0,:]))
        x_rollout = np.array(x_rollout)
        u_rollout = np.array(u_rollout)
#         x_gmm = np.array(x_gmm).reshape(len(x_gmm)*len(x_gmm[0]), len(x_gmm[0][0,:]))
        cost = np.array(cost).reshape(n_rollouts, 1)
        return x_rollout, u_rollout, local_policy, cost
#         return x_rollout, u_rollout, local_policy, x_gmm, cost