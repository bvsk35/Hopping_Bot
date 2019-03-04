#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn import mixture
import matplotlib.pyplot as plt
import theano.tensor as T

from ilqr import iLQR
from ilqr.cost import QRCost
from ilqr.dynamics import constrain
from ilqr.dynamics import AutoDiffDynamics





def accel(X, u, m=1, M=5, L=2, g=9.80665, d=1):
    temp = (u[0] + m * L * X[4]**2 * X[2])/(M + m)
    num = g * X[2] - X[3] * temp
    den = L * (4/3 - (m * X[3]**2)/(M + m))
    ang_acc = num/den
    lin_acc = temp - (m * L * ang_acc * X[3])/(M + m)
    theta = T.arctan2(X[2], X[3])
    next_theta = theta + X[4] * dt
    return lin_acc, ang_acc, next_theta

def augment_state(X):
    if X.ndim == 1:
        x, x_dot, theta, theta_dot = X
    else:
        x = X[..., 0].reshape(-1, 1)
        x_dot = X[..., 1].reshape(-1, 1)
        theta = X[..., 2].reshape(-1, 1)
        theta_dot = X[..., 3].reshape(-1, 1)
    return np.hstack([x, x_dot, np.sin(theta), np.cos(theta), theta_dot])

def deaugment_state(X):
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



# Define required functions
def on_iteration(iteration_count, xs, us, J_opt, accepted, converged):
    J_hist.append(J_opt)
    info = "converged" if converged else ("accepted" if accepted else "failed")
    final_state = deaugment_state(xs[-1])
    print("iteration", iteration_count, info, J_opt, final_state)





class local_estimate:
    #well here instead of the local policy I am taking in data
    #data dimension is 10X4
    def __init__(self,data,components):
        self.data = data
        self.components = components
        self.gmm_data = None
        self.means = None
        self.covariances = None
        self.NIW_id = 0
        self.append_data = 0
        if self.means != None:
            self.append_data = np.append(data,self.data)
                                                

    def estimate(self,gmm_true=True,N=500):
        self.gmm_true = gmm_true
        if self.gmm_true:
            x,y = self.GMM_method()
            print("shape of the data in the gmm",self.gmm_data.shape)
            x,y = self.NIW_distribution(x,y,N)
            A,B,C = self.Post_predict(x,y)
        else:
            return self.old()

    def GMM_method(self):
        if self.gmm_data == None:
            self.gmm_data = self.data
            n,m = self.data.shape 
            print("before",self.gmm_data.shape)
            self.gmm_data = np.append(self.gmm_data,self.data,axis=0)
            print("before",self.gmm_data.shape)
        else:
            self.data = self.data.reshape(1,n,m)
            self.gmm_data = np.append(self.gmm_data,self.data)
        gmm = mixture.GaussianMixture(n_components = self.components , covariance_type = 'full')
        gmm.fit(self.gmm_data)
        self.means = gmm.means_
        self.covariance = gmm.covariances_ 
        p, r = gmm._estimate_log_prob_resp(self.gmm_data)
        r = np.exp(r)
        mix = np.sum(r,axis=0) / len(r)
        #print("relevance size ",r.shape,"gmm_data",self.gmm_data.shape,"mix",mix.shape,'mean',self.means)
        for i in range(self.components):
            r[:,i] = r[:,i]*mix[i]
        sum_relevance = np.dot(r.T,self.gmm_data)
        sum_relevance = np.sum(sum_relevance,axis = 1)

        self.NIW_id = np.where(sum_relevance==max(sum_relevance))
        #print("NIM_id",self.NIW_id)
        #print("print"+str(self.means[:,self.NIW_id].shape))
        #print(" size ",self.means.shape,"cov",self.covariance.shape)
        return (self.means[self.NIW_id,:],self.covariance[self.NIW_id,:,:])
        #here the r is the relevance of the size (1000,n_gaussian)
    
    def NIW_distribution(self,x,y,N):
        gmm_mean = x
        gmm_cov = y.reshape(9,9)
        gmm_mean = gmm_mean.reshape(9,1)
        print("mean",gmm_mean.shape,"cov",gmm_cov.shape)
        NIW_mean = (N*np.mean(self.gmm_data,axis=0).reshape(9,1) + gmm_mean)/(N+1)
        # print(np.mean(self.gmm_data,axis=0).shape)
        cov0 = 0
        print(np.shape(self.gmm_data))
        cov0 = np.dot(self.gmm_data.T,self.gmm_data)
        cov1 = np.dot((NIW_mean - gmm_mean),(NIW_mean - gmm_mean).T)
        print("shape of 1st term",gmm_cov.shape,"shape of 2st term",cov0.shape,"shape of 3rd term",cov1.shape)
        NIW_cov = gmm_cov + N*cov1/(N+1) + cov0
        print('NIW mean',NIW_mean.shape)
        return(NIW_mean,NIW_cov)


    def old(self):
        self.means = np.mean(self.data,0)
        for x in self.data.T:
            x = x.T
            self.covariances += np.dot((x-self.means)*(x-self.means).T)
        self.covariances = self.covariances / (len(self.data) -1)

    def Post_predict(self,x,y):
        cov_reg = 0.01 * np.eye(5)
        mean_a  = x[5:]
        mean_b = x[:5]
        cov_aa = y[5:,5:]
        cov_ab = y[5:,0:5]
        cov_ba = y[0:5,5:]
        cov_bb = y[0:5,0:5]
        A = np.matmul(cov_ab,np.linalg.inv(cov_reg + cov_bb))
        B = mean_a - A.dot(mean_b)
        C = cov_aa - cov_ab.dot(np.linalg.inv(cov_reg + cov_bb)).dot(cov_ba)
        return (A,B,C)

if __name__ == "__main__":
    dt = 0.005
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
    x_dot_dot, theta_dot_dot, theta_prime = accel(x_input, u_input)

    f = T.stack([
    x_input[0] + x_input[1] * dt,
    x_input[1] + x_dot_dot * dt,
    T.sin(theta_prime),
    T.cos(theta_prime),
    x_input[4] + theta_dot_dot * dt,
    ])
    dynamics = AutoDiffDynamics(f, x_input, u_input)
    
    # Note that the augmented state is not all 0.
    x_goal = augment_state(np.array([2.0, 0.0, 0.0, 0.0]))
    
    # Instantenous state cost.
    Q = np.eye(dynamics.state_size)
    Q[2, 2] = 10
    Q[3, 3] = 100
    
    # Terminal state cost.
    Q_terminal = 100 * np.eye(dynamics.state_size)
    
    # Instantaneous control cost.
    R = np.array([[1.0]])
    
    cost = QRCost(Q, R, Q_terminal=Q_terminal, x_goal=x_goal)
    
    N = 500  # Number of time steps in trajectory.
    x0 = augment_state(np.array([-3.0, 0.0, 0.1, 0.0]))  # Initial state
    
    # Random initial action path.
    us_init = np.random.uniform(-1, 1, (N, dynamics.action_size))
    
    J_hist = []
    ilqr = iLQR(dynamics, cost, N)
    xs, us = ilqr.fit(x0, us_init, on_iteration=on_iteration)
    us = us + np.random.normal(0,1,us.shape)
    t = np.arange(N + 1) * dt
    xs = deaugment_state(xs)
    x = xs[:, 0]
    x_dot = xs[:, 1]
    theta = np.unwrap(xs[:, 2])  # Makes for smoother plots.
    theta_dot = xs[:, 3]
    
    us = constrain(us, -1, 1)
    #appending control data with state data
    actual_data = np.append(xs[:-1,:],us,axis=1)
    #appending the previous data with the next state data
    actual_data = np.append(actual_data,xs[1:,:],axis=1)
    print(actual_data.shape)
    estimate = local_estimate(actual_data,5)
    estimate.estimate()
    print(estimate.means.shape)
    #print(mean.shape)
    
   


