
import numpy as np
from agent_mujoco import AgentMujoco
from hyperparameters import Agent_params
from init_gauss_lqr import init_lqr
from hyperparameters import algorithm
from hyperparameters import Cost


class LQR(object):
    def __init__(self, init_policy = None):
        if init_policy == None:
            self.init_policy = init_lqr(algorithm['init_traj_distr'])
        self.init_policy = init_policy
        #  To setup the agent by giving it the setup condition and passing hyperparameters
        self.agent = AgentMujoco(Agent_params)
        self.cost = Cost['cost'](Cost['params'])
        self.M = 
        self.T = 
        self.dX = 
        self.dU = 

    def forward(self):
        '''I need to apply the control action using the initial policy '''
        pass
        




    def backward(self):
        pass

    def _eval_cost(self, cond):
        """
        Evaluate costs for all samples for a condition.
        Args:
            cond: Condition to evaluate cost on.
        """
        # here the self.cur[0].sample_list is the samplelist object which has the data
        # Constants.
        T, dX, dU = self.T, self.dX, self.dU
        N = len(self.cur[cond].sample_list)
        # print(self.cur[cond].sample_list,'testing what is the cost function')
        # Compute cost.
        cs = np.zeros((N, T))
        cc = np.zeros((N, T))
        cv = np.zeros((N, T, dX+dU))
        Cm = np.zeros((N, T, dX+dU, dX+dU))
        for n in range(N):
            sample = self.cur[cond].sample_list[n]
            # Get costs.
            # Here the sample is the indiviual data of each attempt
            l, lx, lu, lxx, luu, lux = self.cost[cond].eval(sample)
            # this function goes to cost_sum.py file in the cost function is divided into two files
            # action cost and sate cost
            cc[n, :] = l
            cs[n, :] = l

            # Assemble matrix and vector.
            cv[n, :, :] = np.c_[lx, lu]
            Cm[n, :, :, :] = np.concatenate(
                (np.c_[lxx, np.transpose(lux, [0, 2, 1])], np.c_[lux, luu]),
                axis=1
            )

            # Adjust for expanding cost around a sample.
            X = sample.get_X()
            U = sample.get_U()
            yhat = np.c_[X, U]
            rdiff = - yhat
            rdiff_expand = np.expand_dims(rdiff, axis=2)
            cv_update = np.sum(Cm[n, :, :, :] * rdiff_expand, axis=1)
            cc[n, :] += np.sum(rdiff * cv[n, :, :], axis=1) + 0.5 * \
                    np.sum(rdiff * cv_update, axis=1)
            cv[n, :, :] += cv_update

        # Fill in cost estimate.
        self.cur[cond].traj_info.cc = np.mean(cc, 0)  # Constant term (scalar).
        self.cur[cond].traj_info.cv = np.mean(cv, 0)  # Linear term (vector).
        self.cur[cond].traj_info.Cm = np.mean(Cm, 0)  # Quadratic term (matrix).

        self.cur[cond].cs = cs  # True value of cost.