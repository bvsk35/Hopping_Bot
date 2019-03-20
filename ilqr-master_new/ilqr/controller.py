# Copyright (C) 2018, Anass Al
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
"""Controllers."""

import six
import abc
import warnings
import numpy as np
from scipy import optimize
from scipy.stats import entropy

@six.add_metaclass(abc.ABCMeta)
class BaseController():

    """Base trajectory optimizer controller."""

    @abc.abstractmethod
    def fit(self, x0, us_init, *args, **kwargs):
        """Computes the optimal controls.

        Args:
            x0: Initial state [state_size].
            us_init: Initial control path [N, action_size].
            *args, **kwargs: Additional positional and key-word arguments.

        Returns:
            Tuple of
                xs: optimal state path [N+1, state_size].
                us: optimal control path [N, action_size].
        """
        raise NotImplementedError


class iLQR(BaseController):

    """Finite Horizon Iterative Linear Quadratic Regulator."""

    def __init__(self, dynamics, cost, N, max_reg=1e10, hessians=False):
        """Constructs an iLQR solver.

        Args:
            dynamics: Plant dynamics.
            cost: Cost function.
            N: Horizon length.
            max_reg: Maximum regularization term to break early due to
                divergence. This can be disabled by setting it to None.
            hessians: Use the dynamic model's second order derivatives.
                Default: only use first order derivatives. (i.e. iLQR instead
                of DDP).
        """
        self.dynamics = dynamics
        self.cost = cost
        self.N = N
        self._use_hessians = hessians and dynamics.has_hessians
        if hessians and not dynamics.has_hessians:
            warnings.warn("hessians requested but are unavailable in dynamics")

        # Regularization terms: Levenberg-Marquardt parameter.
        # See II F. Regularization Schedule.
        self._mu = 1.0
        self._mu_min = 1e-6
        self._mu_max = max_reg
        self._delta_0 = 2.0
        self._delta = self._delta_0

        self._k = np.zeros((N, dynamics.action_size))
        self._K = np.zeros((N, dynamics.action_size, dynamics.state_size))
        super(iLQR, self).__init__()

    def fit(self, x0, us_init, n_iterations=100, tol=1e-6, on_iteration=None):
        """Computes the optimal controls.
        Args:
            x0: Initial state [state_size].
            us_init: Initial control path [N, action_size].
            n_iterations: Maximum number of interations. Default: 100.
            tol: Tolerance. Default: 1e-6.
            on_iteration: Callback at the end of each iteration with the
                following signature:
                (iteration_count, x, J_opt, accepted, converged) -> None
                where:
                    iteration_count: Current iteration count.
                    xs: Current state path.
                    us: Current action path.
                    J_opt: Optimal cost-to-go.
                    accepted: Whether this iteration yielded an accepted result.
                    converged: Whether this iteration converged successfully.
                Default: None.

        Returns:
            Tuple of
                xs: optimal state path [N+1, state_size].
                us: optimal control path [N, action_size].
        """
        # Reset regularization term.
        self._mu = 1.0
        self._delta = self._delta_0

        # Backtracking line search candidates 0 < alpha <= 1.
        alphas = 1.1**(-np.arange(10)**2)

        us = us_init.copy()

        k = self._k
        K = self._K

        changed = True
        converged = False
        for iteration in range(n_iterations):
            accepted = False

            # Forward rollout only if it needs to be recomputed.
            if changed:
                (xs, F_x, F_u, L, L_x, L_u, L_xx, L_ux, L_uu, F_xx, F_ux,
                F_uu) = self._forward_rollout(x0, us, us_init)
                J_opt = L.sum()
                changed = False

            try:
                # Backward pass.
                k, K = self._backward_pass(F_x, F_u, L_x, L_u, L_xx, L_ux, L_uu,
                                        F_xx, F_ux, F_uu)
                # Backtracking line search.
                for alpha in alphas:
                    xs_new, us_new = self._control(xs, us, k, K, alpha)
                    J_new = self._trajectory_cost(xs_new, us_new)
                    if J_new < J_opt:
                        if np.abs((J_opt - J_new) / J_opt) < tol:
                            converged = True
                        J_opt = J_new
                        xs = xs_new
                        us = us_new
                        changed = True

                        # Decrease regularization term.
                        self._delta = min(1.0, self._delta) / self._delta_0
                        self._mu *= self._delta
                        if self._mu <= self._mu_min:
                            self._mu = 0.0

                        # Accept this.
                        accepted = True
                        break
            except np.linalg.LinAlgError as e:
                # Quu was not positive-definite and this diverged.
                # Try again with a higher regularization term.
                warnings.warn(str(e))

            if not accepted:
                # Increase regularization term.
                self._delta = max(1.0, self._delta) * self._delta_0
                self._mu = max(self._mu_min, self._mu * self._delta)
                if self._mu_max and self._mu >= self._mu_max:
                    warnings.warn("exceeded max regularization term")
                    break

            if on_iteration:
                on_iteration(iteration, xs, us, J_opt, accepted, converged)

            if converged:
                break

        # Store fit parameters.
        self._k = k
        self._K = K
        self._nominal_xs = xs
        self._nominal_us = us

        return xs, us

    def _control(self, xs, us, k, K, alpha=1.0):
        """Applies the controls for a given trajectory.

        Args:
            xs: Nominal state path [N+1, state_size].
            us: Nominal control path [N, action_size].
            k: Feedforward gains [N, action_size].
            K: Feedback gains [N, action_size, state_size].
            alpha: Line search coefficient.

        Returns:
            Tuple of
                xs: state path [N+1, state_size].
                us: control path [N, action_size].
        """
        xs_new = np.zeros_like(xs)
        us_new = np.zeros_like(us)
        xs_new[0] = xs[0].copy()

        for i in range(self.N):
            # Eq (12).
            # Applying alpha only on k[i] as in the paper for some reason
            # doesn't converge.
            us_new[i] = us[i] + alpha * (k[i] + K[i].dot(xs_new[i] - xs[i]))

            # Eq (8c).
            xs_new[i + 1] = self.dynamics.f(xs_new[i], us_new[i], i)

        return xs_new, us_new

    def _trajectory_cost(self, xs, us):
        """Computes the given trajectory's cost.

        Args:
            xs: State path [N+1, state_size].
            us: Control path [N, action_size].

        Returns:
            Trajectory's total cost.
        """
        J = map(lambda args: self.cost.l(*args), zip(xs[:-1], us, range(self.N)))
        return sum(J) + self.cost.l(xs[-1], None, self.N, terminal=True)

    def _forward_rollout(self, x0, us, local_policy):
        """Apply the forward dynamics to have a trajectory from the starting
        state x0 by applying the control path us.

        Args:
            x0: Initial state [state_size].
            us: Control path [N, action_size].

        Returns:
            Tuple of:
                xs: State path [N+1, state_size].
                F_x: Jacobian of state path w.r.t. x
                    [N, state_size, state_size].
                F_u: Jacobian of state path w.r.t. u
                    [N, state_size, action_size].
                L: Cost path [N+1].
                L_x: Jacobian of cost path w.r.t. x [N+1, state_size].
                L_u: Jacobian of cost path w.r.t. u [N, action_size].
                L_xx: Hessian of cost path w.r.t. x, x
                    [N+1, state_size, state_size].
                L_ux: Hessian of cost path w.r.t. u, x
                    [N, action_size, state_size].
                L_uu: Hessian of cost path w.r.t. u, u
                    [N, action_size, action_size].
                F_xx: Hessian of state path w.r.t. x, x if Hessians are used
                    [N, state_size, state_size, state_size].
                F_ux: Hessian of state path w.r.t. u, x if Hessians are used
                    [N, state_size, action_size, state_size].
                F_uu: Hessian of state path w.r.t. u, u if Hessians are used
                    [N, state_size, action_size, action_size].
        """
        state_size = self.dynamics.state_size
        action_size = self.dynamics.action_size
        N = us.shape[0]

        xs = np.empty((N + 1, state_size))
        F_x = np.empty((N, state_size, state_size))
        F_u = np.empty((N, state_size, action_size))

        if self._use_hessians:
            F_xx = np.empty((N, state_size, state_size, state_size))
            F_ux = np.empty((N, state_size, action_size, state_size))
            F_uu = np.empty((N, state_size, action_size, action_size))
        else:
            F_xx = None
            F_ux = None
            F_uu = None

        L = np.empty(N + 1)
        L_x = np.empty((N + 1, state_size))
        L_u = np.empty((N, action_size))
        L_xx = np.empty((N + 1, state_size, state_size))
        L_ux = np.empty((N, action_size, state_size))
        L_uu = np.empty((N, action_size, action_size))

        xs[0] = x0
        for i in range(N):
            x = xs[i]
            u = us[i]

            xs[i + 1] = self.dynamics.f(x, u, i)
            F_x[i] = self.dynamics.f_x(x, u, i)
            F_u[i] = self.dynamics.f_u(x, u, i)

            L[i] = self.cost.l(x, u, i, terminal=False)
            L_x[i] = self.cost.l_x(x, u, i, terminal=False)
            L_u[i] = self.cost.l_u(x, u, i, terminal=False)
            L_xx[i] = self.cost.l_xx(x, u, i, terminal=False)
            L_ux[i] = self.cost.l_ux(x, u, i, terminal=False)
            L_uu[i] = self.cost.l_uu(x, u, i, terminal=False)

            if self._use_hessians:
                F_xx[i] = self.dynamics.f_xx(x, u, i)
                F_ux[i] = self.dynamics.f_ux(x, u, i)
                F_uu[i] = self.dynamics.f_uu(x, u, i)

        x = xs[-1]
        L[-1] = self.cost.l(x, None, N, terminal=True)
        L_x[-1] = self.cost.l_x(x, None, N, terminal=True)
        L_xx[-1] = self.cost.l_xx(x, None, N, terminal=True)

        return xs, F_x, F_u, L, L_x, L_u, L_xx, L_ux, L_uu, F_xx, F_ux, F_uu

    def _backward_pass(self,
                       F_x,
                       F_u,
                       L_x,
                       L_u,
                       L_xx,
                       L_ux,
                       L_uu,
                       F_xx=None,
                       F_ux=None,
                       F_uu=None):
        """Computes the feedforward and feedback gains k and K.

        Args:
            F_x: Jacobian of state path w.r.t. x [N, state_size, state_size].
            F_u: Jacobian of state path w.r.t. u [N, state_size, action_size].
            L_x: Jacobian of cost path w.r.t. x [N+1, state_size].
            L_u: Jacobian of cost path w.r.t. u [N, action_size].
            L_xx: Hessian of cost path w.r.t. x, x
                [N+1, state_size, state_size].
            L_ux: Hessian of cost path w.r.t. u, x [N, action_size, state_size].
            L_uu: Hessian of cost path w.r.t. u, u
                [N, action_size, action_size].
            F_xx: Hessian of state path w.r.t. x, x if Hessians are used
                [N, state_size, state_size, state_size].
            F_ux: Hessian of state path w.r.t. u, x if Hessians are used
                [N, state_size, action_size, state_size].
            F_uu: Hessian of state path w.r.t. u, u if Hessians are used
                [N, state_size, action_size, action_size].

        Returns:
            Tuple of
                k: feedforward gains [N, action_size].
                K: feedback gains [N, action_size, state_size].
        """
        V_x = L_x[-1]
        V_xx = L_xx[-1]

        k = np.empty_like(self._k)
        K = np.empty_like(self._K)

        for i in range(self.N - 1, -1, -1):
            if self._use_hessians:
                Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(
                    F_x[i], F_u[i], L_x[i], L_u[i], L_xx[i], L_ux[i], L_uu[i],
                    V_x, V_xx, F_xx[i], F_ux[i], F_uu[i])
            else:
                Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(F_x[i], F_u[i], L_x[i],
                                                     L_u[i], L_xx[i], L_ux[i],
                                                     L_uu[i], V_x, V_xx)

            # Eq (6).
            k[i] = -np.linalg.solve(Q_uu, Q_u)
            K[i] = -np.linalg.solve(Q_uu, Q_ux)

            # Eq (11b).
            V_x = Q_x + K[i].T.dot(Q_uu).dot(k[i])
            V_x += K[i].T.dot(Q_u) + Q_ux.T.dot(k[i])

            # Eq (11c).
            V_xx = Q_xx + K[i].T.dot(Q_uu).dot(K[i])
            V_xx += K[i].T.dot(Q_ux) + Q_ux.T.dot(K[i])
            V_xx = 0.5 * (V_xx + V_xx.T)  # To maintain symmetry.

        return np.array(k), np.array(K)

    def _Q(self,
           f_x,
           f_u,
           l_x,
           l_u,
           l_xx,
           l_ux,
           l_uu,
           V_x,
           V_xx,
           f_xx=None,
           f_ux=None,
           f_uu=None):
        """Computes second order expansion.

        Args:
            F_x: Jacobian of state w.r.t. x [state_size, state_size].
            F_u: Jacobian of state w.r.t. u [state_size, action_size].
            L_x: Jacobian of cost w.r.t. x [state_size].
            L_u: Jacobian of cost w.r.t. u [action_size].
            L_xx: Hessian of cost w.r.t. x, x [state_size, state_size].
            L_ux: Hessian of cost w.r.t. u, x [action_size, state_size].
            L_uu: Hessian of cost w.r.t. u, u [action_size, action_size].
            V_x: Jacobian of the value function at the next time step
                [state_size].
            V_xx: Hessian of the value function at the next time step w.r.t.
                x, x [state_size, state_size].
            F_xx: Hessian of state w.r.t. x, x if Hessians are used
                [state_size, state_size, state_size].
            F_ux: Hessian of state w.r.t. u, x if Hessians are used
                [state_size, action_size, state_size].
            F_uu: Hessian of state w.r.t. u, u if Hessians are used
                [state_size, action_size, action_size].

        Returns:
            Tuple of
                Q_x: [state_size].
                Q_u: [action_size].
                Q_xx: [state_size, state_size].
                Q_ux: [action_size, state_size].
                Q_uu: [action_size, action_size].
        """
        # Eqs (5a), (5b) and (5c).
        Q_x = l_x + f_x.T.dot(V_x)
        Q_u = l_u + f_u.T.dot(V_x)
        Q_xx = l_xx + f_x.T.dot(V_xx).dot(f_x)

        # Eqs (11b) and (11c).
        reg = self._mu * np.eye(self.dynamics.state_size)
        Q_ux = l_ux + f_u.T.dot(V_xx + reg).dot(f_x)
        Q_uu = l_uu + f_u.T.dot(V_xx + reg).dot(f_u)

        if self._use_hessians:
            Q_xx += np.tensordot(V_x, f_xx, axes=1)
            Q_ux += np.tensordot(V_x, f_ux, axes=1)
            Q_uu += np.tensordot(V_x, f_uu, axes=1)

        return Q_x, Q_u, Q_xx, Q_ux, Q_uu

'''
iLQR_GPS: iLQR modified for GPS
'''
class iLQR_GPS(BaseController):
    """Finite Horizon Iterative Linear Quadratic Regulator."""

    def __init__(self, dynamics, cost_GPS, N, A, B, C, max_reg=1e10, hessians=False, epsilon=1):
        """Constructs an iLQR solver.

        Args:
            dynamics: Plant dynamics.
            cost: Cost function.
            N: Horizon length.
            max_reg: Maximum regularization term to break early due to
                divergence. This can be disabled by setting it to None.
            hessians: Use the dynamic model's second order derivatives.
                Default: only use first order derivatives. (i.e. iLQR instead
                of DDP).
        """
        self.dynamics = dynamics
        self.cost_GPS = cost_GPS
        self.N = N
        self._use_hessians = hessians and dynamics.has_hessians
        if hessians and not dynamics.has_hessians:
            warnings.warn("hessians requested but are unavailable in dynamics")

        # Regularization terms: Levenberg-Marquardt parameter.
        # See II F. Regularization Schedule.
        self._mu = 1.0
        self._mu_min = 1e-6
        self._mu_max = max_reg
        self._delta_0 = 2.0
        self._delta = self._delta_0

        self._k = np.random.uniform(-0.1, 0.1, (N, dynamics.action_size))
        self._K = 0.01 * np.random.normal(0, np.eye(dynamics.action_size, dynamics.state_size), 
                                            (N, dynamics.action_size, dynamics.state_size))

        cov_u = []            
        temp = 0.01 * np.eye(dynamics.action_size)
        cov_u.append(temp)
        self.cov_u = np.array(cov_u*self.N)
        
        ### New params
        self.epsilon = epsilon
        self.A = A
        self.B = B
        self.C = C

        super(iLQR_GPS, self).__init__()

    def generate_mean_cov(self, x, u, k, K, A, B, C, mean_old, cov_old, Q_uu):
        ### EQUATION 2.54, 2.55
        mean_xu = []
        cov_xu = []
        mean = [mean_old[0]]
        cov = [cov_old[0]]
        for i in range(mean_old.shape[0]-1):
            temp = u[i][:, np.newaxis] + k[i][:, np.newaxis] + K[i].dot(mean[-1] - x[i][:, np.newaxis])
            temp1 = np.concatenate((mean[-1], temp), axis=0)
            mean_new = np.matmul(A, temp1) + B
            temp2 = np.matmul(cov[-1], K[i].T)
            temp3 = np.linalg.inv(Q_uu[i]) + np.matmul(K[i], temp2)
            temp4 = np.matmul(np.matmul(A, np.block([[cov[-1], temp2], [temp2.T, temp3]])), A.T)
            cov_new = temp4 + C
            mean.append(mean_new)
            cov.append(cov_new)
            mean_xu.append(temp1)
            cov_xu.append(np.block([[cov_old[i], temp2], [temp2.T, temp3]]))
        return np.array(mean_xu), np.array(cov_xu), np.array(mean), np.array(cov)
    
    def cost_estimation(self, eta, mean_xu, cov_xu, us, us_init): self, x, x_old, u_old, k, K, cov_u,
        ### EQUATION 2.51, 2.57
        self.cost_GPS.eta = eta
        J_estimate_1 = self._trajectory_cost_estimate(mean_xu, us) 
        J_estimate_2 = 0
        for i in range(mean_xu.shape[0]):
            mean_u = u_old + k + np.matmul(K, (x[:self.state_size, :] - x_old)) 
            temp = x[self.state_size:, :] - mean_u
            log_term1 = 0.5 * (np.matmul(np.matmul(temp.T, np.linalg.inv(cov_u)), temp))
            log_term2 = np.log(np.sqrt((2 * np.pi)**self.action_size * np.linalg.det(cov_u)))

        J_estimate_3
        for i in range(mean_xu.shape[0]):
            temp =  np.trace(np.matmul(self.cost_GPS.Q, cov_xu[i]))
            J_estimate_3 = J_estimate_3 + temp
        J_estimate_4 = entropy(np.abs(us_init)) + self.epsilon
        J_estimate = J_estimate_1/eta + J_estimate_2/eta - J_estimate_4[0]
        return J_estimate

    def eta_estimation(self, mean_xu, cov_xu, us, us_init): 
        ### Page 54, 55 eta = [0.001, 10]
        eta_max = self.cost_estimation(0.001, mean_xu, cov_xu, us, us_init)
        eta_min = self.cost_estimation(10, mean_xu, cov_xu, us, us_init)
        if eta_max*eta_min < 0:
            print('Doing Brentq')
            eta = optimize.brentq(self.cost_estimation, 0.001, 10, args=(mean_xu, cov_xu, us, us_init))
            print('New eta ',eta)
        else:
            print('Doing Log search')
            param_range = np.geomspace(0.001, 10, 30)
            loss = []
            for i in param_range:
                temp = self.cost_estimation(i, mean_xu, cov_xu, us, us_init)
                loss.append(temp)
            opt_index = loss.index(min(loss))
            eta = param_range[opt_index]
            print('New eta ',eta)
        return eta
        
    def _control_GPS(self, xs, us, k, K, alpha=1.0):
        """Applies the controls for a given trajectory.

        Args:
            xs: Nominal state path [N+1, state_size + action_size].
            us: Nominal control path [N, action_size].
            k: Feedforward gains [N, action_size].
            K: Feedback gains [N, action_size, state_size].
            alpha: Line search coefficient.

        Returns:
            Tuple of
                xs: state path [N+1, state_size + action_size].
                us: control path [N, action_size].
        """
        xs_new = np.zeros_like(xs)
        us_new = np.zeros_like(us)
        xs_new[0] = xs[0].copy()
        state_size = self.dynamics.state_size
        action_size = self.dynamics.action_size


        for i in range(self.N):
            # Applying alpha only on k[i] as in the paper for some reason
            # doesn't converge.
            us_new[i] = us[i] + alpha * (k[i]) + K[i].dot(xs_new[i][:state_size] - xs[i][:state_size])

            xs_new[i + 1][:state_size] = self.dynamics.f(xs_new[i][:state_size], us_new[i], i)
            xs_new[state_size:] = us_new[i]
        return xs_new, us_new

    def _trajectory_cost_GPS(self, xs, us_local):
        """Computes the given trajectory's cost.

        Args:
            xs: State path [N+1, state_size + action_size].
            u_old: Old control path [N, action_size].

        Returns:
            Trajectory's total cost.
        """
        J = map(lambda args: self.cost_GPS.l(*args), zip(xs[:-1], us_local, range(self.N)))
        return sum(J) + self.cost_GPS.l(xs[-1], None, self.N, terminal=True)

    def _trajectory_cost_estimate(self, xs, us_local):
        """Computes the given trajectory's cost.

        Args:
            xs: State path [N+1, state_size + action_size].
            u_old: Old control path [N, action_size].

        Returns:
            Trajectory's total cost.
        """
        J = map(lambda args: self.cost_GPS.l_deaug(*args), zip(xs[:-1], us_local, range(self.N)))
        return sum(J) + self.cost_GPS.l_deaug(xs[-1], us_local, self.N, terminal=True)

    def _forward_rollout_GPS(self, x0, x_old, us_current, us_old, k, K, cov_u):
        """Apply the forward dynamics to have a trajectory from the starting
        state x0 by applying the control path us.

        Args:
            x0: Initial state [state_size + action_size].
            us: Control path [N, action_size].

        Returns:
            Tuple of:
                xs: State path [N+1, state_size+action_size].
                F_linear: Jacobain of state path w.r.t. x and u combined [state_size, state_size + action_size].
                F_quad: Hessian of state path w.r.t. x and u combined [state_size, state_size + action_size, state_size + action_size].
                F_x: Jacobian of state path w.r.t. x
                    [N, state_size, state_size].
                F_u: Jacobian of state path w.r.t. u
                    [N, state_size, action_size].
                L: Cost path [N+1].
                L_x: Jacobian of cost path w.r.t. x [N+1, state_size].
                L_u: Jacobian of cost path w.r.t. u [N, action_size].
                L_xx: Hessian of cost path w.r.t. x, x
                    [N+1, state_size, state_size].
                L_ux: Hessian of cost path w.r.t. u, x
                    [N, action_size, state_size].
                L_uu: Hessian of cost path w.r.t. u, u
                    [N, action_size, action_size].
                F_xx: Hessian of state path w.r.t. x, x if Hessians are used
                    [N, state_size, state_size, state_size].
                F_ux: Hessian of state path w.r.t. u, x if Hessians are used
                    [N, state_size, action_size, state_size].
                F_uu: Hessian of state path w.r.t. u, u if Hessians are used
                    [N, state_size, action_size, action_size].
        """
        state_size = self.dynamics.state_size
        action_size = self.dynamics.action_size
        N = us_current.shape[0]

        xs = np.empty((N + 1, state_size + action_size))

        F_x = np.empty((N, state_size, state_size))
        F_u = np.empty((N, state_size, action_size))

        if self._use_hessians:
            F_xx = np.empty((N, state_size, state_size, state_size))
            F_ux = np.empty((N, state_size, action_size, state_size))
            F_uu = np.empty((N, state_size, action_size, action_size))
        else:
            F_xx = None
            F_ux = None
            F_uu = None

        L = np.empty(N + 1)
        L_x = np.empty((N + 1, state_size))
        L_u = np.empty((N, action_size))
        L_xx = np.empty((N + 1, state_size, state_size))
        L_ux = np.empty((N, action_size, state_size))
        L_uu = np.empty((N, action_size, action_size))

        xs[0] = x0

        for i in range(N):
            xs[i, state_size:] = us_current[i]
            x = xs[i]

            xs[i + 1][:state_size] = self.dynamics.f(x[:state_size], x[state_size:], i)
            F_x[i] = self.dynamics.f_x(x[:state_size], x[state_size:], i)
            F_u[i] = self.dynamics.f_u(x[:state_size], x[state_size:], i)

            L[i] = np.asarray(self.cost_GPS.l(x, x_old[i, :state_size, :], us_old[i], k[i], K[i], cov_u[i], i, terminal=False))
            L_x[i] = self.cost_GPS.l_x(x, x_old[i, :state_size, :], us_old[i], k[i], K[i], cov_u[i], i, terminal=False)
            L_u[i] = self.cost_GPS.l_u(x, x_old[i, :state_size, :], us_old[i], k[i], K[i], cov_u[i], i, terminal=False)
            L_xx[i] = self.cost_GPS.l_xx(x, x_old[i, :state_size, :], us_old[i], k[i], K[i], cov_u[i], i, terminal=False)
            L_ux[i] = self.cost_GPS.l_ux(x, x_old[i, :state_size, :], us_old[i], k[i], K[i], cov_u[i], i, terminal=False)
            L_uu[i] = self.cost_GPS.l_uu(x, x_old[i, :state_size, :], us_old[i], k[i], K[i], cov_u[i], i, terminal=False)

            if self._use_hessians:
                F_xx[i] = self.dynamics.f_xx(x[:state_size], x[state_size:], i)
                F_ux[i] = self.dynamics.f_ux(x[:state_size], x[state_size:], i)
                F_uu[i] = self.dynamics.f_uu(x[:state_size], x[state_size:], i)

        x = xs[-1]
        L[-1] = self.cost_GPS.l(x, None, None, None, None, None, N, terminal=True)
        L_x[-1] = self.cost_GPS.l_x(x, None, None, None, None, None, N, terminal=True)
        L_xx[-1] = self.cost_GPS.l_xx(x, None, None, None, None, None, N, terminal=True)

        return xs, F_x, F_u, L, L_x, L_u, L_xx, L_ux, L_uu, F_xx, F_ux, F_uu

    def _backward_pass_GPS(self,
                       F_x,
                       F_u,
                       L_x,
                       L_u,
                       L_xx,
                       L_ux,
                       L_uu,
                       F_xx=None,
                       F_ux=None,
                       F_uu=None):
        """Computes the feedforward and feedback gains k and K.

        Args:
            F_x: Jacobian of state path w.r.t. x [N, state_size, state_size].
            F_u: Jacobian of state path w.r.t. u [N, state_size, action_size].
            L_x: Jacobian of cost path w.r.t. x [N+1, state_size].
            L_u: Jacobian of cost path w.r.t. u [N, action_size].
            L_xx: Hessian of cost path w.r.t. x, x
                [N+1, state_size, state_size].
            L_ux: Hessian of cost path w.r.t. u, x [N, action_size, state_size].
            L_uu: Hessian of cost path w.r.t. u, u
                [N, action_size, action_size].
            F_xx: Hessian of state path w.r.t. x, x if Hessians are used
                [N, state_size, state_size, state_size].
            F_ux: Hessian of state path w.r.t. u, x if Hessians are used
                [N, state_size, action_size, state_size].
            F_uu: Hessian of state path w.r.t. u, u if Hessians are used
                [N, state_size, action_size, action_size].

        Returns:
            Tuple of
                k: feedforward gains [N, action_size].
                K: feedback gains [N, action_size, state_size].
        """
        V_x = L_x[-1]
        V_xx = L_xx[-1]

        k = np.empty_like(self._k)
        K = np.empty_like(self._K)
        Q = []

        for i in range(self.N - 1, -1, -1):
            if self._use_hessians:
                Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q_GPS(
                    F_x[i], F_u[i], L_x[i], L_u[i], L_xx[i], L_ux[i], L_uu[i],
                    V_x, V_xx, F_xx[i], F_ux[i], F_uu[i])
                Q.append(Q_uu)
            else:
                Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q_GPS(F_x[i], F_u[i], L_x[i],
                                                     L_u[i], L_xx[i], L_ux[i],
                                                     L_uu[i], V_x, V_xx)
                Q.append(Q_uu)

            # Eq (6).
            k[i] = -np.linalg.solve(Q_uu, Q_u)
            K[i] = -np.linalg.solve(Q_uu, Q_ux)

            # Eq (11b).
            V_x = Q_x + K[i].T.dot(Q_uu).dot(k[i])
            V_x += K[i].T.dot(Q_u) + Q_ux.T.dot(k[i])

            # Eq (11c).
            V_xx = Q_xx + K[i].T.dot(Q_uu).dot(K[i])
            V_xx += K[i].T.dot(Q_ux) + Q_ux.T.dot(K[i])
            V_xx = 0.5 * (V_xx + V_xx.T)  # To maintain symmetry.

        return np.array(k), np.array(K), np.array(Q)

    def _Q_GPS(self,
           f_x,
           f_u,
           l_x,
           l_u,
           l_xx,
           l_ux,
           l_uu,
           V_x,
           V_xx,
           f_xx=None,
           f_ux=None,
           f_uu=None):
        """Computes second order expansion.

        Args:
            F_x: Jacobian of state w.r.t. x [state_size, state_size].
            F_u: Jacobian of state w.r.t. u [state_size, action_size].
            L_x: Jacobian of cost w.r.t. x [state_size].
            L_u: Jacobian of cost w.r.t. u [action_size].
            L_xx: Hessian of cost w.r.t. x, x [state_size, state_size].
            L_ux: Hessian of cost w.r.t. u, x [action_size, state_size].
            L_uu: Hessian of cost w.r.t. u, u [action_size, action_size].
            V_x: Jacobian of the value function at the next time step
                [state_size].
            V_xx: Hessian of the value function at the next time step w.r.t.
                x, x [state_size, state_size].
            F_xx: Hessian of state w.r.t. x, x if Hessians are used
                [state_size, state_size, state_size].
            F_ux: Hessian of state w.r.t. u, x if Hessians are used
                [state_size, action_size, state_size].
            F_uu: Hessian of state w.r.t. u, u if Hessians are used
                [state_size, action_size, action_size].

        Returns:
            Tuple of
                Q_x: [state_size].
                Q_u: [action_size].
                Q_xx: [state_size, state_size].
                Q_ux: [action_size, state_size].
                Q_uu: [action_size, action_size].
        """
        # Eqs (5a), (5b) and (5c).
        Q_x = l_x + f_x.T.dot(V_x)
        Q_u = l_u + f_u.T.dot(V_x)
        Q_xx = l_xx + f_x.T.dot(V_xx).dot(f_x)

        # Eqs (11b) and (11c).
        reg = self._mu * np.eye(self.dynamics.state_size)
        Q_ux = l_ux + f_u.T.dot(V_xx + reg).dot(f_x)
        Q_uu = l_uu + f_u.T.dot(V_xx + reg).dot(f_u)

        if self._use_hessians:
            Q_xx += np.tensordot(V_x, f_xx, axes=1)
            Q_ux += np.tensordot(V_x, f_ux, axes=1)
            Q_uu += np.tensordot(V_x, f_uu, axes=1)

        return Q_x, Q_u, Q_xx, Q_ux, Q_uu

    def fit_GPS(self, x0, us_init, us_local, n_iterations=100, tol=1e-6, cov_method='MEM', on_iteration=None):
        """Computes the optimal controls.

        Args:
            x0: Initial state [state_size + action_size].
            us_init: Initial control path, more precisely local policy [N, action_size].
            n_iterations: Maximum number of interations. Default: 100.
            tol: Tolerance. Default: 1e-6.
            cov_method: Defines the way covariances should be calculated
                MEM: Maximum Entropy Method. Cov = (Q_uu)^-1
                FCM: Fixed Covariance Method. Cov = 0.01*eye(state_size + action_size, state_size + action_size)
            on_iteration: Callback at the end of each iteration with the
                following signature:
                (iteration_count, x, J_opt, accepted, converged) -> None
                where:
                    iteration_count: Current iteration count.
                    xs: Current state path.
                    us: Current action path.
                    J_opt: Optimal cost-to-go.
                    accepted: Whether this iteration yielded an accepted result.
                    converged: Whether this iteration converged successfully.
                Default: None.

        Returns:
            Tuple of
                xs: optimal state path [N+1, state_size].
                us: optimal control path [N, action_size].

        Note:
            eta: Range of eta is [0.001, 10].
            epsilon: Range of epsilon is [0.25, 1].
            alpha: Backtracking line search parameter alpha is set to 1 always. If you want to the search then backtracking 
                    line search candidates 0 < alpha <= 1. Code: alphas = 1.1**(-np.arange(10)**2)
        """
        # Determine state size and action size
        # Reset regularization term.
        self._mu = 1.0
        self._delta = self._delta_0

        # Make a copy of initial guess
        us = us_init.copy()
        xs = np.zeros((self.N + 1, self.dynamics.state_size + self.dynamics.action_size))

        # Control parameter
        # k: Open loop controller gain matrix and 
        # K: Closed loop feedback controller gain matrix
        k = self._k
        K = self._K

        changed = True
        converged = False
        for iteration in range(n_iterations):
            accepted = False

            if iteration == 0:
                # Set eta for the first iteration equal to one
                self.cost_GPS.eta = 1.0
                mean = []
                cov = []
                temp = np.matmul(self.A, x0[:, np.newaxis]) + self.B
                temp1 = self.C
                mean.append(temp)
                cov.append(temp1)
                mean = np.array(mean*(self.N+1))
                cov = np.array(cov*(self.N+1))
            # # else:
            # #     # Estimate the eta 
            # #     mean_xu, cov_xu, mean, cov = self.generate_mean_cov(xs[:, :self.dynamics.state_size], xs[:, self.dynamics.state_size:], k, K, 
            # #                                                                 self.A, self.B, self.C, mean, cov, Q_uu)
            # #     self.cost_GPS.eta = self.eta_estimation(mean_xu, cov_xu, us, us_init)

            # Forward rollout only if it needs to be recomputed.
            if changed:
                (xs, F_x, F_u, L, L_x, L_u, L_xx, L_ux, L_uu, F_xx, F_ux,
                F_uu) = self._forward_rollout_GPS(x0, xs, xs[:, self.dynamics.state_size:, :], us, k, K, self.cov_u)
                J_opt = L.sum()
                changed = False

            try:
                # Backward pass.
                k, K, Q_uu = self._backward_pass_GPS(F_x, F_u, L_x, L_u, L_xx, L_ux, L_uu,
                                        F_xx, F_ux, F_uu)

                xs_new, us_new = self._control_GPS(xs, us, k, K, alpha=1.0)
                J_new = self._trajectory_cost_GPS(xs_new, us)
                if J_new < J_opt:
                    if np.abs((J_opt - J_new) / J_opt) < tol:
                        converged = True
                    else:
                        mean_xu, cov_xu, mean, cov = self.generate_mean_cov(xs[:, :self.dynamics.state_size], xs[:, self.dynamics.state_size:], 
                                                                                k, K, self.A, self.B, self.C, mean, cov, Q_uu)
                        self.cost_GPS.eta = self.eta_estimation(mean_xu, cov_xu, us, us_new)
                        J_opt = J_new
                        xs = xs_new
                        us = us_new

                    cov_u = []
                    if cov_method == 'MEM':
                        for i in range(self.N):
                            cov_u.append(np.linalg.inv(Q_uu[i]))
                            self.cov_u = np.array(cov_u)
                    elif cov_method == 'FCM':
                            temp = 0.01 * np.eye(self.dynamics.action_size)
                            cov_u.append(temp)
                            self.cov_u = np.array(cov_u*self.N)
                    changed = True
                    # Decrease regularization term.
                    self._delta = min(1.0, self._delta) / self._delta_0
                    self._mu *= self._delta
                    if self._mu <= self._mu_min:
                        self._mu = 0.0
                    # Accept this.
                    accepted = True
            except np.linalg.LinAlgError as e:
                # Quu was not positive-definite and this diverged.
                # Try again with a higher regularization term.
                warnings.warn(str(e))

            if not accepted:
                # Increase regularization term.
                self._delta = max(1.0, self._delta) * self._delta_0
                self._mu = max(self._mu_min, self._mu * self._delta)
                if self._mu_max and self._mu >= self._mu_max:
                    warnings.warn("exceeded max regularization term")
                    break

            if on_iteration:
                on_iteration(iteration, xs, us, J_opt, accepted, converged, self.cost_GPS.eta)

            if converged:
                break

        # Store fit parameters.
        self._k = k
        self._K = K
        self._nominal_xs = xs
        self._nominal_us = us
        
        return xs, us, self.cov_u
    
    ### Unecessary function added to match the abstract class 'Basecontroller'
    def fit(self, x0, us_init, n_iterations=100, tol=1e-6, on_iteration=None):
        """Computes the optimal controls.
        Args:
            x0: Initial state [state_size].
            us_init: Initial control path [N, action_size].
            n_iterations: Maximum number of interations. Default: 100.
            tol: Tolerance. Default: 1e-6.
            on_iteration: Callback at the end of each iteration with the
                following signature:
                (iteration_count, x, J_opt, accepted, converged) -> None
                where:
                    iteration_count: Current iteration count.
                    xs: Current state path.
                    us: Current action path.
                    J_opt: Optimal cost-to-go.
                    accepted: Whether this iteration yielded an accepted result.
                    converged: Whether this iteration converged successfully.
                Default: None.

        Returns:
            Tuple of
                xs: optimal state path [N+1, state_size].
                us: optimal control path [N, action_size].
        """
        # Reset regularization term.
        self._mu = 1.0
        self._delta = self._delta_0

        # Backtracking line search candidates 0 < alpha <= 1.
        alphas = 1.1**(-np.arange(10)**2)

        us = us_init.copy()

        k = self._k
        K = self._K

        changed = True
        converged = False
        for iteration in range(n_iterations):
            accepted = False

            # Forward rollout only if it needs to be recomputed.
            if changed:
                (xs, F_x, F_u, L, L_x, L_u, L_xx, L_ux, L_uu, F_xx, F_ux,
                F_uu) = self._forward_rollout(x0, us, us_init)
                J_opt = L.sum()
                changed = False

            try:
                # Backward pass.
                k, K = self._backward_pass(F_x, F_u, L_x, L_u, L_xx, L_ux, L_uu,
                                        F_xx, F_ux, F_uu)
                # Backtracking line search.
                for alpha in alphas:
                    xs_new, us_new = self._control(xs, us, k, K, alpha)
                    J_new = self._trajectory_cost(xs_new, us_new)
                    if J_new < J_opt:
                        if np.abs((J_opt - J_new) / J_opt) < tol:
                            converged = True
                        J_opt = J_new
                        xs = xs_new
                        us = us_new
                        changed = True

                        # Decrease regularization term.
                        self._delta = min(1.0, self._delta) / self._delta_0
                        self._mu *= self._delta
                        if self._mu <= self._mu_min:
                            self._mu = 0.0

                        # Accept this.
                        accepted = True
                        break
            except np.linalg.LinAlgError as e:
                # Quu was not positive-definite and this diverged.
                # Try again with a higher regularization term.
                warnings.warn(str(e))

            if not accepted:
                # Increase regularization term.
                self._delta = max(1.0, self._delta) * self._delta_0
                self._mu = max(self._mu_min, self._mu * self._delta)
                if self._mu_max and self._mu >= self._mu_max:
                    warnings.warn("exceeded max regularization term")
                    break

            if on_iteration:
                on_iteration(iteration, xs, us, J_opt, accepted, converged)

            if converged:
                break

        # Store fit parameters.
        self._k = k
        self._K = K
        self._nominal_xs = xs
        self._nominal_us = us

        return xs, us

class RecedingHorizonController(object):

    """Receding horizon controller for Model Predictive Control."""

    def __init__(self, x0, controller):
        """Constructs a RecedingHorizonController.

        Args:
            x0: Initial state [state_size].
            controller: Controller to fit with.
        """
        self._x = x0
        self._controller = controller
        self._random = np.random.RandomState()

    def seed(self, seed):
        self._random.seed(seed)

    def set_state(self, x):
        """Sets the current state of the controller.

        Args:
            x: Current state [state_size].
        """
        self._x = x

    def control(self,
                us_init,
                step_size=1,
                initial_n_iterations=100,
                subsequent_n_iterations=1,
                *args,
                **kwargs):
        """Yields the optimal controls to run at every step as a receding
        horizon problem.

        Note: The first iteration will be slow, but the successive ones will be
        significantly faster.

        Note: This will automatically move the current controller's state to
        what the dynamics model believes will be the next state after applying
        the entire control path computed. Should you want to correct this state
        between iterations, simply use the `set_state()` method.

        Note: If your cost or dynamics are time dependent, then you might need
        to shift their internal state accordingly.

        Args:
            us_init: Initial control path [N, action_size].
            step_size: Number of steps between each controller fit. Default: 1.
                i.e. re-fit at every time step. You might need to increase this
                depending on how powerful your machine is in order to run this
                in real-time.
            initial_n_iterations: Initial max number of iterations to fit.
                Default: 100.
            subsequent_n_iterations: Subsequent max number of iterations to
                fit. Default: 1.
            *args, **kwargs: Additional positional and key-word arguments to
                pass to `controller.fit()`.

        Yields:
            Tuple of
                xs: optimal state path [step_size+1, state_size].
                us: optimal control path [step_size, action_size].
        """
        action_size = self._controller.dynamics.action_size
        n_iterations = initial_n_iterations
        while True:
            xs, us = self._controller.fit(
                self._x, us_init, n_iterations=n_iterations, *args, **kwargs)
            self._x = xs[step_size]
            yield xs[:step_size + 1], us[:step_size]

            # Set up next action path seed by simply moving along the current
            # optimal path and appending random unoptimal values at the end.
            us_start = us[step_size:]
            us_end = self._random.uniform(-1, 1, (step_size, action_size))
            us_init = np.vstack([us_start, us_end])
            n_iterations = subsequent_n_iterations
