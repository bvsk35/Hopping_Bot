import numpy as np
import scipy.linalg as linalg

"""
This file contains three classes. Purpose of this file is fit a Gaussian Mixture Model on the given
    Data:
        X: Rollouts x Trajectory steps x Data_dimension_X
        U: Rollouts x Trajectory steps x Data_dimension_U
    Returns:
        Fm: Trajectory steps x Data_dimension_X x (Data_dimension_X + Data_dimension_U)
        fv: Trajectory steps x Data_dimension_X
        dyn_cov: Trajectory steps x Data_dimension_X x Data_dimension_X
        matrices for all the trajectory steps (nothing but A, B, C of the Linear Guassian Policy)
Classes:
    1 - Gaussian_Mixture_Model(init_sequential=False, eigreg=False, warmstart=True)
    2 - Prior_Dynamics_GMM(Gaussian_Mixture_Model)
    3 - Estimated_Dynamics_Prior(Prior_Dynamics_GMM)
"""

class Gaussian_Mixture_Model(object):
    """Init Function"""
    def __init__(self, init_sequential=False, eigreg=False, warmstart=True):
        """ Gaussian Mixture Model:
            init_sequential: Cluster indices
        """
        # Parameters (need more explanation??)
        self.init_sequential = init_sequential 
        self.eigreg = eigreg
        self.warmstart = warmstart
        self.sigma = None
        
    # Need information regarding the function
    def logsum(self, vec, axis=0, keepdims=True):
        maxv = np.max(vec, axis=axis, keepdims=keepdims)
        maxv[maxv == -float('inf')] = 0
        return np.log(np.sum(np.exp(vec-maxv), axis=axis, keepdims=keepdims)) + maxv
    
    # Need information regarding the function
    def inference(self, pts):
        """
        Evaluate dynamics prior.
        Args:
            Pts: Matrix of points and of size (N x D).
        """
        # Compute posterior cluster weights.
        logwts = self.clusterwts(pts)

        # Compute posterior mean and covariance.
        mu0, Phi = self.moments(logwts)

        # Set hyperparameters.
        m = self.N
        n0 = m - 2 - mu0.shape[0]

        # Normalize.
        m = float(m) / self.N
        n0 = float(n0) / self.N
        return mu0, Phi, m, n0
    
    # Need information regarding the function
    def estep(self, data):
        """
        Compute Log observation probabilities under GMM.
        Args:
            data: Matrix of points of size (N x D).
        Returns:
            logobs: Matrix of log probabilities (for each point on each cluster) and of size (N x K).
        Note: 
            solve_triangular: Solve the equation a x = b for x, assuming a is a triangular matrix.
        """
        # Constants.
        N, D = data.shape
        K = self.sigma.shape[0]

        logobs = -0.5 * np.ones((N, K)) * D * np.log(2*np.pi)
        for i in range(K):
            mu, sigma = self.mu[i], self.sigma[i]
            L = linalg.cholesky(sigma, lower=True)
            logobs[:, i] -= np.sum(np.log(np.diag(L)))

            diff = (data - mu).T
            soln = linalg.solve_triangular(L, diff, lower=True)
            logobs[:, i] -= 0.5 * np.sum(soln**2, axis=0)

        logobs += self.logmass.T
        return logobs
    
    # Need information regarding the function
    def moments(self, logwts):
        """
        Compute the moments of the cluster mixture with logwts.
        Args:
            logwts: Matrix of log cluster probabilities and of size (K x 1).
        Returns:
            mu: Mean vector of size (D,).
            sigma: Covariance matrix of size (D x D).
        """
        # Exponentiate.
        wts = np.exp(logwts)

        # Compute overall mean.
        mu = np.sum(self.mu * wts, axis=0)

        # Compute overall covariance.
        diff = self.mu - np.expand_dims(mu, axis=0)
        diff_expand = np.expand_dims(self.mu, axis=1) * np.expand_dims(diff, axis=2)
        wts_expand = np.expand_dims(wts, axis=2)
        sigma = np.sum((self.sigma + diff_expand) * wts_expand, axis=0)
        return mu, sigma
    
    # Need information regarding the function
    def clusterwts(self, data):
        """
        Compute cluster weights for specified points under GMM.
        Args:
            data: Matrix of points and of size (N x D).
        Returns:
            Column vector of average cluster log probabilities and of size (K x 1).
        """
        # Compute probability of each point under each cluster.
        logobs = self.estep(data)

        # Renormalize to get cluster weights.
        logwts = logobs - self.logsum(logobs, axis=1)

        # Average the cluster probabilities.
        logwts = self.logsum(logwts, axis=0) - np.log(data.shape[0])
        return logwts.T
    
    # Need information regarding the function
    def update(self, data, K, max_iterations=100):
        """
        Run EM to update clusters.
        Args:
            data: An N x D data matrix, where N = number of data points.
            K: Number of clusters to use.
        """
        # Constants.
        N = data.shape[0]
        Do = data.shape[1]

        if (not self.warmstart or self.sigma is None or K != self.sigma.shape[0]):
            # Initialization.
            self.sigma = np.zeros((K, Do, Do))
            self.mu = np.zeros((K, Do))
            self.logmass = np.log(1.0 / K) * np.ones((K, 1))
            self.mass = (1.0 / K) * np.ones((K, 1))
            self.N = data.shape[0]
            N = self.N

            # Set initial cluster indices.
            if not self.init_sequential:
                cidx = np.random.randint(0, K, size=(1, N))
            else:
                raise NotImplementedError()

            # Initialize.
            for i in range(K):
                cluster_idx = (cidx == i)[0]
                mu = np.mean(data[cluster_idx, :], axis=0)
                diff = (data[cluster_idx, :] - mu).T
                sigma = (1.0 / K) * (diff.dot(diff.T))
                self.mu[i, :] = mu
                self.sigma[i, :, :] = sigma + np.eye(Do) * 2e-6

        prevll = -float('inf')
        for itr in range(max_iterations):
            # E-step: compute cluster probabilities.
            logobs = self.estep(data)

            # Compute log-likelihood.
            ll = np.sum(self.logsum(logobs, axis=1))
            if ll < prevll:
                # TODO: Why does log-likelihood decrease sometimes?
                break
            if np.abs(ll-prevll) < 1e-5*prevll:
                break
            prevll = ll

            # Renormalize to get cluster weights.
            logw = logobs - self.logsum(logobs, axis=1)
            assert logw.shape == (N, K)

            # Renormalize again to get weights for refitting clusters.
            logwn = logw - self.logsum(logw, axis=0)
            assert logwn.shape == (N, K)
            w = np.exp(logwn)

            # M-step: update clusters.
            # Fit cluster mass.
            self.logmass = self.logsum(logw, axis=0).T
            self.logmass = self.logmass - self.logsum(self.logmass, axis=0)
            assert self.logmass.shape == (K, 1)
            self.mass = np.exp(self.logmass)
            # Reboot small clusters.
            w[:, (self.mass < (1.0 / K) * 1e-4)[:, 0]] = 1.0 / N
            # Fit cluster means.
            w_expand = np.expand_dims(w, axis=2)
            data_expand = np.expand_dims(data, axis=1)
            self.mu = np.sum(w_expand * data_expand, axis=0)
            # Fit covariances.
            wdata = data_expand * np.sqrt(w_expand)
            assert wdata.shape == (N, K, Do)
            for i in range(K):
                # Compute weighted outer product.
                XX = wdata[:, i, :].T.dot(wdata[:, i, :])
                mu = self.mu[i, :]
                self.sigma[i, :, :] = XX - np.outer(mu, mu)

                if self.eigreg:  # Use eigenvalue regularization.
                    raise NotImplementedError()
                else:  # Use quick and dirty regularization.
                    sigma = self.sigma[i, :, :]
                    self.sigma[i, :, :] = 0.5 * (sigma + sigma.T) + 1e-6 * np.eye(Do)

class Prior_Dynamics_GMM(object):
    """
    A dynamics prior encoded as a GMM over [x_t, u_t, x_t+1] points.
    See:
        S. Levine*, C. Finn*, T. Darrell, P. Abbeel, "End-to-end
        training of Deep Visuomotor Policies", arXiv:1504.00702,
        Appendix A.3.
    """
    def __init__(self, init_sequential=False, eigreg=False, warmstart=True, 
                 min_samples_per_cluster=20, max_clusters=50, max_samples=20, strength=1.0):
        """
        Hyperparameters:
            min_samples_per_cluster: Minimum samples per cluster.
            max_clusters: Maximum number of clusters to fit.
            max_samples: Maximum number of trajectories to use for fitting the GMM at any given time.
            strength: Adjusts the strength of the prior.
        """
        self._min_samples_per_cluster = min_samples_per_cluster
        self._max_clusters = max_clusters
        self._max_samples = max_samples
        self._strength = strength
        self.X = None
        self.U = None
        self.gmm = Gaussian_Mixture_Model(init_sequential=init_sequential, eigreg=eigreg, warmstart=warmstart)

    def initial_state(self):
        """ Return dynamics prior for initial time step. """
        # Compute mean and covariance.
        mu0 = np.mean(self.X[:, 0, :], axis=0)
        Phi = np.diag(np.var(self.X[:, 0, :], axis=0))

        # Factor in multiplier.
        n0 = self.X.shape[2] * self._strength
        m = self.X.shape[2] * self._strength

        # Multiply Phi by m (since it was normalized before).
        Phi = Phi * m
        return mu0, Phi, m, n0

    def update(self, X, U):
        """
        Update prior with additional data.
        Args:
            X: (N x T x dX) matrix of sequential state data.
            U: (N x T x dU) matrix of sequential control data.
        """
        # Constants.
        T = X.shape[1] - 1

        # Append data to dataset.
        if self.X is None:
            self.X = X
        else:
            self.X = np.concatenate([self.X, X], axis=0)

        if self.U is None:
            self.U = U
        else:
            self.U = np.concatenate([self.U, U], axis=0)

        # Remove excess samples from dataset.
        start = max(0, self.X.shape[0] - self._max_samples + 1)
        self.X = self.X[start:, :]
        self.U = self.U[start:, :]

        # Compute cluster dimensionality.
        Do = X.shape[2] + U.shape[2] + X.shape[2]  #TODO: Use Xtgt.

        # Create dataset.
        N = self.X.shape[0]
        xux = np.reshape(
            np.c_[self.X[:, :T, :], self.U[:, :T, :], self.X[:, 1:(T+1), :]],
            [T * N, Do]
        )

        # Choose number of clusters.
        K = int(max(2, min(self._max_clusters,
                           np.floor(float(N * T) / self._min_samp))))

        # Update GMM.
        self.gmm.update(xux, K)

    def eval(self, Dx, Du, pts):
        """
        Evaluate prior.
        Args:
            pts: A N x Dx+Du+Dx matrix.
        """
        # Construct query data point by rearranging entries and adding
        # in reference.
        assert pts.shape[1] == Dx + Du + Dx

        # Perform query and fix mean.
        mu0, Phi, m, n0 = self.gmm.inference(pts)

        # Factor in multiplier.
        n0 = n0 * self._strength
        m = m * self._strength

        # Multiply Phi by m (since it was normalized before).
        Phi *= m
        return mu0, Phi, m, n0

class Estimated_Dynamics_Prior(object):
    """ Dynamics with linear regression, with arbitrary prior. """
    def __init__(self, init_sequential=False, eigreg=False, warmstart=True, 
                 min_samples_per_cluster=20, max_clusters=50, max_samples=20, strength=1.0):
        self.Fm = None
        self.fv = None
        self.dyn_covar = None
        self.prior = Prior_Dynamics_GMM(init_sequential=False, eigreg=False, warmstart=True, 
                                        min_samples_per_cluster=20, max_clusters=50, max_samples=20, strength=1.0)

    def update_prior(self, X, U):
        """ Update dynamics prior. """
        self.prior.update(X, U)
    
    def gauss_fit_joint_prior(self, pts, mu0, Phi, m, n0, dwts, dX, dU, sig_reg):
        """ Perform Gaussian fit to data with a prior. """
        # Build weights matrix.
        D = np.diag(dwts)
        # Compute empirical mean and covariance.
        mun = np.sum((pts.T * dwts).T, axis=0)
        diff = pts - mun
        empsig = diff.T.dot(D).dot(diff)
        empsig = 0.5 * (empsig + empsig.T)
        # MAP estimate of joint distribution.
        N = dwts.shape[0]
        mu = mun
        sigma = (N * empsig + Phi + (N * m) / (N + m) *
                 np.outer(mun - mu0, mun - mu0)) / (N + n0)
        sigma = 0.5 * (sigma + sigma.T)
        # Add sigma regularization.
        sigma += sig_reg
        # Conditioning to get dynamics.
        fd = np.linalg.solve(sigma[:dX, :dX], sigma[:dX, dX:dX+dU]).T
        fc = mu[dX:dX+dU] - fd.dot(mu[:dX])
        dynsig = sigma[dX:dX+dU, dX:dX+dU] - fd.dot(sigma[:dX, :dX]).dot(fd.T)
        dynsig = 0.5 * (dynsig + dynsig.T)
        return fd, fc, dynsig

    def fit(self, X, U):
        """ Fit dynamics. """
        N, T, dX = X.shape
        dU = U.shape[2]

        if N == 1:
            raise ValueError("Cannot fit dynamics on 1 sample")

        self.Fm = np.zeros([T, dX, dX+dU])
        self.fv = np.zeros([T, dX])
        self.dyn_covar = np.zeros([T, dX, dX])

        it = slice(dX+dU)
        ip = slice(dX+dU, dX+dU+dX)
        # Fit dynamics with least squares regression.
        dwts = (1.0 / N) * np.ones(N)
        for t in range(T - 1):
            Ys = np.c_[X[:, t, :], U[:, t, :], X[:, t+1, :]]
            # Obtain Normal-inverse-Wishart prior.
            mu0, Phi, mm, n0 = self.prior.eval(dX, dU, Ys)
            sig_reg = np.zeros((dX+dU+dX, dX+dU+dX))
            sig_reg[it, it] = 1e-6
            Fm, fv, dyn_covar = self.gauss_fit_joint_prior(Ys, mu0, Phi, mm, n0, dwts, dX+dU, dX, sig_reg)
            self.Fm[t, :, :] = Fm
            self.fv[t, :] = fv
            self.dyn_covar[t, :, :] = dyn_covar
        return self.Fm, self.fv, self.dyn_covar