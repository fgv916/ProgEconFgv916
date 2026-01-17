from Worker import WorkerClass
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class GovernmentClass(WorkerClass):

    def __init__(self, par=None):

        # a. default setup
        self.setup_worker()

        # Worker parameters that might not exist yet
        if not hasattr(self.par, 'L'):
            self.par.L = 16
        if not hasattr(self.par, 'w'):
            self.par.w = 1.0
        if not hasattr(self.par, 'epsilon'):
            self.par.epsilon = 1.0
        if not hasattr(self.par, 'nu'):
            self.par.nu = 0.015

        # Government-specific setup
        self.setup_government()

        # Update parameters if provided
        if par is not None:
            for k, v in par.items():
                self.par.__dict__[k] = v

        # Random number generator
        self.rng = np.random.default_rng(12345)

    def setup_government(self):

        par = self.par

        # a. workers
        par.N = 100         # number of workers
        par.sigma_p = 0.3   # std dev of productivity

        # b. public good
        par.chi = 50.0      # weight on public good in SWF
        par.eta = 0.1       # curvature of public good in SWF

    # --------------------------------------------------------
    # 1. Draw productivities p_i
    # --------------------------------------------------------
    def draw_productivities(self):

        par = self.par
        mu = -0.5 * par.sigma_p ** 2

        self.sol.pi = self.rng.lognormal(mean=mu, sigma=par.sigma_p, size=par.N)

    # --------------------------------------------------------
    # 2. Solve labor supply for one worker
    # --------------------------------------------------------
    def solve_labor_supply(self, pi):

        par = self.par

        # Minimum labor supply ensuring c > 0
        # Note: protect against division by zero if tau = 1.
        if (1 - par.tau) * par.w * pi > 0:
            ell_min = max(par.zeta / ((1 - par.tau) * par.w * pi), 0)
        else:
            ell_min = 0

        ell_max = par.L

        # Utility function
        def utility(ell):
            c = (1 - par.tau) * par.w * pi * ell - par.zeta
            if c <= 0:
                return -1e12
            return np.log(c) - par.nu * ell ** (1 + par.epsilon) / (1 + par.epsilon)

        # Objective function (negative utility)
        def objective(x):
            return -utility(x[0])

        # Optimization
        res = minimize(
            objective,
            x0=[max(0.1, ell_min)],
            bounds=[(ell_min, ell_max)]
        )

        ell_opt = res.x[0]
        U_opt = utility(ell_opt)

        return ell_opt, U_opt

    # --------------------------------------------------------
    # 3. Solve all workers
    # --------------------------------------------------------
    def solve_workers(self):

        par = self.par
        sol = self.sol

        ls = np.zeros(par.N)
        util = np.zeros(par.N)

        for i in range(par.N):
            ls[i], util[i] = self.solve_labor_supply(sol.pi[i])

        sol.l = ls
        sol.U = util

    # --------------------------------------------------------
    # 4. Compute tax revenue
    # --------------------------------------------------------
    def tax_revenue(self):

        par = self.par
        sol = self.sol

        # Corrected multiplication *
        G = par.N * par.zeta + np.sum(par.tau * par.w * sol.pi * sol.l)

        return G

    # --------------------------------------------------------
    # 5. Compute SWF = χ G^η + Σ U_i
    # --------------------------------------------------------
    def SWF(self):

        par = self.par
        sol = self.sol

        G = self.tax_revenue()
        if G < 0:
            return np.nan

        SWF_value = par.chi * (G ** par.eta) + np.sum(sol.U)
        return SWF_value
    
    # --------------------------------------------------------
    # 3 Top tax
    # --------------------------------------------------------