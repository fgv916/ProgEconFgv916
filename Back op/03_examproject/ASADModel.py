import numpy as np
import matplotlib.pyplot as plt

class ASADModelClass:

    def __init__(self, par=None):
    
        par = dict(
            ybar    = 1.0,
            pi_star = 0.02,
            b       = 0.6,
            a1      = 1.5,
            a2      = 0.10,
            gamma   = 4.0,
            phi     = 0.6
        )

        self.par = par.copy()

    def _alpha_z(self, v):

        p = self.par

        alpha = p['b'] * p['a1'] / (1.0 + p['b'] * p['a2'])
        z = v / (1.0 + p['b'] * p['a2'])

        return alpha, z

    def AD_curve(self, y, v):

        p = self.par
        alpha, z = self._alpha_z(v)
        inv_alpha = 1.0 / alpha
        return p['pi_star'] - inv_alpha * ((y - p['ybar']) - z)

    def SRAS_curve(self, y, pi_e):
        
        p = self.par
        return pi_e + p['gamma'] * (y - p['ybar'])

# analytical equilibrium y_t^*, pi_t^* given pi_e and v
    def equilibrium(self, pi_e, v):
        """
        Analytical equilibrium y_t^*, pi_t^* given expected inflation pi_e
        and demand shock v
        """
        p = self.par
        alpha, z = self._alpha_z(v)

        # The error was here: the exam requires 1/alpha in the denominator
        inv_alpha = 1.0 / alpha
        denom = inv_alpha + p['gamma'] # Corrected denominator [cite: 131]

        # Equation 10: y_t* calculation [cite: 131]
        y_star = p['ybar'] + (1.0 / denom) * (p['pi_star'] - pi_e + inv_alpha * z)
        
        # Equation 11: pi_t* calculation [cite: 124, 131]
        pi_star = pi_e + p['gamma'] * (y_star - p['ybar'])

        return y_star, pi_star
    

    # simulation (not used directly in 3.2 Q2)
    def simulate(self, rho, eps):
        T = len(eps)
        p = self.par

        v = np.zeros(T)
        y = np.zeros(T)
        pi = np.zeros(T)
        pi_e = np.zeros(T)

        pi_e[0] = p['pi_star']

        for t in range(1, T):
            v[t] = rho * v[t-1] + eps[t]
            pi_e[t] = p['phi'] * pi_e[t-1] + (1 - p['phi']) * pi[t-1]

        for t in range(T):
            y[t], pi[t] = self.equilibrium(pi_e[t], v[t])

        return y, pi, v

# compute sd(y_gap), sd(pi), corr(y_gap, pi) def moments(self, y, pi): raise NotImplementedError