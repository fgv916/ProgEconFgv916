import matplotlib.pyplot as plt
from types import SimpleNamespace
import numpy as np
from scipy.optimize import minimize_scalar, root_scalar

class WorkerClass:

    def __init__(self,par=None):

        self.setup_worker()

        if par is not None: 
            for k,v in par.items():
                self.par.__dict__[k] = v

    def setup_worker(self):

        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # preferences
        par.nu = 0.015
        par.epsilon = 1.0
        
        # productivity and wages
        par.w = 1.0
        par.ps = np.linspace(0.5,3.0,100)
        par.ell_max = 16.0
        
        # taxes
        par.tau = 0.50
        par.zeta = 0.10
        par.kappa = np.nan
        par.omega = 0.20

    # ------------------------------------------------------------------
    # Preferences / utility
    # ------------------------------------------------------------------
          
    def utility(self,c,ell):

        par = self.par

        if np.any(c <= 0):
            return -1e10

        u = np.log(c) - par.nu * ell**(1+par.epsilon)/(1+par.epsilon)
        return u
    
    # ------------------------------------------------------------------
    # Taxes and income
    # ------------------------------------------------------------------


    def tax(self,pre_tax_income):

        par = self.par

        tax = par.tau * pre_tax_income + par.zeta
        return tax
    
    def income(self,p,ell):

        par = self.par
        return par.w * p * ell

    def post_tax_income(self,p,ell):

        pre_tax_income = self.income(p,ell)
        tax = self.tax(pre_tax_income)
        return pre_tax_income - tax
    
    def max_post_tax_income(self,p):

        par = self.par
        return self.post_tax_income(p,par.ell_max)

    def value_of_choice(self,p,ell):

        c = self.post_tax_income(p,ell)
        U = self.utility(c,ell)
        return U
    
    # ------------------------------------------------------------------
    # Lower bound on labor supply (positive consumption condition)
    # ------------------------------------------------------------------

    
    def get_min_ell(self,p):
    
        par = self.par

        min_ell = par.zeta / (par.w * p * (1 - par.tau))
        return np.fmax(min_ell,0.0) + 1e-8

    # ------------------------------------------------------------------
    # Optimal choice via direct maximization of U
    # ------------------------------------------------------------------

    
    def optimal_choice(self,p):

        par = self.par
        opt = SimpleNamespace()

        # objective: negative utility
        def obj(ell):
            c = self.post_tax_income(p,ell)
            return -self.utility(c,ell)

        # bounds
        ell_min = self.get_min_ell(p)
        ell_max = par.ell_max

        res = minimize_scalar(obj, bounds=(ell_min, ell_max), method='bounded')

        # results
        opt.ell = res.x
        opt.U = -res.fun
        opt.c = self.post_tax_income(p,opt.ell)

        return opt

    # ------------------------------------------------------------------
    # First-order condition (FOC)
    # ------------------------------------------------------------------

    
    def FOC(self,p,ell):

        par = self.par

        c = self.post_tax_income(p,ell)

        bad = (c <= 0)

        # compute FOC (eq. 5): φ = (1-τ) w p / c - ν ℓ^ε
        FOC = (1 - par.tau) * par.w * p / c - par.nu * ell**par.epsilon

        FOC = np.where(bad, 1e6, FOC)

        return FOC
    
    # ------------------------------------------------------------------
    # Optimal choice via solving the FOC
    # ------------------------------------------------------------------   

    def optimal_choice_FOC(self,p):

        par = self.par
        opt = SimpleNamespace()

        ell_min = self.get_min_ell(p)
        ell_max = par.ell_max

        def obj(ell):
            return self.FOC(p,ell)

        try:
            res = root_scalar(obj, bracket=(ell_min, ell_max), method='bisect')
            ell_star = res.root
        except ValueError:
            ell_star = ell_max

        opt.ell = ell_star
        opt.c = self.post_tax_income(p,ell_star)
        opt.U = self.utility(opt.c,ell_star)

        return opt

    # ------------------------------------------------------------------
    # Plotting utility and FOC for given epsilon
    # ------------------------------------------------------------------  
        
    def plot_for_epsilon(self, eps, p, fig_title):
        self.par.epsilon = eps
        ell_grid = np.linspace(0.5, self.par.ell_max, 300)

        # grid solve for consumption, utility and FOC
        c_grid  = self.post_tax_income(p, ell_grid)
        U_grid  = self.utility(c_grid, ell_grid)
        phi_grid = self.FOC(p, ell_grid)

        # optimal choices from both methods
        opt_num  = self.optimal_choice(p)
        opt_root = self.optimal_choice_FOC(p)
        print(f"epsilon = {eps}")
        print(f"  ell* (optimizer):   {opt_num.ell:.6f}")
        print(f"  ell* (root finder): {opt_root.ell:.6f}")
        print()

        # --- subplots ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

        # 1) Utility plot (left)
        ax1.plot(ell_grid, U_grid, color='C0', label='Utility U(ℓ)')
        ax1.axhline(0, color='black', linewidth=1)
        ax1.axvline(opt_num.ell,  color='C2', linestyle='--', label='ell* optimizer')
        ax1.axvline(opt_root.ell, color='C3', linestyle=':',  label='ell* root finder')
        ax1.set_ylim(-0.5, 1.5)
        ax1.set_xlabel(r'Labor supply $\ell_i$')
        ax1.set_ylabel('Utility')
        ax1.set_title('Utility')
        ax1.grid(True)
        ax1.legend()

        # 2) FOC plot (right)
        ax2.plot(ell_grid, phi_grid, color='C1', label='FOC φ(ℓ)')
        ax2.axhline(0, color='black', linewidth=1)
        ax2.axvline(opt_num.ell,  color='C2', linestyle='--', label='ell* optimizer')
        ax2.axvline(opt_root.ell, color='C3', linestyle=':',  label='ell* root finder')
        ax2.set_ylim(-0.5, 1.5)
        ax2.set_xlabel(r'Labor supply $\ell_i$')
        ax2.set_ylabel('FOC')
        ax2.set_title('First Order Condition')
        ax2.grid(True)
        ax2.legend()

        fig.suptitle(fig_title, fontsize=14)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ 
    # Plotting optimal labor supply over productivity
    # ------------------------------------------------------------------ 

    def compute_labor_supply(self, zeta):

        worker = WorkerClass()
        worker.par.zeta = zeta

        ps = worker.par.ps
        ell_star = np.zeros_like(ps)

        for i, p in enumerate(ps):
            ell_star[i] = worker.optimal_choice(p).ell

        return ps, ell_star

    def plot_labor_supply(self, *zetas):

        plt.figure(figsize=(7,5))

        for z in zetas:
            ps, ell_star = self.compute_labor_supply(z)
            plt.plot(ps, ell_star, label=fr'$\zeta = {z}$')

        plt.xlabel("Productivity $p_i$")
        plt.ylabel("Optimal labor supply $\\ell^*(p_i)$")
        plt.title("Figur 4: Labor supply for different values of $\\zeta$")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
