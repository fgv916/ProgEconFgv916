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
        par.kappa = 9.0
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

    
    # ------------------------------------------------------------------ 
    # Top tax (Assignment part 3)
    # ------------------------------------------------------------------ 

    # ------------------------------------------------------------------ 
    # Post tax income top
    # ------------------------------------------------------------------ 
    def post_tax_income_top(self, p, ell):
        par = self.par

        pre_tax = self.income(p, ell)             
        top_base = np.maximum(pre_tax - par.kappa, 0.0)

        tax = par.tau * pre_tax + par.omega * top_base + par.zeta
        
        return pre_tax - tax
    # ------------------------------------------------------------------
    # FOC before and after 
    # ------------------------------------------------------------------

    def FOC_top_before(self, p, ell):
        par = self.par
        c = self.post_tax_income_top(p, ell)
        if np.any(c <= 0):
            return 1e6
        return (1 - par.tau) * par.w * p / c - par.nu * ell**par.epsilon

    def FOC_top_after(self, p, ell):
        par = self.par
        c = self.post_tax_income_top(p, ell)
        if np.any(c <= 0):
            return 1e6
        return (1 - par.tau - par.omega) * par.w * p / c - par.nu * ell**par.epsilon

    # ------------------------------------------------------------------
    # Optimal choice with top tax
    # ------------------------------------------------------------------
    def optimal_choice_top(self, p):
        
        par = self.par
        opt = SimpleNamespace()

        def obj(ell):
            c = self.post_tax_income_top(p, ell)
            return -self.utility(c, ell)

        ell_min = self.get_min_ell(p)
        ell_max = par.ell_max

        res = minimize_scalar(obj, bounds=(ell_min, ell_max), method='bounded')

        opt.ell = res.x
        opt.c   = self.post_tax_income_top(p, opt.ell)
        opt.U   = self.utility(opt.c, opt.ell)
        return opt 

    # ------------------------------------------------------------------
    # Optimal choice with top tax using 4-step FOC (with region label)
    # ------------------------------------------------------------------
        
    def optimal_choice_top_FOC(self, p):

        par = self.par
        opt = SimpleNamespace()

        ell_min = self.get_min_ell(p)
        ell_max = par.ell_max

        ell_k = par.kappa / (par.w * p)
        eps = 1e-6 

        # ---------- STEP 1 ----------
        U_b = -1e10
        ell_b = None

        upper_1 = min(ell_k - eps, ell_max)
        if upper_1 > ell_min:

            def obj_before(ell):
                return self.FOC_top_before(p, ell)

            try:
                res1 = root_scalar(obj_before, bracket=(ell_min, upper_1), method='bisect')
                ell_b = res1.root
                c_b = self.post_tax_income_top(p, ell_b)
                U_b = self.utility(c_b, ell_b)
            except ValueError:
                U_b = -1e10
                ell_b = None

        # ---------- STEP 2 ----------
        U_k = -1e10
        ell_k_feasible = None

        if (ell_k >= ell_min) and (ell_k <= ell_max):
            c_k = self.post_tax_income_top(p, ell_k)
            if c_k > 0:
                U_k = self.utility(c_k, ell_k)
                ell_k_feasible = ell_k

        # ---------- STEP 3 ----------
        U_a = -1e10
        ell_a = None

        lower_3 = max(ell_k + eps, ell_min)
        if lower_3 < ell_max:

            def obj_after(ell):
                return self.FOC_top_after(p, ell)

            try:
                res3 = root_scalar(obj_after, bracket=(lower_3, ell_max), method='bisect')
                ell_a = res3.root
                c_a = self.post_tax_income_top(p, ell_a)
                U_a = self.utility(c_a, ell_a)
            except ValueError:
                U_a = -1e10
                ell_a = None

        # ---------- STEP 4 ----------
        U_candidates = [U_b, U_k, U_a]
        ell_candidates = [ell_b, ell_k_feasible, ell_a]

        idx_best = int(np.argmax(U_candidates))
        U_star = U_candidates[idx_best]
        ell_star = ell_candidates[idx_best]

        if (ell_star is None) or (U_star <= -1e9):
            ell_star = ell_max
            c_star = self.post_tax_income_top(p, ell_star)
            U_star = self.utility(c_star, ell_star)
            region = 'corner'
        else:
            c_star = self.post_tax_income_top(p, ell_star)
            region = ['b','k','a'][idx_best]

        opt.ell = ell_star
        opt.c   = c_star
        opt.U   = U_star
        opt.region = region

        return opt


    # ------------------------------------------------------------------
    # Plot under top tax system (utility + FOC before + FOC after)
    # ------------------------------------------------------------------

    def plot_top_tax(self, p):
        par = self.par

        # Grid and kink
        ell_max = par.ell_max
        ell_grid = np.linspace(0.5, ell_max, 400)
        ell_k = par.kappa / (par.w * p)

        # Utility on full grid
        c_grid = self.post_tax_income_top(p, ell_grid)
        U_grid = self.utility(c_grid, ell_grid)

        # Split grid before/after kink
        ell_left  = ell_grid[ell_grid <= ell_k]
        ell_right = ell_grid[ell_grid >= ell_k]

        phi_left  = self.FOC_top_before(p, ell_left)
        phi_right = self.FOC_top_after(p, ell_right)

        # Optimal labor supply using optimizer and four step
        opt_num = self.optimal_choice_top(p)
        print(f"ℓ* (optimizer) = {opt_num.ell:.9f}, U = {opt_num.U:.9f}")

        opt_foc = self.optimal_choice_top_FOC(p)
        print(f"ℓ* (4-step FOC) = {opt_foc.ell:.9f}, U = {opt_foc.U:.9f}")


        # ========== PLOTS ==========
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # --- Panel 1: Utility ---
        ax = axes[0]
        ax.plot(ell_grid, U_grid, label='U(ℓ)')
        ax.axvline(ell_k, color='grey', linestyle='--', label='kink ℓ_k')
        ax.axvline(opt_num.ell, color='red', linestyle=':', label='ℓ* optimizer')
        ax.set_xlabel(r'Labor supply $\ell$')
        ax.set_ylabel('Utility')
        ax.set_title(f'Utility, p = {p}')
        ax.grid(True)
        ax.legend()

        # --- Panel 2: FOC before kink ---
        ax = axes[1]
        ax.plot(ell_left, phi_left, label=r'$\phi(\ell)$ før kink')
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(ell_k, color='grey', linestyle='--', label='kink ℓ_k')
        ax.set_xlabel(r'Labor supply $\ell$')
        ax.set_ylabel('FOC')
        ax.set_title('FOC before top tax')
        ax.grid(True)
        ax.legend()

        # --- Panel 3: FOC after kink ---
        ax = axes[2]
        ax.plot(ell_right, phi_right, label=r'$\phi(\ell)$ efter kink')
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(ell_k, color='grey', linestyle='--', label='kink ℓ_k')
        ax.set_xlabel(r'Labor supply $\ell$')
        ax.set_ylabel('FOC')
        ax.set_title('FOC after top tax')
        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        plt.show()
    
    # ------------------------------------------------------------------
    # Labor supply and consumption under top tax 
    # ------------------------------------------------------------------
    def compute_labor_supply_top(self):

        par = self.par
        ps = par.ps

        ell_star = np.zeros_like(ps)
        c_star   = np.zeros_like(ps)
        regions  = np.empty(ps.shape, dtype=object)

        for i, p in enumerate(ps):
            opt = self.optimal_choice_top_FOC(p)
            ell_star[i] = opt.ell
            c_star[i]   = opt.c
            regions[i]  = opt.region

        return ps, ell_star, c_star, regions

    # ------------------------------------------------------------------
    # Proportions in each region (b, k, a)
    # ------------------------------------------------------------------
    def region_proportions_top(self):

        _, _, _, regions = self.compute_labor_supply_top()

        share_b = np.mean(regions == 'b')
        share_k = np.mean(regions == 'k')
        share_a = np.mean(regions == 'a')

        print(f"Share with ℓ* = ℓ^b (before kink): {share_b:.3f}")
        print(f"Share with ℓ* = ℓ^k (at kink):      {share_k:.3f}")
        print(f"Share with ℓ* = ℓ^a (after kink):   {share_a:.3f}")

        # hvis du vil bruge dem i notebooken
        return share_b, share_k, share_a
    
    # ------------------------------------------------------------------
    # Plot for 3.2.1: ℓ*(p) and c(p)
    # ------------------------------------------------------------------
    def plot_labor_and_consumption_top(self):

        ps, ell_star, c_star, _ = self.compute_labor_supply_top()

        fig, ax = plt.subplots(1, 2, figsize=(12,4))

        ax[0].plot(ps, ell_star)
        ax[0].set_title("Optimal labor supply ℓ*(p)")
        ax[0].set_xlabel("Productivity p")
        ax[0].set_ylabel("Labor supply ℓ*")
        ax[0].grid(True)

        # --- Plot c(p)
        ax[1].plot(ps, c_star)
        ax[1].set_title("Consumption c(p) = y(p, ℓ*(p))")
        ax[1].set_xlabel("Productivity p")
        ax[1].set_ylabel("Consumption c")
        ax[1].grid(True)

        plt.tight_layout()
        plt.show()


    
    