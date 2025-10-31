import numpy as np
import matplotlib.pyplot as plt
import math

# --- Parameters ---
n = 100              # sample size per trial
n_trials = 100       # number of experiments
mu = 0.5             # mean of Uniform(0,1)
var_X = 1/12         # variance of Uniform(0,1)
var_Xbar = var_X / n # variance of sample mean

# --- Simulation ---
samples = np.random.uniform(0, 1, size=(n_trials, n))  # shape: (100,100)
X_bar = samples.mean(axis=1)                           # sample mean per trial
Zn = (X_bar - mu) / math.sqrt(var_Xbar)                # standardized statistic

# --- Empirical CDF ---
a_vals = np.array([-2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2])
empirical_cdf = [(Zn <= a).mean() for a in a_vals]     # P(Zn ≤ a)

# --- Theoretical Normal(0,1) CDF using erf ---
def normal_cdf(x):
    """Compute standard normal CDF Φ(x) using error function"""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

theoretical_cdf = [normal_cdf(a) for a in a_vals]

# --- Plot Empirical vs Theoretical CDF ---
plt.figure(figsize=(8,5))
plt.plot(a_vals, empirical_cdf, 'o-', label='Empirical CDF (Zn)', lw=2)
plt.plot(a_vals, theoretical_cdf, 's--', label='Normal(0,1) CDF', lw=2)
plt.xlabel('a', fontsize=12)
plt.ylabel('P(Zn ≤ a)', fontsize=12)
plt.title('Empirical vs Theoretical CDF of Zn', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# --- Optional: histogram vs normal PDF ---
x = np.linspace(-4, 4, 200)
pdf = (1/np.sqrt(2*np.pi)) * np.exp(-x**2 / 2)

plt.figure(figsize=(8,5))
plt.hist(Zn, bins=15, density=True, alpha=0.6, label='Empirical Zn histogram')
plt.plot(x, pdf, 'r--', lw=2, label='Normal(0,1) PDF')
plt.xlabel('Zn', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Histogram of Zn vs Standard Normal PDF', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
