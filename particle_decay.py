import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2

# ------------------------
# Vacuum decay analysis
# ------------------------

# Load vacuum data (assumes file is in JSON format with a list of decay distances)
with open('Vacuum_decay_dataset.json', 'r') as f:
    vacuum_data = np.array(json.load(f))

# Define the shifted exponential PDF for x >= 1:
def exp_pdf(x, omega):
    # Note: the normalization is such that ∫₁∞ (1/ω exp(-(x-1)/ω)) dx = 1.
    return (1/omega) * np.exp(-(x-1)/omega)

# Log-likelihood for vacuum data
def log_likelihood_vacuum(omega, data):
    # omega must be > 0
    if omega <= 0:
        return -np.inf
    return np.sum(np.log(exp_pdf(data, omega)))

# The analytical MLE is ω = mean(x) - 1.
omega_vacuum = np.mean(vacuum_data) - 1
ll_vacuum = log_likelihood_vacuum(omega_vacuum, vacuum_data)
print("Vacuum decay analysis:")
print("Estimated decay constant ω =", omega_vacuum)
print("Log-likelihood =", ll_vacuum)

# Plot the vacuum data histogram and fitted PDF
plt.figure(figsize=(8,5))
plt.hist(vacuum_data, bins=50, density=True, alpha=0.6, label='Vacuum data')
x_plot = np.linspace(1, vacuum_data.max(), 200)
plt.plot(x_plot, exp_pdf(x_plot, omega_vacuum), 'r-', lw=2, label='Fitted exponential')
plt.xlabel('Decay distance x')
plt.ylabel('Probability density')
plt.legend()
plt.title('Vacuum Decay: Data and Exponential Fit')
plt.show()
plt.savefig("Vacuum_decay.png")

# ------------------------
# Cavity decay analysis
# ------------------------

# Load cavity data (assumes JSON file with a list of decay distances)
with open('Cavity_decay_dataset.json', 'r') as f:
    cavity_data = np.array(json.load(f))

# Define the Gaussian PDF.
def gauss_pdf(x, mu, sigma):
    return (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x-mu)/sigma)**2)

# Mixture model PDF: mixture of shifted exponential and Gaussian.
# Parameters: p = [f, omega, mu, sigma]
# f is the fraction from the Gaussian component.
def mixture_pdf(x, params):
    f, omega, mu, sigma = params
    return (1-f)*exp_pdf(x, omega) + f*gauss_pdf(x, mu, sigma)

# Negative log likelihood for the cavity data.
def neg_log_likelihood(params, data):
    f, omega, mu, sigma = params
    # Enforce valid parameter ranges: f in [0,1], omega > 0, sigma > 0.
    if not (0 <= f <= 1 and omega > 0 and sigma > 0):
        return 1e10
    pdf_vals = mixture_pdf(data, params)
    # Avoid log(0) issues.
    if np.any(pdf_vals <= 0):
        return 1e10
    return -np.sum(np.log(pdf_vals))

# Initial guesses for the parameters:
omega_init = np.mean(cavity_data) - 1  # analogous to the vacuum case
f_init = 0.5                           # guess that about half the events are Gaussian
mu_init = np.mean(cavity_data)          # initial guess for Gaussian mean
sigma_init = np.std(cavity_data)         # initial guess for Gaussian standard deviation
init_params = [f_init, omega_init, mu_init, sigma_init]

# Set bounds for the parameters: f in [0,1], omega > 0, sigma > 0, mu is unbounded.
bounds = [(0, 1), (1e-6, None), (None, None), (1e-6, None)]

# Optimize to find the MLE parameters.
result = minimize(neg_log_likelihood, init_params, args=(cavity_data,), bounds=bounds)
if result.success:
    f_hat, omega_hat, mu_hat, sigma_hat = result.x
    print("\nCavity decay mixture model fit:")
    print("Fraction for Gaussian (f) =", f_hat)
    print("Exponential decay constant (ω) =", omega_hat)
    print("Gaussian mean (µ) =", mu_hat)
    print("Gaussian standard deviation (σ) =", sigma_hat)
    print("Log-likelihood =", -result.fun)
else:
    print("Mixture model fit did not converge.")

# Plot the cavity data histogram and the fitted mixture model.
plt.figure(figsize=(8,5))
plt.hist(cavity_data, bins=50, density=True, alpha=0.6, label='Cavity data')
x_plot = np.linspace(1, cavity_data.max(), 200)
plt.plot(x_plot, mixture_pdf(x_plot, result.x), 'r-', lw=2, label='Mixture fit')
plt.plot(x_plot, (1-f_hat)*exp_pdf(x_plot, omega_hat), 'g--', lw=2, label='Exponential component')
plt.plot(x_plot, f_hat*gauss_pdf(x_plot, mu_hat, sigma_hat), 'b--', lw=2, label='Gaussian component')
plt.xlabel('Decay distance x')
plt.ylabel('Probability density')
plt.legend()
plt.title('Cavity Decay: Data and Mixture Model Fit')
plt.show()
plt.savefig("Cavity_decay.png")

# ------------------------
# Fisher Information (Numerical Hessian)
# ------------------------

def numerical_hessian(func, params, data, epsilon=1e-5):
    n = len(params)
    hessian = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            params_ijpp = np.array(params, copy=True)
            params_ijpm = np.array(params, copy=True)
            params_ijmp = np.array(params, copy=True)
            params_ijmm = np.array(params, copy=True)
            params_ijpp[i] += epsilon; params_ijpp[j] += epsilon
            params_ijpm[i] += epsilon; params_ijpm[j] -= epsilon
            params_ijmp[i] -= epsilon; params_ijmp[j] += epsilon
            params_ijmm[i] -= epsilon; params_ijmm[j] -= epsilon
            f_pp = func(params_ijpp, data)
            f_pm = func(params_ijpm, data)
            f_mp = func(params_ijmp, data)
            f_mm = func(params_ijmm, data)
            hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon**2)
    return hessian

hess = numerical_hessian(neg_log_likelihood, result.x, cavity_data)
print("\nApproximate Fisher Information Matrix (via Hessian of -log L):")
print(hess)

# ------------------------
# Null hypothesis test
# ------------------------
# Null hypothesis: the cavity data follow only the exponential model (f = 0).
def neg_log_likelihood_exp(omega, data):
    if omega <= 0:
        return 1e10
    pdf_vals = exp_pdf(data, omega)
    if np.any(pdf_vals <= 0):
        return 1e10
    return -np.sum(np.log(pdf_vals))

omega_null = np.mean(cavity_data) - 1  # MLE under the null
ll_null = -neg_log_likelihood_exp(omega_null, cavity_data)
ll_mix = -result.fun

# Likelihood ratio statistic with 3 extra parameters (f, µ, σ)
LR = 2 * (ll_mix - ll_null)
p_value = 1 - chi2.cdf(LR, df=3)
print("\nNull hypothesis test (pure exponential vs. mixture model):")
print("Exponential-only log-likelihood =", ll_null)
print("Mixture model log-likelihood =", ll_mix)
print("Likelihood ratio statistic =", LR)
print("p-value =", p_value)
if p_value < 0.05:
    print("Reject the null hypothesis: there is significant evidence for an additional decay contribution.")
else:
    print("Do not reject the null hypothesis: no significant evidence for an additional decay contribution.")


