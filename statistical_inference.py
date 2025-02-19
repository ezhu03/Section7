import json
import numpy as np
from scipy.stats import beta  # convenient Beta distribution utilities
import matplotlib.pyplot as plt
import math

with open('dataset_1.json', 'r') as f1:
    data1 = np.array(json.load(f1))   # list of True/False
with open('dataset_2.json', 'r') as f2:
    data2 = np.array(json.load(f2))
with open('dataset_3.json', 'r') as f3:
    data3 = np.array(json.load(f3))

N = 500
M1 = sum(data1)  # number of Heads (True) in dataset_1
M2 = sum(data2)
M3 = sum(data3)

# Dataset 1 posterior parameters:
alpha1 = M1 + 1
beta1  = (N - M1) + 1

# Dataset 2 posterior parameters:
alpha2 = M2 + 1
beta2  = (N - M2) + 1

# Dataset 3 posterior parameters:
alpha3 = M3 + 1
beta3  = (N - M3) + 1

mean1 = alpha1/(alpha1 + beta1)
var1  = (alpha1 * beta1) / ((alpha1 + beta1)**2 * (alpha1 + beta1 + 1))

mean2 = alpha2/(alpha2 + beta2)
var2  = (alpha2 * beta2) / ((alpha2 + beta2)**2 * (alpha2 + beta2 + 1))

mean3 = alpha3/(alpha3 + beta3)
var3  = (alpha3 * beta3) / ((alpha3 + beta3)**2 * (alpha3 + beta3 + 1))

p_grid = np.linspace(0,1,300)

pdf1 = beta.pdf(p_grid, alpha1, beta1)
pdf2 = beta.pdf(p_grid, alpha2, beta2)
pdf3 = beta.pdf(p_grid, alpha3, beta3)

plt.plot(p_grid, pdf1, label='Dataset 1')
plt.plot(p_grid, pdf2, label='Dataset 2')
plt.plot(p_grid, pdf3, label='Dataset 3')
plt.xlabel('p')
plt.ylabel('Posterior density')
plt.legend()
plt.show()
plt.savefig('posterior_density.png')

batch_size = 50
num_batches = N // batch_size  # 10 if N=500

alpha, beta = 1, 1  # start with Beta(1,1)
posterior_means = []
posterior_vars  = []

for i in range(num_batches):
    # get the i-th batch of data
    batch = data1[i*batch_size : (i+1)*batch_size]
    # number of heads in that batch
    M_batch = sum(batch)
    # update posterior
    alpha += M_batch
    beta  += (batch_size - M_batch)

    # store the posterior stats after this batch
    post_mean = alpha / (alpha + beta)
    post_var  = (alpha*beta) / ((alpha+beta)**2 * (alpha+beta+1))
    posterior_means.append(post_mean)
    posterior_vars.append(post_var)

p_hat = M_batch / float(batch_size)
fisher_var = p_hat * (1 - p_hat) / batch_size
print("Fisher information: ", 1/fisher_var)

# Define Stirling’s approximation for log(n!)
#   log(n!) ~ n log n - n + 0.5 log(2 pi n).
def stirling_log_factorial(n):
    if n <= 0:
        return float('nan')  # not defined for n=0 or negative
    return n*math.log(n) - n + 0.5*math.log(2.0*math.pi*n)

# We'll compare for n=1 through 10
n_values = np.arange(1, 11)

# Exact log(n!) can be obtained from math.lgamma(n+1) because
#   Gamma(n+1) = n!, so log(Gamma(n+1)) = log(n!)
log_factorial_exact = [math.lgamma(n+1) for n in n_values]

# Stirling approximation
log_factorial_stirling = [stirling_log_factorial(n) for n in n_values]

# Compute the difference: stirl - exact
differences = [s - e for (s,e) in zip(log_factorial_stirling, log_factorial_exact)]

# Make the figure
fig, axs = plt.subplots(2, 1, figsize=(6,8))

# ---- (1) Top subplot: log(n!) vs. n  ----
# Scatter the exact values:
axs[0].scatter(n_values, log_factorial_exact, color='k', label='Exact (log Γ(n+1))')

# Overplot a smooth curve for Stirling on the same n-values:
axs[0].plot(n_values, log_factorial_stirling, 'r-o', label='Stirling Approx')

axs[0].set_xlabel('n')
axs[0].set_ylabel('log(n!)')
axs[0].set_title('Exact log(n!) vs. Stirling Approx')
axs[0].legend()

# ---- (2) Bottom subplot: difference  ----
axs[1].plot(n_values, differences, 'b-o')
axs[1].axhline(0, color='k', linestyle='--')
axs[1].set_xlabel('n')
axs[1].set_ylabel('Stirling - Exact')
axs[1].set_title('Difference between Stirling and log Γ(n+1)')

plt.tight_layout()
plt.show()
plt.savefig('stirling_approx.png')

datasets = [data1, data2, data3]
dataset_names = ['Dataset 1', 'Dataset 2', 'Dataset 3']

# These are the sample sizes you want to test:
sample_sizes = [5, 15, 40, 60, 90, 150, 210, 300, 400]
n_bootstrap = 100  # number of bootstrap samples per sample size

figures = []
all_bootstrap_means = []    # will store the mean from each sample size
all_bootstrap_variances = [] # will store the variance from each sample size

for d_i, data_arr in enumerate(datasets):
    # Compute the overall fraction of heads for reference
    overall_mean = data_arr.mean()  # M/N for that dataset
    N = len(data_arr)               # 500

    # Prepare a 3x3 figure for the 9 sample sizes
    fig, axs = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle(f'Bootstrapping {dataset_names[d_i]}', fontsize=16)

    bootstrap_means_for_this_dataset = []
    bootstrap_vars_for_this_dataset = []

    for i, s in enumerate(sample_sizes):
        row = i // 3
        col = i % 3

        # Collect bootstrap estimates
        bootstrap_estimates = []
        for b in range(n_bootstrap):
            # draw a bootstrap sample of size s (with replacement)
            sample = np.random.choice(data_arr, size=s, replace=True)
            # store the fraction of heads in that sample
            bootstrap_estimates.append(sample.mean())

        # Plot histogram in the appropriate subplot
        axs[row, col].hist(bootstrap_estimates, bins=10, alpha=0.7)
        axs[row, col].axvline(overall_mean, color='k', linestyle='--',
                              label='Overall mean')
        axs[row, col].set_title(f'Sample size = {s}')

        # Optionally add legend
        if i == 0:
            axs[row, col].legend()

        # Calculate mean, variance of bootstrap distribution
        mean_b = np.mean(bootstrap_estimates)
        var_b  = np.var(bootstrap_estimates, ddof=1)  # sample variance

        bootstrap_means_for_this_dataset.append(mean_b)
        bootstrap_vars_for_this_dataset.append(var_b)

    plt.tight_layout()
    plt.show()
    
    # Store means/vars for later comparison
    all_bootstrap_means.append(bootstrap_means_for_this_dataset)
    all_bootstrap_variances.append(bootstrap_vars_for_this_dataset)


# -------------------------
# Compare bootstrap means/variances with the “true” estimate from part (b)
# in a separate figure, for example.

for d_i, data_arr in enumerate(datasets):
    plt.figure(figsize=(10, 4))
    # The "true" or reference estimate could be the overall fraction M/N
    overall_mean = data_arr.mean()
    
    # Plot the bootstrap means across sample sizes
    plt.subplot(1,2,1)
    plt.plot(sample_sizes, all_bootstrap_means[d_i], 'o--', label='Bootstrap Means')
    plt.axhline(overall_mean, color='k', linestyle='--',
                label=f'Overall Mean = {overall_mean:.3f}')
    plt.xlabel('Sample size')
    plt.ylabel('Mean of bootstrap distribution')
    plt.title(f'{dataset_names[d_i]}: Bootstrap Means vs. Sample Size')
    plt.legend()

    # Plot the bootstrap variances across sample sizes
    plt.subplot(1,2,2)
    plt.plot(sample_sizes, all_bootstrap_variances[d_i], 'o--', color='red')
    plt.xlabel('Sample size')
    plt.ylabel('Variance of bootstrap distribution')
    plt.title(f'{dataset_names[d_i]}: Bootstrap Variances vs. Sample Size')

    plt.tight_layout()
    plt.show()
    plt.savefig(f'bootstrap_{d_i}.png')


