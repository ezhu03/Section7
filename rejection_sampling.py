import numpy as np
import matplotlib.pyplot as plt

# Define the (unnormalized) target PDF: p(t) = exp(-4*t)*cos^2(4*t)
def target(t, a=4, b=4):
    return np.exp(-b * t) * np.cos(a * t)**2

# (a) Rejection sampling using a uniform proposal U(0, tf)
def rejection_sampling_uniform(N, tf=1.2, a=4, b=4):
    accepted = []
    total_trials = 0
    # For a uniform proposal, q(t)=1/tf. Since max{p(t)} = 1 at t=0, we need M = tf.
    M = tf
    while len(accepted) < N:
        t = np.random.uniform(0, tf)
        u = np.random.uniform(0, 1)
        # Accept with probability p(t) / (M*q(t)) = p(t) / (tf*(1/tf)) = p(t)
        if u < target(t, a, b):
            accepted.append(t)
        total_trials += 1
    rejection_ratio = len(accepted) / (total_trials - len(accepted))
    return np.array(accepted), rejection_ratio

# (b) Rejection sampling using an exponential proposal Exp(1)
def rejection_sampling_exponential(N, a=4, b=4):
    accepted = []
    total_trials = 0
    # For the exponential proposal q(t)=e^(-t), the acceptance probability is
    # p(t)/(M*q(t)) = [e^(-4*t)*cos^2(4*t)]/[M*e^(-t)] = e^(-3*t)*cos^2(4*t).
    # The maximum of e^(-3*t)*cos^2(4*t) is 1 at t=0, so we can take M=1.
    M = 1
    while len(accepted) < N:
        t = np.random.exponential(scale=1.0)
        u = np.random.uniform(0, 1)
        if u < np.exp(-3 * t) * np.cos(4 * t)**2:
            accepted.append(t)
        total_trials += 1
    rejection_ratio = len(accepted) / (total_trials - len(accepted))
    return np.array(accepted), rejection_ratio

# Function to run experiments for various N and plot histograms for both proposals
def plot_histograms():
    Ns = [100, 1000, 10000]
    plt.figure(figsize=(12, 10))
    
    for i, N in enumerate(Ns, 1):
        # Uniform proposal
        samples_uniform, rej_ratio_uniform = rejection_sampling_uniform(N)
        # Exponential proposal
        samples_exponential, rej_ratio_exponential = rejection_sampling_exponential(N)
        
        # Plot for uniform proposal
        plt.subplot(3, 2, 2*i-1)
        plt.hist(samples_uniform, bins=30, density=True, alpha=0.7, color='skyblue')
        plt.title(f"Uniform Proposal (N={N})\nRejection ratio = {rej_ratio_uniform:.3f}")
        plt.xlabel("t")
        plt.ylabel("Density")
        
        # Plot for exponential proposal
        plt.subplot(3, 2, 2*i)
        plt.hist(samples_exponential, bins=30, density=True, alpha=0.7, color='lightgreen')
        plt.title(f"Exponential Proposal (N={N})\nRejection ratio = {rej_ratio_exponential:.3f}")
        plt.xlabel("t")
        plt.ylabel("Density")
    
    plt.tight_layout()
    plt.show()
    plt.savefig("rejection_sampling_histograms.png")

plot_histograms()
