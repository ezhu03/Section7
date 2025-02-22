import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi, sin, asin

# ---------------------------------------------------------------------------
# (a) Symbolic / LaTeX formula for the surface area
# ---------------------------------------------------------------------------
#
# For certain special cases of the ellipsoid, an exact closed-form expression
# can be written in terms of elliptic integrals. One reference form given in
# the problem is (for a spheroid-like shape):
#
#   A = 2 * eps * w^2 * ( 1 + (c / (a * eps)) * sin^{-1}(eps) )
#   where eps = sqrt(1 - w^2/c^2),   or e = 1 - w^2/c^2, etc.
#
# This code snippet just shows a string with the LaTeX expression:
print("=== (a) Symbolic formula above printed. ===")
surface_area_symbolic = r"""
\[
A = 2\,\epsilon \,\omega^{2} 
\Bigl(
1 \;+\; \frac{c}{a\,\epsilon}\;\sin^{-1}(\epsilon)
\Bigr),
\quad
\epsilon = \sqrt{1 - \frac{\omega^{2}}{c^{2}}}.
\]
"""

print("Surface area (symbolic LaTeX form) =")
print(surface_area_symbolic)

# ---------------------------------------------------------------------------
# Helpers: Exact formula and numeric approximations
# ---------------------------------------------------------------------------

def exact_surface_area_spheroid(w, c):
    """
    Returns a 'reference' formula for the surface area of a spheroid-like shape
    if w < c. (Adapt/extend as needed.)
    """
    # e might be imaginary if w > c, so handle cases carefully if needed.
    # We'll assume w < c for demonstration, so e = sqrt(1 - w^2/c^2).
    e = np.sqrt(abs(1.0 - (w**2)/(c**2)))
    
    # If we do not know 'a', assume a=1 for x-axis, as per the problem statement
    a = 1.0
    
    # The formula from the problem text. 
    # Note: This is a typical form for an oblate spheroid if w < c, or prolate if w>c
    # Modify sign or inverse-sin approach depending on shape.
    val = 2.0*e*(w**2)* ( 1.0 + (c/(a*e))*np.arcsin(e) )
    return val

def surface_area_integrand(phi, w, c):
    """
    Example integrand for the surface area if parameterized in a single variable.
    The exact parameterization depends on how you've set up the integral, e.g.:
    
       A = 2 * pi * \int_0^z  r(phi) sqrt(1 + (dr/dphi)^2 ) dphi
    
    or similar.  This is just a placeholder that you may need to adapt.
    """
    # This is NOT a universal integrand; fill in your actual expression.
    # For demonstration, we pretend there's a known function f(phi).
    # 
    # Typically for a spheroid: z = c * sqrt(1 - r^2 / w^2), etc.
    # We'll just put a dummy function below:
    return np.sqrt(1.0 + (w + c)*sin(phi)**2)

def midpoint_rule(f, a, b, n, **kwargs):
    """
    Simple 1D midpoint rule for \int_a^b f(x) dx using n subintervals.
    """
    x_vals = np.linspace(a, b, n+1)
    midpoints = 0.5*(x_vals[:-1] + x_vals[1:])
    h = (b - a)/n
    return h * np.sum([f(x, **kwargs) for x in midpoints])

def gauss_legendre(f, a, b, n, **kwargs):
    """
    Simple Gauss–Legendre integration on [a,b] with n points.
    Uses np.polynomial.legendre.leggauss.
    """
    from numpy.polynomial.legendre import leggauss
    
    # Get Gauss–Legendre points and weights on [-1,1]
    xg, wg = leggauss(n)
    
    # Transform them to [a,b]
    # x in [-1,1] -> t in [a,b],  t = (b-a)/2 * x + (b+a)/2
    t = 0.5*(b - a)*xg + 0.5*(b + a)
    val = 0.0
    for i in range(n):
        val += wg[i] * f(t[i], **kwargs)
    return 0.5*(b - a)*val

# ---------------------------------------------------------------------------
# (b) Deterministic Quadrature Approaches
# ---------------------------------------------------------------------------

def approximate_area_deterministic(w, c, n_mid=50, n_gauss=50):
    """
    Example function that computes the surface area of the ellipsoid
    by a 1D integral approach with midpoint and Gauss–Legendre.
    """
    # Suppose the integral is from phi=0 to phi=pi (just as an example).
    # You may have a different parameter range for your problem.
    a_param = 0.0
    b_param = np.pi
    
    # Midpoint rule
    area_mid = 2.0*pi * midpoint_rule(surface_area_integrand,
                                      a_param, b_param, n_mid,
                                      w=w, c=c)
    # Gaussian quadrature
    area_gauss = 2.0*pi * gauss_legendre(surface_area_integrand,
                                         a_param, b_param, n_gauss,
                                         w=w, c=c)
    return area_mid, area_gauss

def plot_error_heatmap():
    import matplotlib.colors as mcolors
    
    ws = np.logspace(-3, 3, 30)  # e.g. from 0.001 to 1000
    cs = np.logspace(-3, 3, 30)
    
    errors_mid = np.zeros((len(ws), len(cs)))
    errors_gauss = np.zeros((len(ws), len(cs)))
    
    for i, wval in enumerate(ws):
        for j, cval in enumerate(cs):
            # True area (or reference area)
            A_exact = exact_surface_area_spheroid(wval, cval)
            # Numeric approximations
            A_mid, A_gauss = approximate_area_deterministic(wval, cval, n_mid=30, n_gauss=30)
            err_mid = abs(A_mid - A_exact)
            err_gauss = abs(A_gauss - A_exact)
            errors_mid[i, j] = err_mid
            errors_gauss[i, j] = err_gauss
    
    # Plot an example heatmap for midpoint rule
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    # Show errors_mid with log color scale
    plt.imshow(errors_mid, origin='lower', norm=mcolors.LogNorm(),
               extent=[-3,3,-3,3], aspect='auto')
    plt.title("Error (Midpoint Rule)")
    plt.xlabel("log10(c)")
    plt.ylabel("log10(w)")
    plt.colorbar()
    
    # Plot an example heatmap for Gauss–Legendre
    plt.subplot(1,2,2)
    plt.imshow(errors_gauss, origin='lower', norm=mcolors.LogNorm(),
               extent=[-3,3,-3,3], aspect='auto')
    plt.title("Error (Gauss–Legendre)")
    plt.xlabel("log10(c)")
    plt.ylabel("log10(w)")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    plt.savefig("error_heatmap.png")

# ---------------------------------------------------------------------------
# (c) Monte Carlo with Uniform Sampling
# ---------------------------------------------------------------------------
#
# As stated in the problem, "we set 2ω = c = 1" for the demonstration,
# i.e. ω = 0.5, c = 1. We do a simple Monte Carlo to approximate a 2D integral
# that represents the surface area.  This is a skeleton demonstration.

def surface_area_uniform_MC(N, w=0.5, c=1.0):
    """
    Example Monte Carlo approach for a 2D surface integral or a 1D integral
    that represents the surface area, depending on how you parametrize.
    
    For demonstration, we do a 1D integral from phi=0..pi, 
    integral( integrand(phi), dphi ), sampling phi ~ Uniform(0, pi).
    """
    phi_samples = np.random.uniform(low=0.0, high=np.pi, size=N)
    fvals = [surface_area_integrand(phi, w, c) for phi in phi_samples]
    
    # MC integral on [0, pi], average the function values, multiply by the length of the interval
    integral_estimate = (np.pi) * np.mean(fvals)
    
    # Multiply by 2*pi if that is part of the parameterization
    return 2.0*pi * integral_estimate

def run_uniform_MC_trials():
    import matplotlib.pyplot as plt
    
    # For part (c), w=0.5, c=1
    w, c = 0.5, 1.0
    exact_val = exact_surface_area_spheroid(w, c)
    Ns = [10, 100, 1000, 10000, 100000]
    
    errors = []
    for N in Ns:
        approx = surface_area_uniform_MC(N, w, c)
        err = abs(approx - exact_val)
        errors.append(err)
        print(f"N={N}, approx={approx:.6f}, exact={exact_val:.6f}, error={err:.6f}")
    
    # Plot error vs N
    plt.figure()
    plt.loglog(Ns, errors, marker='o')
    plt.xlabel("N (samples)")
    plt.ylabel("Absolute Error")
    plt.title("Uniform MC Error vs. N")
    plt.show()
    plt.savefig("uniform_MC_error.png")

# ---------------------------------------------------------------------------
# (d) Importance Sampling with q1(x)=exp(-3x) and q2(x)=sin^2(5x)
# ---------------------------------------------------------------------------
#
# You would define a new way to draw samples in [0, pi] (or whatever domain),
# and appropriately weight the samples. That is:
#
#   integral(f(x) dx) = E_{x ~ q}[ f(x) * (p(x)/q(x)) ]
#
# For example, if we want uniform p(x)=1/(pi) on [0, pi],
# then p(x)/q(x) is the importance weight.
# The user must figure out how to invert the CDF of q1 and q2, or do
# acceptance/rejection if the CDF is not easily invertible.

def sample_q1_exp3(N):
    """
    Sample from q1(x) = e^{-3x} on x>=0 (we might restrict to [0, pi], or something similar).
    The unnormalized pdf is e^{-3x} for x>=0, 0 otherwise.
    We'll do an inverse transform for x in [0, Xmax], or a truncated approach.
    """
    # For demonstration, let's sample x in [0, L], L>0. If L < pi, we might adjust.
    L = np.pi
    # We want the truncated distribution on [0, L].
    #   Q(x) = (1 - e^{-3x}) / (1 - e^{-3L})
    # so the inverse is x = -1/3 * ln(1 - u(1 - e^{-3L})),  for u in [0,1].
    u = np.random.rand(N)
    norm_factor = 1 - np.exp(-3*L)
    x = -1.0/3.0 * np.log(1 - u*norm_factor)
    return x

def weight_q1_exp3(x):
    """
    Weight = p(x)/q1(x).
    Suppose p is Uniform(0, pi), i.e. p(x) = 1/pi for x in [0, pi].
    q1 is proportional to exp(-3x), truncated to [0, pi].
    So q1(x) = (3 e^{-3x}) / (1 - e^{-3 pi}), for 0<=x<=pi.
    Return ratio p(x)/q1(x).
    """
    L = np.pi
    p_val = 1.0/L  # uniform over [0, pi]
    # q1(x) properly normalized on [0, pi]
    q_val = (3.0*np.exp(-3.0*x)) / (1.0 - np.exp(-3.0*L))
    return p_val / q_val

def importance_sampling_MC_q1(N, w=0.5, c=1.0):
    """
    Importance sampling using q1(x)=e^{-3x} truncated to [0, pi].
    """
    xs = sample_q1_exp3(N)
    # Evaluate the function: f(x)
    fvals = [surface_area_integrand(xx, w, c) for xx in xs]
    # Evaluate weights
    wts = [weight_q1_exp3(xx) for xx in xs]
    # Weighted average
    integral_est = np.sum([fvals[i]*wts[i] for i in range(N)]) / N
    return 2.0*pi * integral_est

def run_importance_sampling_comparison():
    import matplotlib.pyplot as plt
    
    w, c = 0.5, 1.0
    exact_val = exact_surface_area_spheroid(w, c)
    Ns = [10, 100, 1000, 10000, 100000]
    
    errors_uniform = []
    errors_q1 = []
    
    for N in Ns:
        # Uniform
        approx_u = surface_area_uniform_MC(N, w, c)
        err_u = abs(approx_u - exact_val)
        errors_uniform.append(err_u)
        
        # q1 = exp(-3x)
        approx_q1 = importance_sampling_MC_q1(N, w, c)
        err_q1 = abs(approx_q1 - exact_val)
        errors_q1.append(err_q1)
        
        print(f"N={N}, uniform_err={err_u:.6f}, q1_err={err_q1:.6f}")
    
    plt.figure()
    plt.loglog(Ns, errors_uniform, 'o-', label="Uniform")
    plt.loglog(Ns, errors_q1, 's-', label="q1=exp(-3x)")
    plt.xlabel("N (samples)")
    plt.ylabel("Absolute Error")
    plt.legend()
    plt.title("Importance Sampling vs. Uniform MC")
    plt.show()
    plt.savefig("uniform_sampling.png")

# (Similarly, you can implement q2(x)=sin^2(5x) with an inverse CDF or another method.)

# ---------------------------------------------------------------------------
# (e) Box–Muller transform for Normal(μ,σ)
# ---------------------------------------------------------------------------

def box_muller(N, mu=0.0, sigma=1.0):
    """
    Generate N samples from a Normal(mu, sigma^2) using the Box–Muller transform.
    """
    # U1, U2 ~ uniform(0,1)
    U1 = np.random.rand(N)
    U2 = np.random.rand(N)
    
    # R = sqrt(-2 ln U1), Theta = 2 pi U2
    R = np.sqrt(-2.0 * np.log(U1))
    Theta = 2.0 * np.pi * U2
    
    Z1 = R * np.cos(Theta)  # standard normal
    Z2 = R * np.sin(Theta)  # standard normal
    
    # Convert to N(mu, sigma^2)
    X1 = mu + sigma * Z1
    X2 = mu + sigma * Z2
    
    return X1, X2  # returns two arrays of length N

def demo_box_muller():
    import matplotlib.pyplot as plt
    
    N = 10000
    mu, sigma = 0.0, 1.0
    X1, X2 = box_muller(N, mu, sigma)
    
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.hist(X1, bins=50, density=True, alpha=0.7, label="X1")
    plt.title("Histogram of X1 ~ N(0,1)")
    plt.subplot(1,2,2)
    plt.hist(X2, bins=50, density=True, alpha=0.7, label="X2")
    plt.title("Histogram of X2 ~ N(0,1)")
    plt.show()
    plt.savefig("box_muller.png")

# ---------------------------------------------------------------------------
# (f) Monte Carlo integration with Gaussian proposals
# ---------------------------------------------------------------------------
#
# The idea is the same as with importance sampling: we choose x ~ N(mu, sigma),
# then weigh properly if needed, or possibly we re-parameterize the integral
# for indefinite bounds.  Details will depend on how you set up the domain for x.

def sample_gaussian_proposal(N, mu=0.0, sigma=1.0):
    """
    Draw N samples from N(mu, sigma) using box_muller or np.random.normal.
    For demonstration, let's just call np.random.normal here.
    """
    return np.random.normal(loc=mu, scale=sigma, size=N)

def gaussian_importance_sampling_MC(N, w=0.5, c=1.0, mu=0.0, sigma=1.0):
    """
    Illustrative function to do MC with a Gaussian proposal q(x)=N(mu, sigma).
    Suppose we want x in [0, pi], but now we have x ~ N(mu, sigma).
    We'll just accept x that fall in [0,pi], or weigh them, etc.

    For demonstration, we do a truncated approach: we only keep samples in [0,pi].
    In real code, you might do a better approach or the full importance-weight approach.
    """
    x_proposal = sample_gaussian_proposal(5*N, mu, sigma)  # oversample
    # keep only those in [0, pi]
    x_keep = x_proposal[(x_proposal>=0) & (x_proposal<=np.pi)]
    # if not enough, re-sample more or limit to first N:
    if len(x_keep) < N:
        x_keep = x_keep[:N]  # naive approach
    else:
        x_keep = x_keep[:N]
    
    # We now treat p(x)=1/pi on [0,pi] (uniform), and q(x) is the truncated normal.
    # So the weight is p(x)/q(x).  Let's define that function:
    def weight_gauss(x):
        # p(x) = 1/pi for x in [0, pi]
        p_val = 1.0/np.pi
        # q(x) = [ normal_pdf(x; mu, sigma ) ] / [ normalization over [0,pi] ]
        # normal_pdf = 1/(sqrt(2 pi)*sigma) exp(-(x-mu)^2/(2 sigma^2))
        denom = 0.5*(1 + erf((np.pi - mu)/(np.sqrt(2)*sigma))) \
                - 0.5*(1 + erf((0 - mu)/(np.sqrt(2)*sigma)))  # cdf(pi) - cdf(0)
        # We'll do an explicit normal pdf:
        normpdf = (1.0/(sigma*np.sqrt(2*pi))) * np.exp(-0.5*((x - mu)/sigma)**2)
        q_val = normpdf / denom
        
        return p_val / q_val
    
    from math import erf
    fvals = [surface_area_integrand(xx, w, c) for xx in x_keep]
    wts = [weight_gauss(xx) for xx in x_keep]
    integral_est = np.sum([fvals[i]*wts[i] for i in range(len(x_keep))]) / len(x_keep)
    return 2.0*pi * integral_est

def run_gaussian_proposal_demo():
    import matplotlib.pyplot as plt
    w, c = 0.5, 1.0
    exact_val = exact_surface_area_spheroid(w, c)
    Ns = [10, 100, 1000, 10000]
    
    errors_gauss = []
    for N in Ns:
        approx = gaussian_importance_sampling_MC(N, w, c, mu=0.0, sigma=1.0)
        err = abs(approx - exact_val)
        errors_gauss.append(err)
        print(f"N={N}, approx={approx:.6f}, exact={exact_val:.6f}, error={err:.6f}")
    
    plt.figure()
    plt.loglog(Ns, errors_gauss, marker='o')
    plt.xlabel("N (samples)")
    plt.ylabel("Abs Error (Gaussian prop.)")
    plt.title("Gaussian-proposal MC vs. N")
    plt.show()
    plt.savefig("gaussian_proposal.png")
    
    # Try different mu, sigma for fixed N
    test_mus = [0.0, 1.0, 2.0]
    test_sigs = [0.5, 1.0, 2.0]
    N_fixed = 10000
    
    for mval in test_mus:
        for sval in test_sigs:
            approx = gaussian_importance_sampling_MC(N_fixed, w, c, mu=mval, sigma=sval)
            err = abs(approx - exact_val)
            print(f"mu={mval}, sigma={sval}, approx={approx:.6f}, error={err:.6f}")

# ---------------------------------------------------------------------------
# Putting it all together (example main)
# ---------------------------------------------------------------------------

def main():

    print("\n=== (b) Deterministic Quadrature Example (Single w,c) ===")
    w_demo, c_demo = 0.5, 1.0
    A_exact = exact_surface_area_spheroid(w_demo, c_demo)
    A_mid, A_gauss = approximate_area_deterministic(w_demo, c_demo)
    print(f"Exact area ~ {A_exact:.6f}")
    print(f"Midpoint rule ~ {A_mid:.6f} (error={abs(A_mid - A_exact):.6g})")
    print(f"Gauss-Legendre ~ {A_gauss:.6f} (error={abs(A_gauss - A_exact):.6g})")

    print("\nPlotting error heatmaps for w,c in [0.001..1000] might be slow,")
    print("so comment/uncomment the call if desired.")
    # plot_error_heatmap()

    print("\n=== (c) Monte Carlo with Uniform Sampling ===")
    run_uniform_MC_trials()

    print("\n=== (d) Importance Sampling (q1=exp(-3x)) ===")
    run_importance_sampling_comparison()
    # similarly implement q2(x)=sin^2(5x) if desired

    print("\n=== (e) Box–Muller test ===")
    demo_box_muller()

    print("\n=== (f) Gaussian proposals for MC ===")
    run_gaussian_proposal_demo()

if __name__ == "__main__":
    main()
