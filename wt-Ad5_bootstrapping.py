import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import csv

# --- Helper Function ---
def safe_log10(x):
    return np.log10(np.maximum(x, 1e-10))

# --- Experimental Data (real days post-implant: 24-66) ---
t_data = np.array([0, 2, 4, 7, 9, 11, 14, 16, 18, 21, 23, 25, 28, 30, 32, 35, 37, 39, 42])
T_data_list = [
    np.array([102,125,110,128,139]),
    np.array([107,135,119,135,148]),
    np.array([116,143,124,143,157]),
    np.array([125,156,133,159,170]),
    np.array([137,169,144,173,187]),
    np.array([149,201,188,211,206]),
    np.array([167,222,208,237,223]),
    np.array([177,238,223,256,237]),
    np.array([187,251,233,272,250]),
    np.array([291,336,354,405,321]),
    np.array([367,469,451,471,382]),
    np.array([386,500,477,492,411]),
    np.array([441, np.nan, 562, np.nan, 425]),
    np.array([460, np.nan, 587, np.nan, 443]),
    np.array([605, np.nan, 679, np.nan, 518]),
    np.array([733, np.nan, np.nan, np.nan, 570]),
    np.array([846, np.nan, np.nan, np.nan, 617]),
    np.array([922, np.nan, np.nan, np.nan, 661]),
    np.array([np.nan, np.nan, np.nan, np.nan, 763])
]
V_data_list = [
    np.array([1.372643, 0.018196, 1.158547, 99.04117, 26509.89]),
    np.array([0.01, 0.012721, 0.01, 585.1756, 6010.401]),
    np.array([np.nan, 0.01, 0.01, 0.275531, 92476.8])
]
# --- Model: T, R, I, V, F ---
def base_model(Y, t, lam, beta, delta, p, c, gamma, roh, alpha):
    T, R, I, V, F = Y
    dTdt = (lam * T) - (beta * T * V) - (gamma * F * T) + (roh * R)
    dRdt = (gamma * F * T) - (roh * R)
    dIdt = (beta * T * V) - (delta * I)
    dVdt = (p * I) - (c * V)
    dFdt = V - (alpha * F)
    return [dTdt, dRdt, dIdt, dVdt, dFdt]

lam_fixed = 0.064425

# --- Simulation: segments now run on the REAL day clock (24-66), ---
# --- with virus injections at the real injection days (26, 28, 31). ---
def simulate_model(params):
    beta, delta, p, c, gamma, roh, alpha = params
    y0 = [19.958647, 0, 0, 0, 0]  # state at t=24 (implant day)

    t1 = [24, 26]
    y1 = odeint(base_model, y0, t1, args=(lam_fixed, beta, delta, p, c, gamma, roh, alpha))

    # Inject virus at t=26
    y2_init = y1[-1].copy(); y2_init[3] += 100
    t2 = np.sort(np.unique(np.concatenate([np.linspace(26, 28, 25), [26, 28]])))
    y2 = odeint(base_model, y2_init, t2, args=(lam_fixed, beta, delta, p, c, gamma, roh, alpha))

    # Inject virus at t=28
    y3_init = y2[-1].copy(); y3_init[3] += 100
    t3 = np.sort(np.unique(np.concatenate([np.linspace(28, 31, 25), [28, 31]])))
    y3 = odeint(base_model, y3_init, t3, args=(lam_fixed, beta, delta, p, c, gamma, roh, alpha))

    # Inject virus at t=31
    y4_init = y3[-1].copy(); y4_init[3] += 100
    t4 = np.sort(np.unique(np.concatenate([np.linspace(31, 66, 200), t_data[t_data >= 31]])))
    y4 = odeint(base_model, y4_init, t4, args=(lam_fixed, beta, delta, p, c, gamma, roh, alpha))

    all_times = np.concatenate([t1, t2, t3, t4])
    # Total tumor = susceptible + resistant + infected cells (T + R + I)
    total_tumor = (
        np.concatenate([y1[:,0], y2[:,0], y3[:,0], y4[:,0]]) +
        np.concatenate([y1[:,1], y2[:,1], y3[:,1], y4[:,1]]) +
        np.concatenate([y1[:,2], y2[:,2], y3[:,2], y4[:,2]])
    )
    return all_times, total_tumor

# --- Flatten experimental data ---
exp_T = np.concatenate([T[np.isfinite(T)] for T in T_data_list])
exp_t = np.repeat(t_data[:len(T_data_list)], [np.sum(np.isfinite(T)) for T in T_data_list])

# --- Objective for fitting to observed tumor data ---
def objective(params, t_exp, T_obs):
    beta, delta, p, c, gamma, roh, alpha = params
    sim_times, sim_tumor = simulate_model((beta, delta, p, c, gamma, roh, alpha))
    sim_interp = np.interp(t_exp, sim_times, sim_tumor)

    obs = np.maximum(T_obs, 1e-10)
    pred = np.maximum(sim_interp, 1e-10)
    return np.sum((np.log10(obs) - np.log10(pred)) ** 2)

initial_guess = [4e-5, 0.012, 1005.78, 20.0, 0.001, 0.01, 0.1]

bounds = [
    (1e-8, 1e-2),
    (0.01, 5.0),
    (156.0, 100000.0),
    (1.0, 500.0),
    (0.0002554, 1.0),
    (0.0001, 10.0),
    (0.001, 10.5324)
]
param_names = ["beta", "delta", "p", "c", "gamma", "roh", "alpha"]

# --- CSV setup ---
csv_filename = "wtAd5-bootstrap_results.csv"
with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["type", *param_names, "SSR"])

# --- Original fit ---
#res = minimize(objective, initial_guess, args=(exp_t, exp_T), bounds=bounds)
best_params = [2.1924e-06, 0.0107, 8.7034e+04, 54.2611, 0.0003, 4.5419, 9.5663]

print("\nOriginal Fit Parameters:")
for n, v in zip(param_names, best_params):
    print(f"{n} = {v:.4e}")
#print("SSR:", )

#with open(csv_filename, mode="a", newline="", encoding="utf-8") as file:
    #writer = csv.writer(file)
    #writer.writerow(["original", *best_params, res.fun])

# --- Residuals ---
all_times, model_tumor = simulate_model(best_params)
model_interp = np.interp(exp_t, all_times, model_tumor)
residuals = exp_T - model_interp

# --- Bootstrap: residual bootstrap, not random permutation of the data
num_boot = 1000
bootstrap_estimates = np.zeros((num_boot, len(best_params)))

for i in range(num_boot):
    sampled_residuals = np.random.choice(residuals, size=len(residuals), replace=True)
    T_surrogate = model_interp + sampled_residuals

    res_boot = minimize(objective, best_params, args=(exp_t, T_surrogate),
                        bounds=bounds, method="Nelder-Mead")
    bootstrap_estimates[i, :] = res_boot.x

    print(f"Bootstrap {i+1}/{num_boot} done (SSR={res_boot.fun:.3e})")

    with open(csv_filename, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([f"bootstrap_{i+1}", *res_boot.x, res_boot.fun])

# --- Confidence Intervals ---
def bootstrap_ci(values, alpha=0.05):
    sorted_vals = np.sort(values)
    lower = np.percentile(sorted_vals, 100*alpha/2)
    upper = np.percentile(sorted_vals, 100*(1-alpha/2))
    return lower, upper

print("\n95% Confidence Intervals:")
for i, name in enumerate(param_names):
    low, high = bootstrap_ci(bootstrap_estimates[:, i])
    print(f"{name}: [{low:.4e}, {high:.4e}]")

# --- Plot Histograms ---
plt.figure(figsize=(12, 8))
for i, name in enumerate(param_names):
    plt.subplot(3, 3, i+1)
    plt.hist(bootstrap_estimates[:, i], bins=10, color='skyblue', edgecolor='black', density=True)
    low, high = bootstrap_ci(bootstrap_estimates[:, i])
    plt.axvline(low, color='red', linestyle='--')
    plt.axvline(high, color='red', linestyle='--')
    plt.title(f"{name} (95% CI)")
plt.tight_layout()
plt.savefig("bootstrap_histograms.png", dpi=150)
plt.show()

# --- Plot Model Fit ---
plt.figure(figsize=(10,6))
plt.plot(all_times, model_tumor, 'b-', lw=2, label="Best Fit")
plt.scatter(exp_t, exp_T, s=50, alpha=0.6, label="Experimental Data")
plt.xlabel("Time (days post-implant)")
plt.ylabel("Tumor Size (mm3)")
plt.title("Tumor Growth Fit with Bootstrap Confidence Intervals")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("model_fit.png", dpi=150)
plt.show()