import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import csv

# --- Helper Function ---
def safe_log10(x):
    return np.log10(np.maximum(x, 1e-10))

# --- Experimental Data ---
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

# --- Model ---
def base_model(Y, t, lam, beta, delta, p, c, r, roh):
    T, R, I, V = Y
    dTdt = (lam * T) - (beta * T * V) - (r * I * T) + (roh * R)
    dRdt = (r * I * T) - (roh * R)
    dIdt = (beta * T * V) - (delta * I)
    dVdt = (p * I) - (c * V)
    return [dTdt, dRdt, dIdt, dVdt]

lam_fixed = 0.0645

# --- Simulation ---
def simulate_model(params):
    beta, delta, p, c, r, roh = params
    y0 = [114.2, 0, 0, 0]

    t1 = [0, 1]
    y1 = odeint(base_model, y0, t1, args=(lam_fixed, beta, delta, p, c, r, roh))

    y2_init = y1[-1]; y2_init[3] += 100
    t2 = [1, 2, 4]
    y2 = odeint(base_model, y2_init, t2, args=(lam_fixed, beta, delta, p, c, r, roh))

    y3_init = y2[-1]; y3_init[3] += 100
    t3 = [4, 5, 7]
    y3 = odeint(base_model, y3_init, t3, args=(lam_fixed, beta, delta, p, c, r, roh))

    y4_init = y3[-1]; y4_init[3] += 100
    t4 = np.linspace(7, 42, 200)
    y4 = odeint(base_model, y4_init, t4, args=(lam_fixed, beta, delta, p, c, r, roh))

    all_times = np.concatenate([t1, t2, t3, t4])
    total_tumor = (
        np.concatenate([y1[:,0], y2[:,0], y3[:,0], y4[:,0]]) +
        np.concatenate([y1[:,1], y2[:,1], y3[:,1], y4[:,1]]) +
        np.concatenate([y1[:,2], y2[:,2], y3[:,2], y4[:,2]])
    )
    return all_times, total_tumor

# --- Flatten experimental data ---
exp_T = np.concatenate([T[np.isfinite(T)] for T in T_data_list])
exp_t = np.repeat(t_data[:len(T_data_list)], [np.sum(np.isfinite(T)) for T in T_data_list])

# --- Objective ---
def objective(params, t_exp, T_exp):
    all_times, model_tumor = simulate_model(params)
    model_interp = np.interp(t_exp, all_times, model_tumor)
    return np.sum((model_interp - T_exp)**2)

# --- Initial guesses and bounds ---
initial_guess = [7.95187498e-07, 3.01647123e+00, 1.90261237e+03, 1.59973419e-01, 2.50125194e+01, 4.50824870e-01]

bounds = [
    (1e-11, 1e-3),    # β: infection rate (wider on both ends)
    (0.01, 50.0),      # δ: infected-cell death rate
    (1.0, 5e10),       # p: viral production rate
    (0.05, 50.0),     # c: viral clearance rate
    (0.001, 100000.0),     # r: resistance formation
    (1e-6, 150.0)       #(roh): resistant reversion
]

# ASCII-safe names (fixes Unicode error)
param_names = ["beta", "delta", "p", "c", "r", "roh"]

# --- CSV SETUP (UTF-8, no more UnicodeEncodeError) ---
csv_filename = "wtad5-parameter_results.csv"
with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["type", *param_names, "SSR"])

# --- ORIGINAL FIT ---
res = minimize(objective, initial_guess, args=(exp_t, exp_T),
               bounds=bounds)
best_params = res.x

print("\nOriginal Fit Parameters:")
for n, v in zip(param_names, best_params):
    print(f"{n} = {v:.4e}")
print("SSR:", res.fun)

# Save original fit to CSV
with open(csv_filename, mode="a", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["original", *best_params, res.fun])

# --- Residuals ---
all_times, model_tumor = simulate_model(best_params)
model_interp = np.interp(exp_t, all_times, model_tumor)
residuals = exp_T - model_interp

# --- BOOTSTRAP ---
num_boot = 1000
bootstrap_estimates = np.zeros((num_boot, len(best_params)))

for i in range(num_boot):
    shuffled_residuals = np.random.permutation(residuals)
    T_surrogate = model_interp + shuffled_residuals

    res_boot = minimize(objective, initial_guess, args=(exp_t, T_surrogate),
                        bounds=bounds, method="Nelder-Mead")
    bootstrap_estimates[i, :] = res_boot.x

    print(f"Bootstrap {i+1}/{num_boot} done (SSR={res_boot.fun:.3e})")

    # Save each bootstrap iteration
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
plt.figure(figsize=(12, 6))
for i, name in enumerate(param_names):
    plt.subplot(2, 3, i+1)
    plt.hist(bootstrap_estimates[:, i], bins=10, color='skyblue', edgecolor='black', density=True)
    low, high = bootstrap_ci(bootstrap_estimates[:, i])
    plt.axvline(low, color='red', linestyle='--')
    plt.axvline(high, color='red', linestyle='--')
    plt.title(f"{name} (95% CI)")
plt.tight_layout()
plt.show()

# --- Plot Model Fit ---
plt.figure(figsize=(10,6))
plt.plot(all_times, model_tumor, 'b-', lw=2, label="Best Fit")
plt.scatter(exp_t, exp_T, s=50, alpha=0.6, label="Experimental Data")
plt.xlabel("Time (days)")
plt.ylabel("Tumor Size (mm³)")
plt.title("Tumor Growth Fit with Bootstrap Confidence Intervals")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()