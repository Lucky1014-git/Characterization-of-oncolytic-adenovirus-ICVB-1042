import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import csv

# --- Helper Function ---
def safe_log10(x):
    return np.log10(np.maximum(x, 1e-10))

# --- Experimental Data (real days post-implant: 24-66) ---
t_data = np.array([24, 26, 28, 31, 33, 35, 38, 40, 42, 45, 47, 49, 52, 54, 56, 59, 61, 63, 66])
T_data_list = [
    np.array([101,104,126,118,122]),
    np.array([109,113,137,126,133]),
    np.array([117,121,144,134,138]),
    np.array([130,138,160,148,149]),
    np.array([141,151,173,159,166]),
    np.array([157,167,221,186,215]),
    np.array([172,181,242,205,232]),
    np.array([183,193,257,220,249]),
    np.array([192,199,271,235,264]),
    np.array([303,219,339,332,318]),
    np.array([323,236,359,351,340]),
    np.array([343,251,386,367,358]),
    np.array([336,233,392,306,380]),
    np.array([326,223,404,296,396]),
    np.array([373,194,364,274,322]),
    np.array([319,291,264,261,345]),
    np.array([np.nan,327,np.nan,303,389]),
    np.array([np.nan,360,np.nan,334,418]),
    np.array([np.nan,341,np.nan,305,440])
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

# --- Objective ---
def objective(params, t_exp, T_surrogate):
    β, δ, p, c, γ, roh, α = params
    y0 = [19.958647, 0, 0, 0, 0]
    total_ssr = 0.0
    # Segment 1: 24–26 (no virus yet)
    t1 = [24, 26]
    y1 = odeint(base_model, y0, t1, args=(lam_fixed, β, δ, p, c, γ, roh, α))
    for i in range(len(T_data_list[0])):
        if not np.isnan(T_data_list[0][i]):
            total_ssr += (np.log10(T_data_list[0][i]) - np.log10(y1[0, 0] + y1[0, 1] + y1[0, 2])) ** 2  # t=24

    # Inject virus at t=26
    y2_init = y1[-1].copy()
    y2_init[3] += 100

    # Segment 2: 26–28
    t2 = [26, 26.04167, 27, 28]
    y2 = odeint(base_model, y2_init, t2, args=(lam_fixed, β, δ, p, c, γ, roh, α))

    for i in range(len(T_data_list[1])):
        if not np.isnan(T_data_list[1][i]):
            total_ssr += (np.log10(T_data_list[1][i]) - np.log10(y2[0, 0] + y2[0, 1] + y2[0, 2])) ** 2  # t=26

    for i in range(len(V_data_list[0])):
        if not np.isnan(V_data_list[0][i]):
            total_ssr += (np.log10(y2[1, 3]) - np.log10(V_data_list[0][i])) ** 2  # t=26.04167

    for i in range(len(T_data_list[2])):
        if not np.isnan(T_data_list[2][i]):
            total_ssr += (np.log10(T_data_list[2][i]) - np.log10(y2[3, 0] + y2[3, 1] + y2[3, 2])) ** 2  # t=28

    # Inject virus at t=28
    y3_init = y2[-1].copy()
    y3_init[3] += 100

    # Segment 3: 28–31
    t3 = [28, 31]
    y3 = odeint(base_model, y3_init, t3, args=(λ_fixed, β, δ, p, c, γ, roh, α))

    for i in range(len(V_data_list[1])):
        if not np.isnan(V_data_list[1][i]):
            total_ssr += (np.log10(y3[0, 3]) - np.log10(V_data_list[1][i])) ** 2  # t=28

    for i in range(len(T_data_list[3])):
        if not np.isnan(T_data_list[3][i]):
            total_ssr += (np.log10(T_data_list[3][i]) - np.log10(y3[1, 0] + y3[1, 1] + y3[1, 2])) ** 2  # t=31

    # Inject virus at t=31
    y4_init = y3[-1].copy()
    y4_init[3] += 100

    # Segment 4: 31–66
    t4 = [31, 33, 35, 38, 40, 42, 45, 47, 49, 52, 54, 56, 59, 61, 63, 66]
    y4 = odeint(base_model, y4_init, t4, args=(λ_fixed, β, δ, p, c, γ, roh, α))

    for i, ti in enumerate(t4):
        for j in range(len(T_data_list[i + 3])):
            if not np.isnan(T_data_list[i + 3][j]):
                total_ssr += (np.log10(T_data_list[i + 3][j]) - np.log10(y4[i, 0] + y4[i, 1] + y4[i, 2])) ** 2

    for i in range(len(V_data_list[2])):
        if not np.isnan(V_data_list[2][i]):
            total_ssr += (np.log10(y4[1, 3]) - np.log10(V_data_list[2][i])) ** 2  # t=33

    return total_ssr

initial_guess = [1.3122e-05, 0.05, 5.000e+03, 20.0, 0.001, 0.01, 0.1]

bounds = [
    (1e-8, 1e-3),          # beta
    (0.01564, 5.0),        # delta
    (1000.0, 50000.654),   # p
    (1.0, 500.0),          # c
    (0.00001, 0.5),        # gamma
    (0.001, 0.5),          # roh
    (0.001, 10.5324),      # alpha
]

param_names = ["beta", "delta", "p", "c", "gamma", "roh", "alpha"]

# --- CSV setup ---
csv_filename = "adenoICB1042-bootstrap_results.csv"
with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["type", *param_names, "SSR"])

# --- Original fit ---
#res = minimize(objective, initial_guess, args=(exp_t, exp_T), bounds=bounds)
best_params = [1.1491e-07, 0.0156, 5.000e+04, 1.0029, 0.000, 0.04909, 10.5051]

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

# --- Bootstrap ---
num_boot = 1000
bootstrap_estimates = np.zeros((num_boot, len(best_params)))

for i in range(num_boot):
    shuffled_residuals = np.random.permutation(residuals)
    T_surrogate = model_interp + shuffled_residuals
    #print(T_surrogate)

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