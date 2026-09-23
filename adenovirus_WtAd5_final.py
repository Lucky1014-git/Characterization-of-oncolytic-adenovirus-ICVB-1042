import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

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
def base_model(Y, t, λ, β, δ, p, c, r, roh):
    T, R, I, V = Y
    dTdt = (λ * T) - (β * T * V) - (r * I * T) + (roh * R)
    dRdt = (r * I * T) - (roh * R)
    dIdt = (β * T * V) - (δ * I)
    dVdt = (p * I) - (c * V)
    return [dTdt, dRdt, dIdt, dVdt]

# --- Fixed parameter ---
λ_fixed = 0.0645

# --- Objective function ---
def simulate_model(params):
    β, δ, p, c, r, roh = params

    y0 = [114.2, 0, 0, 0]

    # Segments with injections (same as before)
    t1 = [0, 1]
    y1 = odeint(base_model, y0, t1, args=(λ_fixed, β, δ, p, c, r, roh))

    y2_init = y1[-1]; y2_init[3] += 100
    t2 = [1, 2, 4]
    y2 = odeint(base_model, y2_init, t2, args=(λ_fixed, β, δ, p, c, r, roh))

    y3_init = y2[-1]; y3_init[3] += 100
    t3 = [4, 5, 7]
    y3 = odeint(base_model, y3_init, t3, args=(λ_fixed, β, δ, p, c, r, roh))

    y4_init = y3[-1]; y4_init[3] += 100
    t4 = np.linspace(7, 42, 200)
    y4 = odeint(base_model, y4_init, t4, args=(λ_fixed, β, δ, p, c, r, roh))

    all_times = np.concatenate([t1, t2, t3, t4])
    all_T = np.concatenate([y1[:,0], y2[:,0], y3[:,0], y4[:,0]])
    all_R = np.concatenate([y1[:,1], y2[:,1], y3[:,1], y4[:,1]])
    all_I = np.concatenate([y1[:,2], y2[:,2], y3[:,2], y4[:,2]])

    total_tumor = all_T + all_R + all_I
    return all_times, total_tumor

def objective(params):
    all_times, model_tumor = simulate_model(params)

    # Flatten experimental data
    exp_T = np.concatenate([T[np.isfinite(T)] for T in T_data_list])
    exp_t = np.repeat(t_data[:len(T_data_list)], [np.sum(np.isfinite(T)) for T in T_data_list])

    # Interpolate model at experimental times
    model_interp = np.interp(exp_t, all_times, model_tumor)

    # Mean squared error
    mse = np.mean((model_interp - exp_T)**2)
    return mse

# --- Initial guesses and bounds ---
initial_guess = [1e-8, 0.5, 1e5, 1.0, 0.1, 0.1]

bounds = [
    (1e-11, 1e-5),    # β: infection rate (wider on both ends)
    (0.01, 5.0),      # δ: infected-cell death rate
    (1.0, 5e7),       # p: viral production rate
    (0.05, 20.0),     # c: viral clearance rate
    (0.001, 1000.0),     # r: resistance formation
    (1e-6, 1.0)       #(roh): resistant reversion
]



# --- Optimization ---
res = minimize(objective, initial_guess, bounds=bounds, method="Nelder-Mead")
best_params = res.x
print("Best-fit parameters (β, δ, p, c, r, roh):")
print(best_params)
print("Final MSE:", res.fun)

# --- Simulate with best parameters ---
all_times, total_tumor = simulate_model(best_params)

# --- Plot results ---
plt.figure(figsize=(10,6))
plt.plot(all_times, total_tumor, 'b-', linewidth=2, label='Model Fit')

for i in range(len(t_data)):
    if i < len(T_data_list):
        mask = ~np.isnan(T_data_list[i])
        plt.scatter(np.full(np.sum(mask), t_data[i]), T_data_list[i][mask],
                    s=50, alpha=0.6, color='red', label='Experimental Data' if i == 0 else "")

plt.title("Tumor Growth Fit for WtAd5", fontsize=30)
plt.ylabel("Tumor Size (mm³)", fontsize=30)
plt.xlabel("Time (days)", fontsize=30)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
