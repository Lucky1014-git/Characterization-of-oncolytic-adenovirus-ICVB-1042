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
    (1e-6, 100.0)       #(roh): resistant reversion
]


# --- Optimization ---
res = minimize(objective, initial_guess, bounds=bounds, method="Nelder-Mead")
best_params = res.x
print("Best-fit parameters (β, δ, p, c, r, roh):")
print(best_params)
print("Final MSE:", res.fun)

all_times, total_tumor = simulate_model(best_params)

# --- Plot results ---
plt.figure(figsize=(10,6))
plt.plot(all_times, total_tumor, 'b-', linewidth=2, label='Model Fit')

for i in range(len(t_data)):
    if i < len(T_data_list):
        mask = ~np.isnan(T_data_list[i])
        plt.scatter(np.full(np.sum(mask), t_data[i]), T_data_list[i][mask],
                    s=50, alpha=0.6, color = 'red', label='Experimental Data' if i == 0 else "")

plt.title("Tumor Growth Fit for ICVB-1042", fontsize = 30)
plt.ylabel("Tumor Size (mm³)", fontsize = 30)
plt.xlabel("Time (days)", fontsize = 30)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
