import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def safe_log10(x):
    return np.log10(np.maximum(x, 1e-10))

λ_fixed = 0.064425

t_data = np.array([24, 26, 28, 31, 33, 35, 38, 40, 42, 45, 47, 49, 52, 54, 56, 59, 61, 63, 66])
T_data_list = [
    np.array([101,104,126,118,122]),
    np.array([109,113,137,126,133]),
    np.array([117, 121, 144, 134, 138]),
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
    np.array([np.nan,327, np.nan,303,389]),
    np.array([np.nan, 360, np.nan,334,418]),
    np.array([np.nan, 341, np.nan, 305, 440])
]
v_data = np.array([0.04167, 1, 3, 7, 28])
V_data_list = [
    np.array([133.8419,0.01,223.9923,14025.97,1278666]),
    np.array([1117.786,0.147289,383.6778,1134.211,784866.8]),
    np.array([397.7273,0.01,229.1866,202.887,407074])
]

def base_model(Y, t, λ, β, δ, p, c, γ, roh, α):
    T, R, I, V, F = Y
    dTdt = (λ * T) - (β * T * V) - (γ * F * T) + (roh * R)
    dRdt = (γ * F * T) - (roh * R)
    dIdt = (β * T * V) - (δ * I)
    dVdt = (p * I) - (c * V)
    dFdt = V - (α * F)
    return [dTdt, dRdt, dIdt, dVdt, dFdt]

def ssr(params):
    β, δ, p, c, γ, roh, α = params
    y0 = [19.958647, 0, 0, 0, 0]
    total_ssr = 0.0
    # Segment 1: 24–26 (no virus yet)
    t1 = [24, 26]
    y1 = odeint(base_model, y0, t1, args=(λ_fixed, β, δ, p, c, γ, roh, α))
    for i in range(len(T_data_list[0])):
        if not np.isnan(T_data_list[0][i]):
            total_ssr += (np.log10(T_data_list[0][i]) - np.log10(y1[0, 0] + y1[0, 1] + y1[0, 2])) ** 2  # t=24

    # Inject virus at t=26
    y2_init = y1[-1].copy()
    y2_init[3] += 100

    # Segment 2: 26–28
    t2 = [26, 26.04167, 27, 28]
    y2 = odeint(base_model, y2_init, t2, args=(λ_fixed, β, δ, p, c, γ, roh, α))

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

# params order: β, δ, p, c, γ, roh, α
initial_guess = [1.3122e-05, 0.05, 5.000e+03, 20.0, 0.001, 0.01, 0.1]

bounds = [
    (1e-8, 1e-3),        # β
    (0.01564, 5.0),          # δ
    (1000.0, 50000.654),    # p
    (1.0, 500.0),         # c
    (0.00001, 0.5),       # γ
    (0.001, 0.5),         # roh
    (0.001, 10.5324),        # α
]
result = minimize(ssr, initial_guess, bounds=bounds, method='Nelder-Mead')

β_fit, δ_fit, p_fit, c_fit, γ_fit, roh_fit, α_fit = result.x
print(f"Fitted β: {β_fit:.4e}")
print(f"Fitted δ: {δ_fit:.4f}")
print(f"Fitted p: {p_fit:.4e}")
print(f"Fitted c: {c_fit:.4f}")
print(f"Fitted γ: {γ_fit:.4f}")
print(f"Fitted roh: {roh_fit:.4f}")
print(f"Fitted α: {α_fit:.4f}")
print(f"SSR: {result.fun:.4f}")

# --- Simulation ---

t_segments = [
    np.sort(np.unique(np.concatenate([np.linspace(0, 24, 50), [0, 24]]))),        # t=0 to t=24, no virus
    np.sort(np.unique(np.concatenate([np.linspace(24, 26, 50), [24, 26]]))),       # t=24 to t=26, no virus
    np.sort(np.unique(np.concatenate([[26, 26.04167], np.linspace(26, 28, 50), [28]]))),
    np.sort(np.unique(np.concatenate([[28], np.linspace(28, 31, 50), [31]]))),
    np.sort(np.unique(np.concatenate([[31], np.linspace(31, 66, 100), t_data[t_data >= 33]])))
]

injection_times = [26, 28, 31]
y0 = [19.958647, 0, 0, 0, 0]  # start from t=0 (T, R, I, V, F)
all_times = np.array([])
all_virus = np.array([])
all_T = np.array([])
all_R = np.array([])
all_I = np.array([])
all_F = np.array([])
for i, t_seg in enumerate(t_segments):
    sol = odeint(base_model, y0, t_seg, args=(λ_fixed, β_fit, δ_fit, p_fit, c_fit, γ_fit, roh_fit, α_fit))
    all_times = np.concatenate((all_times, t_seg))
    all_virus = np.concatenate((all_virus, sol[:, 3]))
    all_T = np.concatenate((all_T, sol[:, 0]))
    all_R = np.concatenate((all_R, sol[:, 1]))
    all_I = np.concatenate((all_I, sol[:, 2]))
    all_F = np.concatenate((all_F, sol[:, 4]))
    y0 = sol[-1].copy()
    if i in [1,2,3]:  # inject only at t=26, 28, 31
        y0[3] += 100

plt.figure(figsize=(10, 6))
plt.plot(all_times, safe_log10(all_virus), 'b-', label='Model Prediction')

V_data_list = np.array(V_data_list)
t0 = 26
plot_times = t0 + v_data

for j in range(len(v_data)):
    mask = ~np.isnan(V_data_list[:, j])
    plt.scatter([plot_times[j]] * np.sum(mask),
                safe_log10(V_data_list[:, j][mask]),
                s=60, zorder=10,
                label='Experimental Data' if j == 0 else "")

for t in injection_times:
    plt.axvline(t, color='gray', linestyle='--', alpha=0.5)
    plt.text(t, 6.5, f'Injection {t}', ha='center', va='top', fontsize=9, backgroundcolor='white')

plt.title("Virus Dynamics Comparison", fontsize=14)
plt.xlabel("Time (days post implant)", fontsize=12)
plt.ylabel("log$_{10}$(Virus Concentration)", fontsize=12)
plt.legend(loc='best')
plt.grid(alpha=0.3)
plt.ylim([-3, 7])
plt.xlim([0, 66])
plt.tight_layout()
plt.show()
# --- Tumor Plot ---
plt.figure(figsize=(10, 6))

for i in range(len(t_data)):
    if i < len(T_data_list):
        mask = ~np.isnan(T_data_list[i])
        plt.scatter(np.full(np.sum(mask), t_data[i]), T_data_list[i][mask],
                    s=60, color='orangered', alpha=0.7, label='Experimental Data' if i == 0 else "")

total_tumor = all_T + all_R + all_I
plt.plot(all_times, total_tumor, 'b-', linewidth=2, label='Model Prediction')

for t in injection_times:
    plt.axvline(t, color='gray', linestyle='--', alpha=0.5)
    plt.text(t, max(total_tumor) * 0.95, f'Injection {t}', ha='center', va='top', fontsize=9, backgroundcolor='white')

plt.title("Tumor Growth Dynamics", fontsize=14)
plt.ylabel("Tumor Size (mm³)", fontsize=12)
plt.xlabel("Time (days post implant)", fontsize=12)
plt.grid(alpha=0.3)
plt.legend(loc='best')
plt.xlim([0, 66])
plt.ylim([0, max(total_tumor) * 1.1])
plt.tight_layout()
plt.show()