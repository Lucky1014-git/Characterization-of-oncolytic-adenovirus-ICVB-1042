import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# --- Helper Function ---
def safe_log10(x):
    return np.log10(np.maximum(x, 1e-10))


# --- Experimental Data ---
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
λ_fit = 0.1005 # Fixed tumor growth rate
#T - uninfected tumor cells, R - resistant cells, I - infected cells, V - virus particles
def base_model(Y, t, λ, β, δ, p, c, r, roh):
    T, R, I, V = Y
    dTdt = (λ * T) - (β * T * V) - (r * I * T) + (roh * R)
    dRdt = (r * I * T) - (roh * R)
    dIdt = (β * T * V) - (δ * I)
    dVdt = (p * I) - (c * V)
    return [dTdt, dRdt, dIdt, dVdt]

# --- SSR Function ---
def ssr(params):
    log_β, δ, log_p, c, r, roh = params
    β = 10 ** log_β
    p = 10 ** log_p
    total_ssr = 0.0
    # T - uninfected tumor cells(1.76), R - resistant cells(0), I - infected cells(0), V - virus particles(0)
    y0 = [1.76, 0, 0, 0]

    # Segment 1: 0–26 (no virus injection yet)
    t1 = [0, 24, 25]
    y1 = odeint(base_model, y0, t1, args=(λ_fit, β, δ, p, c, r, roh))
    for i in range(len(T_data_list[0])):
        if not np.isnan(T_data_list[0][i]):
            total_ssr += np.nansum((np.log10(T_data_list[0][i]) - np.log10(y1[1,0] + y1[1,1] + y1 [1,2])) ** 2)  # t=24
    #total_ssr += np.nansum((T_data_list[0]) - ((y1[1,0])) ** 2)  # t=24

    # Inject virus at t=25
    y2_init = y1[-1]
    y2_init[3] += 100  # Add 100 units of virus

    # Segment 2: 26–28
    t2 = [25, 26.04167, 27, 28]
    y2 = odeint(base_model, y2_init, t2, args=(λ_fit, β, δ, p, c, r, roh))

   # total_ssr +=  np.nansum(((y2[1, 3]) - (V_data_list[0])) ** 2)  # t=26.04167

    y1 = odeint(base_model, y0, t1, args=(λ_fit, β, δ, p, c, r, roh))
    for i in range(len(T_data_list[0])):
        if not np.isnan(T_data_list[0][i]):
            total_ssr += np.nansum((np.log10(T_data_list[1][i]) - np.log10(y2[0,0] + y2[0,1] + y2[0,2])) ** 2)  # t=26
    #total_ssr +=  np.nansum((T_data_list[1]) - ((y2[2, :3])) ** 2)  # t=26

    for i in range(len(V_data_list[0])):
        if not np.isnan(V_data_list[0][i]):
            total_ssr += np.nansum((np.log10(y2[1, 3]) - np.log10(V_data_list[0][i])) ** 2)  # t=26.04167

    for i in range(len(T_data_list[0])):
        if not np.isnan(T_data_list[0][i]):
            total_ssr += np.nansum((np.log10(T_data_list[2][i]) - np.log10(y2[3, 0] + y2[3,1] + y2[3,2])) ** 2)  # t=28

    y3_init = y2[-1]
    y3_init[3] += 100

    # Segment 3: 28–31
    t3 = [28, 31]
    y3 = odeint(base_model, y3_init, t3, args=(λ_fit, β, δ, p, c, r, roh))


    for i in range(len(V_data_list[0])):
        if not np.isnan(V_data_list[0][i]):
            total_ssr += np.nansum((np.log10(y3[0, 3]) - np.log10(V_data_list[1][i])) ** 2) # t=28


    for i in range(len(T_data_list[0])):
        if not np.isnan(T_data_list[0][i]):
            total_ssr += np.nansum((np.log10(T_data_list[4][i]) - np.log10(y3[1,0] + y3[1,1] + y3[1,2])) ** 2) # t=31

    # Inject virus again at t=31
    y4_init = y3[-1]
    y4_init[3] += 100

    # Segment 4: 31–66
    t4 = [31, 33, 35, 38, 40, 42, 45, 47, 49, 52, 54, 56, 59, 61, 63, 66]
    y4 = odeint(base_model, y4_init, t4, args=(λ_fit, β, δ, p,c, r, roh))
    print(len(T_data_list))
    for i, ti in enumerate(t4):
        print(i)
        for j in range(len(T_data_list[0])):
            if not np.isnan(T_data_list[0][j]):
                total_ssr += np.nansum((np.log10(T_data_list[i+3][j]) - np.log10(y4[i, 0]) + y4[i,1] + y4[i,2]) ** 2)  # t=31

    total_ssr += np.nansum((np.log10(y4[1, 3]) - np.log(V_data_list[2])) ** 2) # t=33

    print(f"SSR: {total_ssr:.4f}")
    return total_ssr




initial_guess = [
    np.log10(1e-8),        # β (infection rate)
    0.5,                   # δ (death rate of infected cells)
    np.log10(1e5),         # p (production rate of virus)
    1.0,                    # c (clearance rate of free virus)
    0.1,                   # r (resistance rate to infection)
    0.1                  # roh (reversion rate from resistant to susceptible)
]
bounds = [
    (np.log10(1e-11), np.log10(1e-5)),
    (0.05, 5.0),
    (np.log10(1e2), np.log10(1e8)),
    (0.05, 10.0),
    (0.001, 100),
    (0.05, 10.0)
]

result = minimize(ssr, initial_guess, bounds=bounds, method = "Nelder-Mead")

β_fit_log, δ_fit, p_fit_log, c_fit, r_fit, roh_fit = result.x
β_fit = 10 ** β_fit_log
p_fit = 10 ** p_fit_log
λ_fixed = λ_fit

print(f"Fitted λ: {λ_fixed:.4f}")
print(f"Fitted β: {β_fit:.4e}")
print(f"Fitted r: {r_fit:.4f}")
print(f"Fitted δ: {δ_fit:.4f}")
print(f"Fitted p: {p_fit:.4e}")
print(f"Fitted c: {c_fit:.4f}")

# --- Enhanced Simulation for Virus Dynamics ---
# Build time segments with critical points
t_segments = [
    np.sort(np.unique(np.concatenate([np.linspace(0, 25, 50), [0, 24, 25]]))),
    np.sort(np.unique(np.concatenate([[25, 26.04167], np.linspace(25, 28, 50), [28]]))),
    np.sort(np.unique(np.concatenate([[28], np.linspace(28, 31, 50), [31]]))),
    np.sort(np.unique(np.concatenate([[31], np.linspace(31, 66, 100), t_data[t_data >= 33]])))
]

# Simulate with injections
y0 = [1.76, 0, 0, 0]
all_times = np.array([])
all_virus = np.array([])
all_T = np.array([])
all_R = np.array([])
all_I = np.array([])

for i, t_seg in enumerate(t_segments):
    sol = odeint(base_model, y0, t_seg, args=(λ_fit, β_fit, δ_fit, p_fit, c_fit, r_fit, roh_fit))

    # Store results
    all_times = np.concatenate((all_times, t_seg))
    all_virus = np.concatenate((all_virus, sol[:, 3]))
    all_T = np.concatenate((all_T, sol[:, 0]))
    all_R = np.concatenate((all_R, sol[:, 1]))
    all_I = np.concatenate((all_I, sol[:, 2]))

    # Apply next injection (except after last segment)
    if i < len(t_segments) - 1:
        y0 = sol[-1].copy()
        y0[3] += 100  # Virus injection
        print(f"Injected 100 virus units at t={t_seg[-1]:.2f}")

# --- Virus Plot ---
plt.figure(figsize=(10, 6))
print([np.size(all_times), np.size(all_virus), np.size(all_T), np.size(all_R), np.size(all_I)])
# Model trajectory
plt.plot(all_times, safe_log10(all_virus), 'b-', label='Model Prediction')

# Experimental data points
plt.scatter([25] * len(V_data_list[0]), safe_log10(V_data_list[0]),
            color='red', s=60, zorder=10, label='t=25 (1hr post-injection)')
plt.scatter([28] * len(V_data_list[1]), safe_log10(V_data_list[1]),
            color='green', s=60, zorder=10, label='t=28 (injection time)')
plt.scatter([31] * len(V_data_list[2]), safe_log10(V_data_list[2]),
            color='purple', s=60, zorder=10, label='t=33 (2 days post-injection)')
# Injection markers
injection_times = [25, 28, 31]
for t in injection_times:
    plt.axvline(t, color='gray', linestyle='--', alpha=0.5)
    plt.text(t, plt.ylim()[1] * 0.95, f'Injection {t}',
             ha='center', va='top', fontsize=9, backgroundcolor='white')

# Formatting
plt.title("Virus Dynamics Comparison", fontsize=14)
plt.xlabel("Time (days)", fontsize=12)
plt.ylabel("log$_{10}$(Virus Concentration)", fontsize=12)
plt.legend(loc='best')
plt.grid(alpha=0.3)
plt.ylim([-3, 6])  # Adjusted for log scale
plt.xlim([25, 66])

# Add virus replication info
plt.annotate(f'Viral Production: {p_fit:.1e}\nClearance: {c_fit:.3f}/day',
             xy=(0.75, 0.15), xycoords='axes fraction',
             bbox=dict(boxstyle='round', alpha=0.2))

plt.tight_layout()
plt.show()

# --- Tumor Plot ---
plt.figure(figsize=(10, 6))

# Plot experimental data points
for i in range(len(t_data)):
    if i < len(T_data_list):
        mask = ~np.isnan(T_data_list[i])
        plt.scatter(np.full(np.sum(mask), t_data[i]), T_data_list[i][mask],
                    s=60, color='orangered', alpha=0.7, label='Experimental Data' if i == 0 else "")

# Plot tumor trajectory
total_tumor = all_T+all_R+all_I
plt.plot(all_times, total_tumor, 'b-', linewidth=2, label='Model Prediction')

# Mark injection times
for t in injection_times:
    plt.axvline(t, color='gray', linestyle='--', alpha=0.5)
    plt.text(t, plt.ylim()[1] * 0.95, f'Injection {t}',
             ha='center', va='top', fontsize=9, backgroundcolor='white')

plt.title("Tumor Growth Dynamics", fontsize=14)
plt.ylabel("Tumor Size (mm³)", fontsize=12)
plt.xlabel("Time (days)", fontsize=12)
plt.grid(alpha=0.3)
plt.legend(loc='best')
plt.show()

