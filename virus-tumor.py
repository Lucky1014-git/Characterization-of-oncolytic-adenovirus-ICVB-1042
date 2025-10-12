import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# --- Helper Function ---
def safe_log10(x):
    return np.log10(np.maximum(x, 1e-10))


# --- Experimental Data ---
t_data = np.array([24, 26, 28, 31, 33, 35, 38, 40, 42, 45, 47, 49, 52, 54, 56, 59, 61, 63, 66])
T_data_list = [
    np.array([102,125,110,128,139]),
    np.array([107,135,119,135,148]),
    np.array([116, 143, 124, 143, 157]),
    np.array([125, 156, 133, 159, 170]),
    np.array([137, 169, 144, 173, 187]),
    np.array([149, 201, 188, 211, 206]),
    np.array([167, 222, 208, 237, 223]),
    np.array([187, 251, 233, 272, 250]),
    np.array([291, 336, 354, 405, 321]),
    np.array([367, 469, 451, 471, 382]),
    np.array([386, 500, 477, 492, 411]),
    np.array([441, np.nan, 562, np.nan, 425]),
    np.array([460, np.nan, 587, np.nan, 443]),
    np.array([605, np.nan, 679, np.nan, 518]),
    np.array([733, np.nan, np.nan, np.nan, 570]),
    np.array([846, np.nan, np.nan, np.nan, 617]),
    np.array([922, np.nan, np.nan, np.nan, 661]),
    np.array([np.nan, np.nan, np.nan, np.nan, 763])
]
v_data = np.array([0.04167, 1, 3, 7, 28])
V_data_list = [
    np.array([1.372643, 0.018196, 1.158547, 99.04117, 26509.89]),
    np.array([0.01, 0.012721, 0.01, 585.1756, 6010.401]),
    np.array([np.nan, 0.01, 0.01, 0.275531, 92476.8])
]


# --- Model ---
def base_model(Y, t, λ, β, k, δ, p, c):
    T, E, I, V = Y
    dTdt = λ * T - β * T * V
    dEdt = β * T * V - k * E
    dIdt = k * E - δ * I
    dVdt = p * I - c * V
    return [dTdt, dEdt, dIdt, dVdt]


# --- SSR Function ---
def ssr(params):
    λ, log_β, k, δ, log_p, c = params
    β = 10 ** log_β
    p = 10 ** log_p

    total_ssr = 0.0
    y0 = [1.76, 0, 0, 0]

    # Segment 1: 0–26 (no virus injection yet)
    t1 = [0, 24, 26]
    y1 = odeint(base_model, y0, t1, args=(λ, β, k, δ, p, c))
    total_ssr += np.nansum(safe_log10(T_data_list[0]) - np.sum(safe_log10(y1[1, :3])) ** 2)  # t=24

    # Inject virus at t=26
    y2_init = y1[-1]
    y2_init[3] += 100  # Add 100 units of virus

    # Segment 2: 26–28
    t2 = [26, 26.04167, 27, 28]
    y2 = odeint(base_model, y2_init, t2, args=(λ, β, k, δ, p, c))

    total_ssr += np.nansum((safe_log10(y2[1, 3]) - safe_log10(V_data_list[0])) ** 2)  # t=26.04167
    total_ssr += np.nansum(safe_log10(T_data_list[1]) - np.sum(safe_log10(y2[2, :3])) ** 2)  # t=26
    total_ssr += np.nansum(safe_log10(T_data_list[2]) - np.sum(safe_log10(y2[3, :3])) ** 2)  # t=28

    # Inject virus again at t=28
    y3_init = y2[-1]
    y3_init[3] += 100

    # Segment 3: 28–31
    t3 = [28, 29, 31]
    y3 = odeint(base_model, y3_init, t3, args=(λ, β, k, δ, p, c))
    total_ssr += np.nansum((safe_log10(y3[0, 3]) - safe_log10(V_data_list[1])) ** 2)  # t=28
    total_ssr += np.nansum((safe_log10(T_data_list[3]) - np.sum(safe_log10(y3[1, :3])) ** 2))  # t=29
    total_ssr += np.nansum(safe_log10(T_data_list[4]) - np.sum(safe_log10((y3[2, :3])) ** 2)) # t=31

    # Inject virus again at t=31
    y4_init = y3[-1]
    y4_init[3] += 100

    # Segment 4: 31–66
    t4 = [31, 33, 35, 38, 40, 42, 45, 47, 49, 52, 54, 56, 59, 61, 63, 66]
    y4 = odeint(base_model, y4_init, t4, args=(λ, β, k, δ, p, c))
    for i, ti in enumerate(t4):
        if i + 5 < len(T_data_list):
            total_ssr += np.nansum(safe_log10(T_data_list[i + 5]) - np.sum(safe_log10(y4[i, :3])) ** 2)

    total_ssr += np.nansum((safe_log10(y4[1, 3]) - safe_log10(V_data_list[2])) ** 2) # t=33

    print(f"SSR: {total_ssr:.4f}")
    return total_ssr


# --- Optimization ---
initial_guess = [0.12, 1.1735e-10, 0.1854, 0.0880, 7.4416e+04, 0.0913]
bounds = [
    (0.10, 0.35),
    (np.log10(1e-11), np.log10(1e-5)),
    (0.05, 5.0),
    (0.05, 5.0),
    (np.log10(1e2), np.log10(1e8)),
    (0.05, 10.0)
]

result = minimize(ssr, initial_guess, bounds=bounds)

λ_fit, β_fit_log, k_fit, δ_fit, p_fit_log, c_fit = result.x
β_fit = 10 ** β_fit_log
p_fit = 10 ** p_fit_log

print(f"Fitted λ: {λ_fit:.4f}")
print(f"Fitted β: {β_fit:.4e}")
print(f"Fitted k: {k_fit:.4f}")
print(f"Fitted δ: {δ_fit:.4f}")
print(f"Fitted p: {p_fit:.4e}")
print(f"Fitted c: {c_fit:.4f}")

# --- Enhanced Simulation for Virus Dynamics ---
# Build time segments with critical points
t_segments = [
    np.sort(np.unique(np.concatenate([np.linspace(0, 26, 50), [0, 24, 26]]))),
    np.sort(np.unique(np.concatenate([[26, 26.04167], np.linspace(26, 28, 50), [28]]))),
    np.sort(np.unique(np.concatenate([[28], np.linspace(28, 31, 50), [31]]))),
    np.sort(np.unique(np.concatenate([[31], np.linspace(31, 66, 100), t_data[t_data >= 33]])))
]

# Simulate with injections
y0 = [1.76, 0, 0, 0]
all_times = np.array([])
all_virus = np.array([])

for i, t_seg in enumerate(t_segments):
    sol = odeint(base_model, y0, t_seg, args=(λ_fit, β_fit, k_fit, δ_fit, p_fit, c_fit))

    # Store results
    all_times = np.concatenate((all_times, t_seg))
    all_virus = np.concatenate((all_virus, sol[:, 3]))

    # Apply next injection (except after last segment)
    if i < len(t_segments) - 1:
        y0 = sol[-1].copy()
        y0[3] += 100  # Virus injection
        print(f"Injected 100 virus units at t={t_seg[-1]:.2f}")

# --- Virus Plot ---
plt.figure(figsize=(10, 6))

# Model trajectory
plt.plot(all_times, safe_log10(all_virus), 'b-', label='Model Prediction')

# Experimental data points
plt.scatter([26.04167] * len(V_data_list[0]), safe_log10(V_data_list[0]),
            color='red', s=60, zorder=10, label='t=26.04167 (1hr post-injection)')
plt.scatter([28] * len(V_data_list[1]), safe_log10(V_data_list[1]),
            color='green', s=60, zorder=10, label='t=28 (injection time)')
plt.scatter([33] * len(V_data_list[2]), safe_log10(V_data_list[2]),
            color='purple', s=60, zorder=10, label='t=33 (2 days post-injection)')

# Injection markers
injection_times = [26, 28, 31]
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

# --- Tumor Plot for Reference ---
# plt.figure(figsize=(10, 4))
# for i in range(len(t_data)):
#     if i < len(T_data_list):
#         mask = ~np.isnan(T_data_list[i])
#         plt.scatter(np.full(np.sum(mask), t_data[i]), T_data_list[i][mask], s=20, color='orangered', alpha=0.7)
#
#
#         # Plot tumor trajectory
#         total_tumor = np.sum(odeint(base_model, [1.76, 0, 0, 0], all_times,
#                                     args=(λ_fit, β_fit, k_fit, δ_fit, p_fit, c_fit))[:, :3], axis=1)
#         plt.plot(all_times, total_tumor, 'b-', label='Tumor Model')
#
#         plt.title("Tumor Volume (For Context)")
#         plt.ylabel("Tumor Size (mm³)")
#         plt.xlabel("Time (days)")
#         plt.grid(alpha=0.3)
#         plt.legend()
#         plt.tight_layout()
#         plt.show()

# --- Tumor Plot ---
plt.figure(figsize=(10, 6))

# Plot experimental data points
for i in range(len(t_data)):
    if i < len(T_data_list):
        mask = ~np.isnan(T_data_list[i])
        plt.scatter(np.full(np.sum(mask), t_data[i]), T_data_list[i][mask],
                    s=60, color='orangered', alpha=0.7, label='Experimental Data' if i == 0 else "")

# Plot tumor trajectory
total_tumor = np.sum(odeint(base_model, [1.76, 0, 0, 0], all_times,
                            args=(λ_fit, β_fit, k_fit, δ_fit, p_fit, c_fit))[:, :3], axis=1)
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

# Add model parameters
plt.annotate(f'λ: {λ_fit:.4f}\nβ: {β_fit:.2e}\nk: {k_fit:.3f}\nδ: {δ_fit:.3f}',
             xy=(0.75, 0.15), xycoords='axes fraction',
             bbox=dict(boxstyle='round', alpha=0.2))

plt.tight_layout()
plt.show()
