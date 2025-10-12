import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Load the data
data = np.loadtxt("gaussian.dat")
x_data = data[:, 0]
y_data = data[:, 1]

# Define the Gaussian model
def gaussian(x, A, m, s):
    return A * np.exp(-((x - m) ** 2) / s)
# Define the objective function: Sum of Squared Residuals (SSR)
def compute_ssr(params):
    A, m, s = params
    y_model = gaussian(x_data, A, m, s)
    print(np.sum((y_model - y_data) ** 2))
    return np.sum((y_model - y_data) ** 2)

# Initial guess for A, m, s
initial_guess = [1.0, 0.0, 1.0]

# Call the minimizer
result = minimize(compute_ssr, initial_guess, method='L-BFGS-B', bounds=[(0, None), (None, None), (0.1, None)])

# Extract best-fit parameters
best_A, best_m, best_s = result.x

# Plotting
plt.scatter(x_data, y_data, color="black", label="Actual Data", alpha=0.6)
x_fit = np.linspace(min(x_data), max(x_data), 300)
y_fit = gaussian(x_fit, best_A, best_m, best_s)
plt.plot(x_fit, y_fit, color="red", label="Best Fit Gaussian", linewidth=2)
plt.title("Best Fit")
plt.xlabel("x data")
plt.ylabel("y data")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
