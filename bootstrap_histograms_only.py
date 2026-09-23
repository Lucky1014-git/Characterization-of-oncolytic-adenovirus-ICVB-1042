import csv
import numpy as np
import matplotlib.pyplot as plt

PARAM_NAMES = ["beta", "delta", "p", "c", "gamma", "roh", "alpha"]


def bootstrap_ci(values, alpha=0.05):
    sorted_vals = np.sort(values)
    lower = np.percentile(sorted_vals, 100 * alpha / 2)
    upper = np.percentile(sorted_vals, 100 * (1 - alpha / 2))
    return lower, upper


def load_bootstrap_csv(csv_path):
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"CSV file is empty: {csv_path}")

        rows = []
        for row in reader:
            if not row or not row[0].strip():
                continue
            try:
                rows.append([float(x) for x in row[1:-1]])
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No numeric bootstrap rows found in {csv_path}")

    return np.array(rows, dtype=float)


def plot_bootstrap_histograms(bootstrap_estimates, param_names=None, bins=100):
    if param_names is None:
        param_names = PARAM_NAMES

    if bootstrap_estimates.shape[1] != len(param_names):
        raise ValueError(
            f"Expected {len(param_names)} columns in bootstrap_estimates, "
            f"but got {bootstrap_estimates.shape[1]}"
        )

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()

    for i, name in enumerate(param_names):
        ax = axes[i]
        vals = bootstrap_estimates[:, i]
        low, high = bootstrap_ci(vals)

        ax.hist(
            vals,
            bins=bins,
            color='skyblue',
            edgecolor='black',
            linewidth=1.0,
            alpha=0.8,
            density=True,
        )
        ax.axvline(low, color='red', linestyle='--', linewidth=1.5, label='95% CI lower')
        ax.axvline(high, color='red', linestyle='--', linewidth=1.5, label='95% CI upper')
        ax.axvline(np.median(vals), color='black', linestyle='-', linewidth=1.0, label='Median')

        ax.set_title(f"{name} (95% CI)", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Density", fontsize=10)

        if np.all(vals > 0):
            ax.set_xscale('log')
        ax.tick_params(axis='x', labelbottom=False, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

    for j in range(len(param_names), len(axes)):
        axes[j].axis('off')

    fig.tight_layout()
    plt.show()


def plot_bootstrap_histograms_csv(csv_path, param_names=None, bins=10):
    bootstrap_estimates = load_bootstrap_csv(csv_path)
    plot_bootstrap_histograms(bootstrap_estimates, param_names=param_names, bins=bins)


if __name__ == "__main__":
    example_csv = "adenoICB1042-bootstrap_results.csv"
    try:
        plot_bootstrap_histograms_csv(example_csv, PARAM_NAMES)
    except FileNotFoundError:
        example_bootstrap_estimates = np.array([
            [1.2e-05, 0.02, 5000, 20, 0.001, 0.01, 0.1],
            [1.1e-05, 0.021, 5100, 21, 0.0011, 0.011, 0.11],
            [1.3e-05, 0.019, 4900, 19, 0.0009, 0.009, 0.09],
            [1.15e-05, 0.0205, 5050, 20.5, 0.00105, 0.0105, 0.105],
            [1.18e-05, 0.0208, 4990, 20.8, 0.00108, 0.0108, 0.108],
        ])
        plot_bootstrap_histograms(example_bootstrap_estimates, PARAM_NAMES)
