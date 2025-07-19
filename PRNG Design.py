import numpy as np
import matplotlib.pyplot as plt

# Shourov's XOR-Shift Pseudo-Random Number Generator
class ShourovXORShift:
    def __init__(self, shourov_seed=42):
        self.shourov_state = shourov_seed

    def shourov_next_random(self):
        temp = self.shourov_state
        temp ^= (temp << 13) & 0xFFFFFFFF
        temp ^= (temp >> 17)
        temp ^= (temp << 5) & 0xFFFFFFFF
        self.shourov_state = temp & 0xFFFFFFFF
        return self.shourov_state / 0xFFFFFFFF  # Normalize to [0, 1)

    def shourov_generate_series(self, total_values):
        return np.array([self.shourov_next_random() for _ in range(total_values)])

# Total number of values
shourov_total_points = 1000
shourov_rng = ShourovXORShift(shourov_seed=123456789)

shourov_x_values = shourov_rng.shourov_generate_series(shourov_total_points)
shourov_y_values = shourov_rng.shourov_generate_series(shourov_total_points)

# NumPy reference for comparison
np.random.seed(42)
shourov_numpy_x = np.random.uniform(0, 1, shourov_total_points)
shourov_numpy_y = np.random.uniform(0, 1, shourov_total_points)

# === Scatter Plots ===
fig, (shourov_plot1, shourov_plot2) = plt.subplots(1, 2, figsize=(12, 5))

shourov_plot1.scatter(shourov_x_values, shourov_y_values, alpha=0.6, s=20, color='darkorange')
shourov_plot1.set_title('Shourov XOR-Shift Distribution', fontsize=14, fontweight='bold')
shourov_plot1.set_xlabel('X values')
shourov_plot1.set_ylabel('Y values')
shourov_plot1.grid(True, alpha=0.3)
shourov_plot1.set_xlim(0, 1)
shourov_plot1.set_ylim(0, 1)

shourov_plot2.scatter(shourov_numpy_x, shourov_numpy_y, alpha=0.6, s=20, color='blue')
shourov_plot2.set_title('NumPy Uniform Distribution', fontsize=14, fontweight='bold')
shourov_plot2.set_xlabel('X values')
shourov_plot2.set_ylabel('Y values')
shourov_plot2.grid(True, alpha=0.3)
shourov_plot2.set_xlim(0, 1)
shourov_plot2.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("shourov_xorshift_scatter.pdf")
plt.show()

# === Statistics ===
print("=== Shourov PRNG vs NumPy ===")
print(f"Shourov XOR-Shift - Mean X: {np.mean(shourov_x_values):.3f}, Std X: {np.std(shourov_x_values):.3f}")
print(f"Shourov XOR-Shift - Mean Y: {np.mean(shourov_y_values):.3f}, Std Y: {np.std(shourov_y_values):.3f}")
print(f"NumPy Uniform - Mean X: {np.mean(shourov_numpy_x):.3f}, Std X: {np.std(shourov_numpy_x):.3f}")
print(f"NumPy Uniform - Mean Y: {np.mean(shourov_numpy_y):.3f}, Std Y: {np.std(shourov_numpy_y):.3f}")

# === Histograms ===
fig, ((shourov_hist1, shourov_hist2), (shourov_hist3, shourov_hist4)) = plt.subplots(2, 2, figsize=(10, 8))

shourov_hist1.hist(shourov_x_values, bins=30, alpha=0.7, color='darkorange', edgecolor='black')
shourov_hist1.set_title('Shourov RNG - X Distribution')
shourov_hist1.set_xlabel('Value')
shourov_hist1.set_ylabel('Frequency')

shourov_hist2.hist(shourov_y_values, bins=30, alpha=0.7, color='darkorange', edgecolor='black')
shourov_hist2.set_title('Shourov RNG - Y Distribution')
shourov_hist2.set_xlabel('Value')
shourov_hist2.set_ylabel('Frequency')

shourov_hist3.hist(shourov_numpy_x, bins=30, alpha=0.7, color='blue', edgecolor='black')
shourov_hist3.set_title('NumPy - X Distribution')
shourov_hist3.set_xlabel('Value')
shourov_hist3.set_ylabel('Frequency')

shourov_hist4.hist(shourov_numpy_y, bins=30, alpha=0.7, color='blue', edgecolor='black')
shourov_hist4.set_title('NumPy - Y Distribution')
shourov_hist4.set_xlabel('Value')
shourov_hist4.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig("shourov_xorshift_histograms.pdf")
plt.show()
