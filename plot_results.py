import matplotlib
matplotlib.use('Agg')  # Bypass Tkinter GUI and render directly to disk
import matplotlib.pyplot as plt

# Experimental Data Collected
configurations = [
    "Baseline\n(150k steps)",
    "Mid-Tuned\n(400k steps)",
    "Naive Shaping\n(300k steps)",
    "Solved Master\n(700k steps)"
]

mean_scores = [67.55, 140.45, 0.63, 276.15]
std_devs = [114.51, 115.40, 64.51, 21.50]
colors = ['#4A90E2', '#50E3C2', '#E74C3C', '#2ECC71']

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

# Bar chart with standard deviation error bars
bars = ax.bar(configurations, mean_scores, yerr=std_devs, capsize=6, color=colors, edgecolor='black', alpha=0.85, width=0.55)

# Benchmark line at 200 points
ax.axhline(y=200, color='crimson', linestyle='--', linewidth=1.8, label="Benchmark Solved Threshold (>= 200)")

# Labels and styling
ax.set_title("Autonomous Descent Controller Performance across Policy Iterations", fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel("Mean Episodic Reward (20 Evaluation Episodes)", fontsize=11)
ax.set_ylim(-80, 340)
ax.grid(axis='y', linestyle=':', alpha=0.6)

# Value callouts on bars
for bar, score in zip(bars, mean_scores):
    height = bar.get_height()
    va = 'bottom' if height >= 0 else 'top'
    offset = 8 if height >= 0 else -16
    ax.annotate(f"{score:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, offset),
                textcoords="offset points",
                ha='center', va=va, fontsize=10, fontweight='bold')

ax.legend(loc='upper left', frameon=True)
plt.tight_layout()

# Save image file directly
output_filename = "convergence_comparison.png"
plt.savefig(output_filename, dpi=300)
print(f"Success! Plot generated and saved to {output_filename}")