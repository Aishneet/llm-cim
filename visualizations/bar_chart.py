import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# -----------------------------
# Data
# -----------------------------
models = ["GPT2-350M", "GPT2-774M", "GPT2-1.5B", "OPT-1.3B", "LLaMA-7B", "OPT-6.7B"]

# CPU-only total latency (seconds)
cpu_only_s = np.array([20.333, 78.006, 185.278, 406.759, 346.593, 751.096])

# Hybrid CPU–CiM:
# CPU non-GEMV portion (seconds)
cpu_non_gemv_s = np.array([2.875, 6.282, 10.335, 15.020, 10.021, 20.363])

# Total CiM GEMV latency (seconds)
cim_total_s = np.array([0.000558, 0.001274, 0.002132, 0.003483, 0.002130, 0.003640])

# IMPORTANT:
# Replace these with your measured breakdown values if you have them.
# They must sum to cim_total_s for each model.
cim_xbar_adc_s   = cim_total_s * np.array([0.55, 0.55, 0.55, 0.55, 0.55, 0.55])
cim_comm_s       = cim_total_s * np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
cim_buffer_s     = cim_total_s * np.array([0.20, 0.20, 0.20, 0.20, 0.20, 0.20])
cim_digital_s    = cim_total_s - (cim_xbar_adc_s + cim_comm_s + cim_buffer_s)

# Hybrid total
hybrid_total_s = cpu_non_gemv_s + cim_total_s

# Convert to milliseconds for nicer plotting
cpu_only_ms = cpu_only_s * 1e3
cpu_non_gemv_ms = cpu_non_gemv_s * 1e3
cim_xbar_adc_ms = cim_xbar_adc_s * 1e3
cim_comm_ms = cim_comm_s * 1e3
cim_buffer_ms = cim_buffer_s * 1e3
cim_digital_ms = cim_digital_s * 1e3
hybrid_total_ms = hybrid_total_s * 1e3

# -----------------------------
# Plot setup
# -----------------------------
x = np.arange(len(models))
w = 0.34

fig, ax = plt.subplots(figsize=(14, 6.5), dpi=300, constrained_layout=True)

# Colors
cpu_color = "#1f1f1f"
xbar_color = "#f4a261"
comm_color = "#e9c46a"
buffer_color = "#2a4d69"
digital_color = "#4ecdc4"

# Main bars: CPU-only
ax.bar(
    x - w/2,
    cpu_only_ms,
    width=w,
    color=cpu_color,
    edgecolor="black",
    linewidth=0.8,
    label="CPU-only (with GEMV)"
)

# Main bars: Hybrid CPU–CiM (stacked)
bottom = np.zeros_like(x, dtype=float)

ax.bar(
    x + w/2,
    cim_xbar_adc_ms + cim_comm_ms + cim_buffer_ms + cpu_non_gemv_ms,
    width=w,
    bottom=bottom,
    color="white",
    edgecolor="black",
    linewidth=0.8,
    label="_nolegend_"
)

# Stack from bottom to top
ax.bar(x + w/2, cim_xbar_adc_ms, width=w, bottom=bottom,
       color=xbar_color, edgecolor="black", linewidth=0.6,
       label="CiM Xbar + DAC + ADC")
bottom += cim_xbar_adc_ms

ax.bar(x + w/2, cim_comm_ms, width=w, bottom=bottom,
       color=comm_color, edgecolor="black", linewidth=0.6,
       label="CiM Communication")
bottom += cim_comm_ms

ax.bar(x + w/2, cim_buffer_ms, width=w, bottom=bottom,
       color=buffer_color, edgecolor="black", linewidth=0.6,
       label="CiM Buffer")
bottom += cim_buffer_ms

ax.bar(x + w/2, cim_digital_ms, width=w, bottom=bottom,
       color=digital_color, edgecolor="black", linewidth=0.6,
       label="Digital Circuitry")
bottom += cim_digital_ms

ax.bar(x + w/2, cpu_non_gemv_ms, width=w, bottom=bottom,
       color="#8ecae6", edgecolor="black", linewidth=0.6,
       label="CPU attn+other")

# -----------------------------
# Formatting
# -----------------------------
ax.set_yscale("log")
ax.set_ylabel("Latency (ms, log scale)", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=25, ha="right", fontsize=11)
ax.set_xlabel("Models", fontsize=13, fontweight="bold")
ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.35)

ax.set_ylim(0.5, cpu_only_ms.max() * 1.6)

# Annotate totals
for i, val in enumerate(cpu_only_ms):
    ax.text(x[i] - w/2, val * 1.08, f"{cpu_only_s[i]:.1f}s",
            ha="center", va="bottom", fontsize=9, color="white", fontweight="bold")

for i, val in enumerate(hybrid_total_ms):
    label = f"{hybrid_total_s[i]*1e3:.1f} ms" if hybrid_total_s[i] < 1 else f"{hybrid_total_s[i]:.2f}s"
    ax.text(x[i] + w/2, val * 1.10, label,
            ha="center", va="bottom", fontsize=8, color="black", fontweight="bold")

# Legend
ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.01, 1.0),
    frameon=False,
    fontsize=10
)

# -----------------------------
# Zoom inset for the tiny CiM breakdown
# -----------------------------
axins = inset_axes(
    ax,
    width="34%",
    height="34%",
    loc="lower right",
    borderpad=2.0
)

# Show only the hybrid bars in the inset
# Use linear scale in ms to reveal the tiny stacked breakdown.
bottom = np.zeros_like(x, dtype=float)
axins.bar(x, cim_xbar_adc_ms, width=0.35, bottom=bottom,
          color=xbar_color, edgecolor="black", linewidth=0.4)
bottom += cim_xbar_adc_ms
axins.bar(x, cim_comm_ms, width=0.35, bottom=bottom,
          color=comm_color, edgecolor="black", linewidth=0.4)
bottom += cim_comm_ms
axins.bar(x, cim_buffer_ms, width=0.35, bottom=bottom,
          color=buffer_color, edgecolor="black", linewidth=0.4)
bottom += cim_buffer_ms
axins.bar(x, cim_digital_ms, width=0.35, bottom=bottom,
          color=digital_color, edgecolor="black", linewidth=0.4)
bottom += cim_digital_ms
axins.bar(x, cpu_non_gemv_ms, width=0.35, bottom=bottom,
          color="#8ecae6", edgecolor="black", linewidth=0.4)

axins.set_ylim(0, max(hybrid_total_ms) * 1.25)
axins.set_xlim(-0.6, len(models) - 0.4)
axins.set_xticks([])
axins.set_ylabel("ms", fontsize=9, fontweight="bold")
axins.set_title("Zoom: Hybrid CPU–CiM", fontsize=10, fontweight="bold")
axins.grid(True, axis="y", linestyle="--", alpha=0.25)

# Optional: connect inset to main plot
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="gray", lw=1.0)

plt.tight_layout()
plt.savefig("end_to_end_latency_breakdown.png", bbox_inches="tight")
plt.savefig("end_to_end_latency_breakdown.pdf", bbox_inches="tight")
plt.show()