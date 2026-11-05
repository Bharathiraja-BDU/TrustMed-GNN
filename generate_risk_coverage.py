import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

folder = Path("results/phase4c_robustness")
df = pd.read_csv(folder / "phase4c_corrected_risk_coverage.csv")

summary = (
    df.groupby(["method", "target_coverage"], as_index=False)
    .agg(
        mean_coverage=("evaluation_coverage", "mean"),
        mean_risk=("evaluation_risk", "mean"),
        seeds=("seed", "nunique")
    )
)

assert (summary["seeds"] == 5).all(), "Expected five seeds per point"

plt.figure(figsize=(7.5, 5.2))

methods = {
    "entropy": ("Predictive entropy", "o"),
    "mc_sd": ("MC-dropout standard deviation", "s"),
    "fusion": ("Score-level uncertainty fusion", "^")
}

for method, (label, marker) in methods.items():
    data = summary[summary["method"] == method].sort_values("mean_coverage")

    plt.plot(
        data["mean_coverage"] * 100,
        data["mean_risk"] * 100,
        marker=marker,
        linewidth=2,
        markersize=6,
        label=label
    )

plt.xlabel("Actual retained coverage (%)")
plt.ylabel("Error rate among retained predictions (%)")
plt.title("Risk–Coverage Comparison")
plt.xlim(40, 102)
plt.ylim(0, 25)
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

png = folder / "trustmed_corrected_risk_coverage.png"
pdf = folder / "trustmed_corrected_risk_coverage.pdf"

plt.savefig(png, dpi=300, bbox_inches="tight")
plt.savefig(pdf, bbox_inches="tight")
plt.close()

print("Created:", png)
print("Created:", pdf)
