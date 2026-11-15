import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

comparison_file = RESULTS / "model_comparison.csv"
df = pd.read_csv(comparison_file)

baseline = df[df["model"] == "Baseline"].copy()
baseline = baseline[["epsilon", "accuracy", "attack_success_rate"]]
baseline = baseline.sort_values("epsilon")

if len(baseline) != 5:
    raise ValueError("Expected five baseline epsilon results.")

baseline.to_csv(RESULTS / "fgsm_accuracy.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(baseline["epsilon"], baseline["accuracy"], marker="o")
plt.xlabel("FGSM Epsilon")
plt.ylabel("Accuracy (%)")
plt.title("Baseline CNN Accuracy under FGSM Attack")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS / "fgsm_accuracy.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(
    baseline["epsilon"],
    baseline["attack_success_rate"],
    marker="o"
)
plt.xlabel("FGSM Epsilon")
plt.ylabel("Attack Success Rate (%)")
plt.title("Baseline CNN FGSM Attack Success Rate")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS / "fgsm_attack_success_rate.png", dpi=300)
plt.close()

print("Updated results/fgsm_accuracy.csv")
print("Updated results/fgsm_accuracy.png")
print("Updated results/fgsm_attack_success_rate.png")
print("\nUpdated baseline results:")
print(baseline.to_string(index=False))
