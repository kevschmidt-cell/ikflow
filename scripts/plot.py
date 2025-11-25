import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

# Trainingszeit in Minuten berechnen
df["minutes"] = 10 + (df["version"] - 1) * 5

plt.figure(figsize=(8,5))
plt.plot(df["minutes"], df["l2_error_mm"], marker="o", markersize=3, label="Posefehler")

plt.axhline(y=5, linestyle="--", linewidth=1, label="5 mm Grenze")

plt.title("Posefehler vs. Trainingszeit (in Minuten)")
plt.xlabel("Trainingszeit (min)")
plt.ylabel("Posefehler (mm)")

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

