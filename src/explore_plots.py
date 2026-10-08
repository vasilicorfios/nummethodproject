"""Quick exploratory plots for the airfoil self-noise dataset."""

import matplotlib.pyplot as plt

from data_loader import load_airfoil_data

df = load_airfoil_data()

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
features = ["frequency_hz", "attack_angle_deg", "chord_length_m", "velocity_mps", "disp_thickness_m"]

for ax, feature in zip(axes.flat, features):
    ax.scatter(df[feature], df["spl_db"], s=8, alpha=0.4)
    ax.set_xlabel(feature)
    ax.set_ylabel("spl_db")
    if feature == "frequency_hz":
        ax.set_xscale("log")

axes.flat[-1].axis("off")
fig.tight_layout()

# SPL vs. each remaining feature, split out by chord length
chord_lengths = sorted(df["chord_length_m"].unique())
other_features = ["frequency_hz", "attack_angle_deg", "velocity_mps", "disp_thickness_m"]

fig2, axes2 = plt.subplots(len(chord_lengths), len(other_features), figsize=(16, 3 * len(chord_lengths)), sharey=True)

for row, chord in enumerate(chord_lengths):
    subset = df[df["chord_length_m"] == chord]
    for col, feature in enumerate(other_features):
        ax = axes2[row, col]
        ax.scatter(subset[feature], subset["spl_db"], s=8, alpha=0.4)
        if feature == "frequency_hz":
            ax.set_xscale("log")
        if row == len(chord_lengths) - 1:
            ax.set_xlabel(feature)
        if col == 0:
            ax.set_ylabel(f"chord={chord} m\nspl_db")

fig2.tight_layout()
plt.show()
