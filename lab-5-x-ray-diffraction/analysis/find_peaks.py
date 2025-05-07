import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from lmfit.models import VoigtModel

# Load data
df = pd.read_csv("../data/polycrystaline-7.TXT", sep="\t", header=None, names=["2theta", "intensity"])
x = df["2theta"].values
y = df["intensity"].values

# Find peak positions
peaks, _ = find_peaks(y, height=200, distance=100)  # tune height/distance as needed
peak_positions = x[peaks]
print(peak_positions, len(peak_positions))

# Constants for Cu Kα lines (in Å), to compute angular separation
lambda1 = 1.5406  # Kα1
lambda2 = 1.5444  # Kα2

def delta_2theta(theta_deg):
    """Return angular separation between Kα1 and Kα2 in degrees at a given theta."""
    theta_rad = np.radians(theta_deg / 2)
    d_spacing = lambda1 / (2 * np.sin(theta_rad))
    theta2 = np.degrees(2 * np.arcsin(lambda2 / (2 * d_spacing)))
    return theta2 - theta_deg

# Create the composite model
model = None
params = None

for i, pos in enumerate(peak_positions):
    prefix1 = f"p{i}_ka1_"
    prefix2 = f"p{i}_ka2_"
    
    # Kα1 peak
    m1 = VoigtModel(prefix=prefix1)
    p1 = m1.make_params(center=pos, amplitude=1000, sigma=0.05)

    # Kα2 peak (shifted and half intensity)
    shift = delta_2theta(pos)
    m2 = VoigtModel(prefix=prefix2)
    p2 = m2.make_params(center=pos + shift, amplitude=500, sigma=0.05)

    # Fix relationship
    p2[prefix2+"amplitude"].set(expr=f"{prefix1}amplitude * 0.5")
    p2[prefix2+"sigma"].set(expr=f"{prefix1}sigma")
    p2[prefix2+"center"].set(expr=f"{prefix1}center + {shift:.6f}")

    # Add to global model
    if model is None:
        model = m1 + m2
        params = p1 + p2
    else:
        model += m1 + m2
        params.update(p1)
        params.update(p2)

# Fit and plot
result = model.fit(y, params, x=x)
result.plot_fit()
plt.title("Fitted XRD Peaks with Kα1 and Kα2 Doublets")
plt.xlabel("2θ (degrees)")
plt.ylabel("Intensity")
plt.show()

# Print fit report
print(result.fit_report())
