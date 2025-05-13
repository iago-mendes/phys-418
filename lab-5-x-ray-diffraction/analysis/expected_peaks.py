import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks as scipy_find_peaks
from lmfit.models import VoigtModel, GaussianModel, LorentzianModel
from cycler import cycler
import warnings
warnings.filterwarnings("ignore", message="Using UFloat objects with std_dev==0")
colors = ['#4c72b0', '#c44e52', '#55a868', '#8172b3', '#937860', '#da8bc3', '#8c8c8c', '#ccb974', '#64b5cd']
plt.rcParams.update({
'text.usetex' : True,
'font.family' : 'serif',
'font.serif' : ['Computer Modern Serif'],
'font.size': 15,
'axes.prop_cycle': cycler('color', colors)
})

# YBa2Cu3O7_peaks = np.loadtxt('../data/YBa2Cu3O7-peaks.csv', dtype=float, delimiter=' ', comments='#')

# print(len(YBa2Cu3O7_peaks))
# print(len(YBa2Cu3O7_peaks[YBa2Cu3O7_peaks[:,1] > 5]))
# print(len(YBa2Cu3O7_peaks[YBa2Cu3O7_peaks[:,1] > 10]))

# Load data
M4_data = np.loadtxt('../data/super-7.TXT', dtype=float)
M3_data = np.loadtxt('../data/super-m3-1.TXT', dtype=float)
YBa2Cu3O7_peaks = np.loadtxt('../data/YBa2Cu3O7-peaks.csv', dtype=float, delimiter=' ', comments='#')
# Y2O3_peaks = np.loadtxt('../data/Y2O3-peaks.csv', dtype=float, delimiter=' ', comments='#')
# BaO_peaks = np.loadtxt('../data/BaO-peaks.csv', dtype=float, delimiter=' ', comments='#')

fig, axes = plt.subplots(2, 1, squeeze=True)
axes[1].scatter(M3_data[:,0], M3_data[:,1], label='M3 data', color='black')
axes[0].scatter(M4_data[:,0], M4_data[:,1], label='M4 data', color='black')

for ax in axes:
  min_intensity = 0
  is_first = True
  # for i, peaks in enumerate([YBa2Cu3O7_peaks, Y2O3_peaks, BaO_peaks]):
  for i, peaks in enumerate([YBa2Cu3O7_peaks]):
    for peak in peaks:
      if peak[1] < min_intensity:
          continue
      ax.axvline(peak[0], color=['blue', 'red', 'green'][i], label=r'Expected peaks' if is_first else None, alpha=0.25)
      is_first = False
    # ax.text(center, ax.get_ylim()[1] * 0.98, f"{center:.2f}", 
    #         rotation=90, ha='right', va='top', color='gray')

  ax.legend()
  ax.set_ylabel("Counts")
axes[1].set_xlabel(r'$2\theta \ (^\circ)$')

fig.set_size_inches(8,6)
plt.tight_layout()
plt.subplots_adjust(hspace=0.03)
fig.savefig('peaks-YBa2Cu3O7.pdf', format='pdf', bbox_inches='tight')
plt.show()


# find_peaks('../data/polycrystaline-6.TXT', 'peaks-Si_polycrystalline-1st_peak.pdf', xlim=(28.2,28.8))
# find_peaks('../data/polycrystaline-7.TXT', 'peaks-Si_polycrystalline.pdf', xlim=(45,100))
# find_peaks('../data/single-std3.txt', 'peaks-Si_single_crystal-std.pdf', xlim=(65, 73.5))
# find_peaks('../data/single-510-1.TXT', 'peaks-Si_single_crystal-510.pdf')
# find_peaks('../data/super-7.TXT', 'peaks-superconductor-M4.pdf', min_height=75, min_distance=20)

