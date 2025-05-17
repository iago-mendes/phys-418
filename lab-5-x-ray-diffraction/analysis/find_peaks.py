import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks as scipy_find_peaks
from lmfit.models import VoigtModel
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

K_ALPHA_1_WAVELENGTH = 1.54051
K_ALPHA_2_WAVELENGTH = 1.54433

def find_peaks(data_fname, plot_fname, min_height=200, min_distance=100, xlim=None, peak_labels=None):
  print(f'Finding peaks in {data_fname}')

  # Load data
  data = np.loadtxt(data_fname, dtype=float)
  two_theta = data[:, 0]
  intensity = data[:, 1]

  # Ignore background counts
  mask = two_theta >= 20
  two_theta = two_theta[mask]
  intensity = intensity[mask]

  # Find peak positions
  peaks_data_indices, _ = scipy_find_peaks(intensity, height=min_height, distance=min_distance)
  peak_positions = two_theta[peaks_data_indices]
  print(f'\tFound {len(peak_positions)} peaks')

  # Create the composite model
  composite_model = None
  composite_params = None
  for peak_number, data_index in enumerate(peaks_data_indices):
    prefix1 = f"p{peak_number}_ka1_"
    prefix2 = f"p{peak_number}_ka2_"

    peak1_two_theta = two_theta[data_index]
    peak1_intensity = intensity[data_index]
    # print(f"\tPeak {peak_number}: {peak1_two_theta:.2f} ({peak1_intensity:.2f})")
    
    # K alpha 1 peak
    model1 = VoigtModel(prefix=prefix1)
    params1 = model1.make_params(center=peak1_two_theta, amplitude=peak1_intensity, sigma=0.005)

    # Constrain K alpha 1 peak to detected region
    params1[prefix1 + "center"].set(min=peak1_two_theta - 0.5, max=peak1_two_theta + 0.5)
    params1[prefix1 + "amplitude"].set(value=peak1_intensity, min=peak1_intensity * 0.1, max=peak1_intensity * 2.)
    # params1[prefix1 + "amplitude"].set(value=peak1_intensity, vary=False)

    # Estimate location of K alpha 2 peak
    d_spacing = K_ALPHA_1_WAVELENGTH / (2 * np.sin(np.radians(peak1_two_theta / 2)))
    peak2_two_theta = 2. * np.degrees(np.arcsin(K_ALPHA_2_WAVELENGTH / (2 * d_spacing)))
    splitting = peak2_two_theta - peak1_two_theta
    peak2_intensity = peak1_intensity * 0.5
    
    # K alpha 2 peak
    model2 = VoigtModel(prefix=prefix2)
    params2 = model2.make_params(center=peak2_two_theta, amplitude=peak2_intensity, sigma=0.05)

    # Enforce relationship between K alpha 1 and K alpha 2 peaks
    params2[prefix2+"amplitude"].set(expr=f"{prefix1}amplitude * 0.5")
    params2[prefix2+"sigma"].set(expr=f"{prefix1}sigma")
    params2[prefix2+"center"].set(expr=f"{prefix1}center + {splitting}")

    # Add to composite model
    if composite_model is None:
        composite_model = model1 + model2
        composite_params = params1 + params2
    else:
        composite_model += model1 + model2
        composite_params.update(params1)
        composite_params.update(params2)

  if len(peaks_data_indices) > 0:
    fit_result = composite_model.fit(
      intensity, 
      composite_params,
      x=two_theta,
      max_nfev=1000,
      # iter_cb=lambda p, i, r, *a, **k: print(f"\tIteration {i:04}: residual norm = {np.linalg.norm(r):.2f}") if i % 10 == 0 else None
    )
    peaks = []
    for peak_number in range(len(peaks_data_indices)):
      peaks.append(fit_result.params[f"p{peak_number}_ka1_center"].value)
    print(f"\tFitted peaks: {peaks}")

  fig, ax = plt.subplots(1, 1, squeeze=True)
  ax.scatter(two_theta, intensity, label='Data', color='black')
  if len(peaks_data_indices) > 0:
    ax.plot(two_theta, fit_result.best_fit, label='Fit', color='red')
  ax.set_xlabel(r'$2\theta \ (^\circ)$')
  ax.set_ylabel("Counts")
  ax.legend()

  if xlim is not None:
    ax.set_xlim(xlim)
    visible_y = intensity[(two_theta >= xlim[0]) & (two_theta <= xlim[1])]
    ax.set_ylim(-100, visible_y.max() * 1.2)

  if peak_labels is not None:
    for peak_number in range(len(peaks_data_indices)):
        peak_two_theta = two_theta[peaks_data_indices[peak_number]]
        if xlim and (peak_two_theta < xlim[0] or peak_two_theta > xlim[1]):
            continue
        center = fit_result.params[f"p{peak_number}_ka1_center"].value
        ax.axvline(center, color='gray', linestyle='--', linewidth=1)
        ax.text(center, ax.get_ylim()[1] * 0.98, peak_labels[peak_number], 
                rotation=0, ha='right', va='top', color='gray')
  
  fig.set_size_inches(8,4)
  plt.tight_layout()
  fig.savefig(plot_fname, format='pdf', bbox_inches='tight')
  plt.show()


find_peaks(
  '../data/polycrystaline-6.TXT',
  'peaks-Si_polycrystalline-1st_peak.pdf',
  xlim=(28.2,28.8),
  peak_labels=['(111)']
)
find_peaks(
  '../data/polycrystaline-7.TXT',
  'peaks-Si_polycrystalline.pdf',
  peak_labels=['(111)', '(220)', '(311)', '(400)', '(331)', '(422)', '(511)']
)
find_peaks(
  '../data/single-std3.txt',
  'peaks-Si_single_crystal-std.pdf',
  xlim=(65, 73.5),
  peak_labels=['(400)']
)
find_peaks('../data/single-510-1.TXT', 'peaks-Si_single_crystal-510.pdf')
