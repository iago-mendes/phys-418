import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from cycler import cycler
colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b3', '#937860', '#da8bc3', '#8c8c8c', '#ccb974', '#64b5cd']
plt.rcParams.update({
  'text.usetex' : True,
  'font.family' : 'serif',
  'font.serif' : ['Computer Modern Serif'],
  'font.size': 15,
  'axes.prop_cycle': cycler('color', colors)
})


# Read data
data_fname = './data/avay_iago12.txt'
data = np.loadtxt(data_fname, dtype=float)
time = data[:,0]
absorption = data[:,1]
transmission = data[:,2]


# Cubic fit done in Igor Pro for number of peaks vs time
K0 = -0.6
K0_error = 0.2
K1 = 1910
K1_error = 50
K2 = 60000
K2_error = 4000
K3 = -1.57e6
K3_error = 0.08e6
def peak_number(time):
  return K0 + K1 * time + K2 * time**2 + K3 * time**3


# Time-to-frequency conversion
FSR = 166.6 / 1e3 # in GHz
frequency = FSR * peak_number(time)
frequency_error = FSR * np.sqrt(K0_error**2 + K1_error**2 * time**2 + K2_error**2 * time**4 + K3_error**2 * time**6)


# Initialize plot
fig, ax = plt.subplots(1, 1, squeeze=True)
ax.scatter(frequency, absorption, label='Data points', marker='o')


# Use Gaussians to fit the absorption peaks
def gaussian(x, amplitude, center, std, offset):
  return amplitude * np.exp(-((x - center) / std)**2 / 2.) + offset
is_first_call = True # useful boolean for showing legend properly
def gaussian_fit(start, end):
  global is_first_call

  if is_first_call:
    ax.scatter(frequency[start:end], absorption[start:end], color=colors[1], marker='o', label='Points used for fits')
  else:
    ax.scatter(frequency[start:end], absorption[start:end], color=colors[1], marker='o')

  fit_initial_parameters = [
    -1, # amplitude
    np.mean(frequency[start:end]), # center
    np.std(frequency[start:end]), # std
    np.max(absorption[start:end]) # offset
  ]
  fit_parameters, covariance_matrix = curve_fit(gaussian, frequency[start:end], absorption[start:end], p0=fit_initial_parameters)
  fit_errors = np.sqrt(np.diag(covariance_matrix))
  
  plot_extra_points = 300
  if is_first_call:
    ax.plot(frequency[start-plot_extra_points:end+plot_extra_points],
            gaussian(frequency[start-plot_extra_points:end+plot_extra_points], *fit_parameters),
            color=colors[2],
            label='Gaussian fits')
  else:
    ax.plot(frequency[start-plot_extra_points:end+plot_extra_points],
            gaussian(frequency[start-plot_extra_points:end+plot_extra_points], *fit_parameters),
            color=colors[2])
    
  is_first_call = False

  return {
    'amplitude': fit_parameters[0],
    'center': fit_parameters[1],
    'std': fit_parameters[2],
    'offset': fit_parameters[3],
    'amplitude_error': fit_errors[0],
    'center_error': fit_errors[1],
    'std_error': fit_errors[2],
    'offset_error': fit_errors[3],
  }

gaussian1 = gaussian_fit(438, 550)
gaussian2 = gaussian_fit(588, 715)
gaussian3 = gaussian_fit(1000, 1125)
gaussian4 = gaussian_fit(1400, 1500)


# Find the frequency data point closest to the gaussian centers
def get_peak_center(gaussian_center):
  closest_index = 0
  for i in range(len(frequency)):
    if abs(frequency[i] - gaussian_center) < abs(frequency[closest_index] - gaussian_center):
      closest_index = i
  peak_center = frequency[closest_index]
  peak_center_error = frequency_error[closest_index]
  return peak_center, peak_center_error

peak1_center, peak1_center_error = get_peak_center(gaussian1['center'])
peak2_center, peak2_center_error = get_peak_center(gaussian2['center'])
peak3_center, peak3_center_error = get_peak_center(gaussian3['center'])
peak4_center, peak4_center_error = get_peak_center(gaussian4['center'])


# Find hyperfine splittings
Rb85_splitting = peak3_center - peak2_center
Rb85_splitting_error = np.sqrt(peak3_center_error**2 + peak2_center_error**2)
print(f'Rb85 splitting = {Rb85_splitting} +- {Rb85_splitting_error} GHz')

Rb87_splitting = peak4_center - peak1_center
Rb87_splitting_error = np.sqrt(peak4_center_error**2 + peak1_center_error**2)
print(f'Rb87 splitting = {Rb87_splitting} +- {Rb87_splitting_error} GHz')


print()


# Find width of resonance curves
def get_resonance_width(gaussian):
  width = 2. * np.sqrt(2. * np.log(2)) * gaussian['std'] * 1e3 # in MHz
  error = 2. * np.sqrt(2. * np.log(2)) * gaussian['std_error'] * 1e3 # in MHz
  return f'{width} +- {error} MHz'
print('Widths of resonance curves:')
print(get_resonance_width(gaussian1))
print(get_resonance_width(gaussian2))
print(get_resonance_width(gaussian3))
print(get_resonance_width(gaussian4))


# Edit and save plot
ax.legend()
ax.set_xlabel(r'$\nu - \nu_0$ (GHz)')
ax.set_ylabel(r'Absorption')
ax.set_yticks([])
ax.grid('on', linestyle='--', alpha=0.5)

fig.set_size_inches(12,6)
plt.tight_layout()

fig.savefig(f'./data/analysis.pdf', format='pdf', bbox_inches='tight')
plt.show()
