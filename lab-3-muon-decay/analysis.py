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
data_fname = './data/aij2025_03_18-14-19.txt'
data = np.loadtxt(data_fname, dtype=float)
timestamps = data[:,0]
times = data[:,1]
time_min = np.min(times)
time_max = np.max(times)


# Initialize plot
fig, ax = plt.subplots(1, 1, squeeze=True)


def exponential_decay(t, amplitude, time_scale, offset):
  return amplitude * np.exp(- t / time_scale) + offset


def test_binning(num_bins, log = False):
  if log:
    print(f'number of bins given = {num_bins}')

  # Set up bins
  bin_width = (time_max - time_min) / (num_bins-1)
  bin_times = []
  for i in range(num_bins):
    bin_times.append(time_min + (2*i+1) * bin_width / 2.)
  bin_times = np.array(bin_times)

  # Count points in each bin
  bin_counts = np.zeros(num_bins)
  for t in times:
    bin_index = int(np.floor((t - time_min) / bin_width))
    bin_counts[bin_index] += 1

  # Remove bins with zero counts
  num_bins = np.count_nonzero(bin_counts)
  bin_times = np.delete(bin_times, bin_counts == 0)
  bin_counts = np.delete(bin_counts, bin_counts == 0)

  # Compute errors
  bin_errors = np.sqrt(bin_counts)

  # Plot data points and errors
  if log:
    ax.scatter(bin_times, bin_counts, color=colors[0], label='Used data points')
    ax.errorbar(bin_times, bin_counts, yerr=bin_errors, fmt="o", capsize=3, color=colors[0])

  # Ignore counts before maximum count (due to experimental issues at low time intervals)
  fit_start = 0
  while bin_counts[fit_start] < bin_counts[fit_start+1]:
    fit_start += 1
  if log:
    ax.scatter(bin_times[:fit_start], bin_counts[:fit_start], color=colors[2], label='Ignored data points')
    ax.errorbar(bin_times[:fit_start], bin_counts[:fit_start], yerr=bin_errors[:fit_start], fmt="o", capsize=3, color=colors[2])
  num_bins = num_bins - fit_start
  bin_times = bin_times[fit_start:]
  bin_counts = bin_counts[fit_start:]
  bin_errors = bin_errors[fit_start:]
  if log:
    print(f'number of bins used = {num_bins}')

  # Fit exponential decay
  fit_parameters, covariance_matrix = curve_fit(
    exponential_decay,
    bin_times,
    bin_counts,
    sigma=bin_errors,
    p0=[bin_counts[0], 2000., 0.]
  )
  fit_counts = exponential_decay(bin_times, *fit_parameters)
  fit_errors = np.sqrt(np.diag(covariance_matrix))

  # Compute chi squared
  chi_squared = 0.
  for i in range(num_bins):
    chi_squared += (bin_counts[i] - fit_counts[i])**2 / (bin_errors[i]**2)
  degrees_of_freedom = num_bins - len(fit_parameters)
  reduced_chi_squared = chi_squared / degrees_of_freedom
  if log:
    print(f'degrees of freedom = {degrees_of_freedom}')
    print(f'reduced chi squared = {reduced_chi_squared}')
    print(f'Muon decay time = {fit_parameters[1] / 1e3} +- {fit_errors[1] / 1e3} microseconds')

  # Plot results
  if log:
    ax.plot(bin_times, fit_counts, color=colors[1], label='Exponential decay fit', linewidth=3)
    
    ax.legend()
    ax.set_xlabel(r'Decay times (ns)')
    ax.set_ylabel(r'Counts')
    ax.grid('on', linestyle='--', alpha=0.5)

    fig.set_size_inches(12,6)
    plt.tight_layout()

    fig.savefig(f'./data/analysis.pdf', format='pdf', bbox_inches='tight')

    plt.show()
  
  return reduced_chi_squared


# Find the number of bins that gives the lowest reduced chi squared
print('Optimizing binning...')
best_num_bins = 0
best_reduced_chi_squared = np.Infinity
for num_bins in range(50, 500, 5):
  reduced_chi_squared = test_binning(num_bins)
  if abs(reduced_chi_squared-1.) < abs(best_reduced_chi_squared-1.):
    best_reduced_chi_squared = reduced_chi_squared
    best_num_bins = num_bins
print(f'best_num_bins = {best_num_bins}')
print(f'best_reduced_chi_squared = {best_reduced_chi_squared}')
print()


# Run optimal binning again to show results
test_binning(best_num_bins, log = True)
