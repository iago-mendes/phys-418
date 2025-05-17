import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
plt.rcParams.update({
  'text.usetex' : True,
  'font.family' : 'serif',
  'font.serif' : ['Computer Modern Serif'],
  'font.size': 15,
})

two_thetas = [
  28.464,
  47.330,
  56.149,
  69.163,
  76.407,
  88.063,
  94.986,
]
lattice_parameters = [
  5.42677,
  5.42786,
  5.42844,
  5.42854,
  5.42896,
  5.42920,
  5.42936,
]

two_thetas = np.array(two_thetas)
lattice_parameters = np.array(lattice_parameters)

def lattice_parameter_model(two_theta, A, Delta_h):
  R = 285. # distance from sample to detector in mm
  theta = np.radians(two_theta) / 2.
  return A / (1. - Delta_h / R * np.cos(theta)**2)


fit_parameters, covariance_matrix = curve_fit(
  lattice_parameter_model,
  two_thetas,
  lattice_parameters,
  p0=[5.431, 0.]
)
fit_model = lattice_parameter_model(two_thetas, *fit_parameters)
fit_errors = np.sqrt(np.diag(covariance_matrix))

print('Fit parameters:')
print(f'\tA = {fit_parameters[0]:.5f} +- {fit_errors[0]:.5f}')
print(f'\tDelta_h = {fit_parameters[1]:.5f} +- {fit_errors[1]:.5f}')

fractional_correction = (fit_model - fit_parameters[0]) / fit_model
corrected_lattice_parameters = (1. - fractional_correction) * lattice_parameters

print()
print(f'Corrected lattice parameter = {np.average(corrected_lattice_parameters):.5f} +- {np.std(corrected_lattice_parameters):.5f}')


fig, (ax1, ax2) = plt.subplots(1, 2, squeeze=True)

ax1.scatter(two_thetas, lattice_parameters, label='Data', color='black')
ax1.plot(two_thetas, fit_model, label='Fit', color='red')

ax2.scatter(two_thetas, corrected_lattice_parameters, label='Corrected', color='black')

all_data = np.concatenate([lattice_parameters, fit_model, corrected_lattice_parameters])
for ax in (ax1, ax2):
  extra_space = 0.1 * (np.max(all_data) - np.min(all_data))
  ax.set_ylim(np.min(all_data) - extra_space, np.max(all_data) + extra_space)

  ax.legend()
  ax.set_xlabel(r'$2\theta\,(^\circ)$')
  ax.grid('on', linestyle='--', alpha=0.5)
ax1.set_ylabel(r'Lattice parameter $a$')
ax2.set_yticklabels([])

fig.set_size_inches(8,4)
plt.tight_layout()
plt.subplots_adjust(wspace=0.03)

fig.savefig('offset_correction.pdf', format='pdf', bbox_inches='tight')
plt.show()
