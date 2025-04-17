import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from cycler import cycler
colors = ['#4c72b0', '#c44e52', '#55a868', '#8172b3', '#937860', '#da8bc3', '#8c8c8c', '#ccb974', '#64b5cd']
plt.rcParams.update({
  'text.usetex' : True,
  'font.family' : 'serif',
  'font.serif' : ['Computer Modern Serif'],
  'font.size': 15,
  'axes.prop_cycle': cycler('color', colors)
})

data = np.array([
  [0., 4, 23, 14],
  [20., 23, 7, 13],
  [40., 35, 1, 8],
  [60., 14, 10, 7],
  [80., 0, 27, 2],
  [100., 11, 14, 7],
  [120., 40, 2, 12],
  [140., 29, 4, 20],
  [160., 10, 22, 12],
  [180., 3, 29, 8],
])
angles = data[:,0]
counts0 = data[:,1]
counts90 = data[:,2]
counts45 = data[:,3]

def sin2(angle, amplitude, phase, cycles):
  return amplitude * (np.sin(angle * np.pi / 180. * cycles + phase))**2

fit_parameters0, covariance_matrix = curve_fit(
  sin2,
  angles,
  counts0,
  p0=[1., 2., 0.]
)
print(fit_parameters0)

fit_parameters90, covariance_matrix = curve_fit(
  sin2,
  angles,
  counts90,
  p0=[1., 2., 0.]
)
print(fit_parameters90)

start_index = 4
fit_parameters45, covariance_matrix = curve_fit(
  sin2,
  angles[start_index:],
  counts45[start_index:],
  p0=[1., 2., 0.]
)
print(fit_parameters45)

fine_angles = np.linspace(0, 180, 100)

fig, (ax1, ax2) = plt.subplots(1, 2, squeeze=True)
ax1.scatter(angles, counts0, label=r'$(0^\circ,0^\circ)$ data')
ax1.scatter(angles, counts90, label=r'$(90^\circ,90^\circ)$ data')
ax1.plot(fine_angles, sin2(fine_angles, *fit_parameters0), label=r'$(0^\circ,0^\circ)$ fit')
ax1.plot(fine_angles, sin2(fine_angles, *fit_parameters90), label=r'$(90^\circ,90^\circ)$ fit')

ax2.scatter(angles[start_index:], counts45[start_index:], label=r'$(45^\circ,45^\circ)$ data')
ax2.plot(fine_angles, sin2(fine_angles, *fit_parameters45), label=r'$(45^\circ,45^\circ)$ fit')

ax1.set_xlabel(r'$\lambda_{H}$')
ax2.set_xlabel(r'$\lambda_{Q}$')

for ax in (ax1, ax2):
  ax.legend()
  ax.set_ylabel(r'Coincidence counts')

fig.set_size_inches(12,6)
plt.tight_layout()

fig.savefig(f'wave-plates.pdf', format='pdf', bbox_inches='tight')

plt.show()

