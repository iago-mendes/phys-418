import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

plt.rcParams.update({
	'text.usetex' : True,
	'font.family' : 'serif',
	'font.serif' : ['Computer Modern Serif'],
	'font.size': 15,
})


# data_fname = 'aij-b-2-13-0-data'
# data_fname = 'aij-b-2-13-1-data'
data_fname = 'aij-a-2-13-0-data'

data = np.loadtxt(f'./data/{data_fname}.txt', dtype=float, skiprows=12)
applied_field = data[:,0]
moment = data[:,1]


def langevin(x):
  return 1. / np.tanh(x) - 1. / x

def fit_function(x, A, B):
  return A * langevin(B * x)

fit_results = curve_fit(fit_function, applied_field, moment)
fit_A = fit_results[0][0]
fit_B = fit_results[0][1]
print(f'F(x): {fit_A} L({fit_B} x)')

extrapolated_applied_field = np.linspace(-20000, 20000, 1000)

fig, ax = plt.subplots(1, 1, squeeze=True)

ax.scatter(applied_field, moment, marker='+', color='black', label='Data points')
ax.plot(extrapolated_applied_field, fit_A * langevin(fit_B * extrapolated_applied_field), label=f'{fit_A:.2f} L({fit_B:.1e} H)')

ax.legend()
ax.set_xlabel(r'Applied field $H$ (G)')
ax.set_ylabel(r'Moment $M$ (emu)')
ax.grid('on', linestyle='--', alpha=0.5)

fig.set_size_inches(8,5)
plt.tight_layout()
plt.show()

# print(data)
# np.