import numpy as np
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


data_fname = './data/avay_iago12.txt'
data = np.loadtxt(data_fname, dtype=float)
time = data[:,0]
absorption = data[:,1]
transmission = data[:,2]


fig, ax = plt.subplots(1, 1, squeeze=True)
ax.plot(time, transmission - 5., label='Transmission')
ax.plot(time, absorption, label='Absorption')

ax.legend()
ax.set_xlabel('Time')
ax.set_yticks([])
ax.grid('on', linestyle='--', alpha=0.5)

fig.set_size_inches(12,6)
plt.tight_layout()

fig.savefig(f'./data/time_plot.pdf', format='pdf', bbox_inches='tight')
plt.show()
