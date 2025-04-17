import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from cycler import cycler
colors = ['#4c72b0', '#c44e52', '#55a868', '#8172b3', '#937860', '#da8bc3', '#8c8c8c', '#ccb974', '#64b5cd']
plt.rcParams.update({
  'text.usetex' : True,
  'font.family' : 'serif',
  'font.serif' : ['Computer Modern Serif'],
  'font.size': 15,
  'axes.prop_cycle': cycler('color', colors)
})

from coincidences import N, alpha, beta

def expected_coincidences(alpha, beta, A, theta_l, phi_m, C):
  # # Unpack angles
  # alpha, beta = angles

  # Convert from degrees to radians
  alpha = np.radians(alpha)
  beta = np.radians(beta)
  theta_l = np.radians(theta_l)
  phi_m = np.radians(phi_m)
  
  return A * (
    np.sin(alpha)**2 * np.sin(beta)**2 * np.cos(theta_l)**2 +
    np.cos(alpha)**2 * np.cos(beta)**2 * np.sin(theta_l)**2 +
    1./4. * np.sin(2.*alpha) * np.sin(2.*beta) * np.sin(2.*theta_l) * np.cos(phi_m)
  ) + C


# fit_parameters, covariance_matrix = curve_fit(expected_coincidences, (alpha, beta), N, p0=[1., 45., 0., 0.])
# print(fit_parameters)
# fit_errors = np.sqrt(np.diag(covariance_matrix))

def residuals(params, alpha, beta):
    A, theta_l, phi_m, C = params
    # Convert degrees to radians
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    theta_l = np.radians(theta_l)
    phi_m = np.radians(phi_m)

    expected_N = A * (
        np.sin(alpha)**2 * np.sin(beta)**2 * np.cos(theta_l)**2 +
        np.cos(alpha)**2 * np.cos(beta)**2 * np.sin(theta_l)**2 +
        0.25 * np.sin(2.*alpha) * np.sin(2.*beta) * np.sin(2.*theta_l) * np.cos(phi_m)
    ) + C

    return expected_N - N


p0 = [1.0, 45.0, 0.0, 0.0]  # [A, theta_l, phi_m, C]
result = least_squares(residuals, p0, args=(alpha, beta))

print(result)

fig = plt.figure(figsize=(14, 7))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
# print(result)

# Surface plot
# Create grid
A_fit, theta_l_fit, phi_m_fit, C_fit = result.x
# A_fit = 1.
# theta_l_fit = 25
# phi_m_fit = 0.
# C_fit = 0.
# alpha_grid = np.linspace(np.min(alpha), np.max(alpha), 4)
# beta_grid = np.linspace(np.min(beta), np.max(beta), 4)
alpha_grid = np.linspace(-120, 120, 20)
beta_grid = np.linspace(-120, 120, 20)
alpha_grid, beta_grid = np.meshgrid(alpha_grid, beta_grid)
N_fit = expected_coincidences(alpha_grid, beta_grid, A_fit, theta_l_fit, phi_m_fit, C_fit)
N_ideal = expected_coincidences(alpha_grid, beta_grid, A_fit, 45, phi_m_fit, C_fit)
ax1.plot_surface(alpha_grid, beta_grid, N_fit, cmap='coolwarm', alpha=0.7, label='Fit')
ax2.plot_surface(alpha_grid, beta_grid, N_ideal, cmap='coolwarm', alpha=0.7, label='Ideal fit')
# ax.plot_surface(alpha_grid, beta_grid, N_ideal - N_fit, cmap='viridis', alpha=0.7)


# Scatter plot of data
ax1.scatter(alpha, beta, N, color='r', label='Data')
# N = N.reshape(4, 4)
# alpha_grid = np.linspace(np.min(alpha), np.max(alpha), 4)
# beta_grid = np.linspace(np.min(beta), np.max(beta), 4)
# alpha_grid, beta_grid = np.meshgrid(alpha_grid, beta_grid)
# ax.plot_surface(alpha_grid, beta_grid, N)

for ax in (ax1, ax2):
  ax.view_init(elev=30, azim=75)

  ax.set_xlabel(r'$\alpha$')
  ax.set_ylabel(r'$\beta$')
  ax.set_zlabel(r'$N(\alpha,\beta)$')
  ax.legend()

# fig.set_size_inches(12,6)
plt.tight_layout()

fig.savefig(f'state.pdf', format='pdf', bbox_inches='tight')

plt.show()

