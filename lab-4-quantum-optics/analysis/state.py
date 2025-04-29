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

from coincidences import alpha_check, beta_check, N_check, alpha, beta, N

# Define model of expected coincidences
def expected_coincidences(alpha, beta, A, theta_l, phi_m, C):
  alpha = np.radians(alpha)
  beta = np.radians(beta)
  theta_l = np.radians(theta_l)
  phi_m = np.radians(phi_m)
  return A * (
    np.sin(alpha)**2 * np.sin(beta)**2 * np.cos(theta_l)**2 +
    np.cos(alpha)**2 * np.cos(beta)**2 * np.sin(theta_l)**2 +
    1./4. * np.sin(2.*alpha) * np.sin(2.*beta) * np.sin(2.*theta_l) * np.cos(phi_m)
  ) + C

############################################################
# USE 5 MEASUREMENTS TO CHECK STATE WITH ANALYTIC FORMULAS #
############################################################

# Map indices to angles
def N_c(a, b):
  for i in range(len(N_check)):
    if alpha_check[i] == a and beta_check[i] == b:
      return N_check[i]
  return None

# Use analytic formulas to get state parameters
C_check = N_c(0, 90)
C_check_error = np.sqrt(N_c(0, 90))
A_check = N_c(0, 0) + N_c(90, 90) - 2. * C_check
A_check_error = np.sqrt(N_c(0, 0) + N_c(90, 90) + 4. * N_c(0, 90))
theta_l_check = np.arctan2(
  np.sqrt(N_c(90, 90) - C_check),
  np.sqrt(N_c(0, 0) - C_check)
)
theta_l_check_error = np.sqrt(
  N_c(90, 90) / (4.0 * (N_c(0, 0) - N_c(0, 90)) * (N_c(90, 90) - N_c(0, 90)) * (1 + (N_c(90, 90) - N_c(0, 90)) / (N_c(0, 0) - N_c(0, 90)))**2)
  + (N_c(0, 0) * (N_c(90, 90) - N_c(0, 90))) / (4.0 * (N_c(0, 0) - N_c(0, 90))**3 * (1 + (N_c(90, 90) - N_c(0, 90)) / (N_c(0, 0) - N_c(0, 90)))**2)
  + ((N_c(0, 0) - N_c(0, 90)) * N_c(0, 90) * (-1/(N_c(0, 0) - N_c(0, 90)) + (N_c(90, 90) - N_c(0, 90)) / (N_c(0, 0) - N_c(0, 90))**2)**2)
    / (4.0 * (N_c(90, 90) - N_c(0, 90)) * (1 + (N_c(90, 90) - N_c(0, 90)) / (N_c(0, 0) - N_c(0, 90)))**2)
)
cos_phi_m_check = 1. / np.sin(2. * theta_l_check) * (4. * (N_c(45, 45) - C_check) / A_check - 1.)
cos_phi_m_check_error = np.sqrt(
  # First term: Contribution from N_c(45, 45)
  (16 * N_c(45, 45) * (1 / np.sin(2 * theta_l_check))**2) /
  (N_c(0, 0) - 2 * N_c(0, 90) + N_c(90, 90))**2
  # Second term: Contribution from N_c(90, 90)
  + N_c(90, 90) * (
    (
      -4 * (N_c(45, 45) - N_c(0, 90)) * (1 / np.sin(2 * theta_l_check)) /
      (N_c(0, 0) - 2 * N_c(0, 90) + N_c(90, 90))**2
      - (
        (-1 + (4 * (N_c(45, 45) - N_c(0, 90))) /
        (N_c(0, 0) - 2 * N_c(0, 90) + N_c(90, 90)))
        * (1 / np.tan(2 * theta_l_check))
        * (1 / np.sin(2 * theta_l_check))
      ) / (
        (N_c(0, 0) - N_c(0, 90)) *
        np.sqrt((N_c(90, 90) - N_c(0, 90)) / (N_c(0, 0) - N_c(0, 90))) *
        (1 + (N_c(90, 90) - N_c(0, 90)) / (N_c(0, 0) - N_c(0, 90)))
      )
    )**2
  )
  # Third term: Contribution from N_c(0, 0)
  + N_c(0, 0) * (
    (
      -4 * (N_c(45, 45) - N_c(0, 90)) * (1 / np.sin(2 * theta_l_check)) /
      (N_c(0, 0) - 2 * N_c(0, 90) + N_c(90, 90))**2
      + (
        (N_c(90, 90) - N_c(0, 90)) *
        (-1 + (4 * (N_c(45, 45) - N_c(0, 90))) /
        (N_c(0, 0) - 2 * N_c(0, 90) + N_c(90, 90)))
        * (1 / np.tan(2 * theta_l_check))
        * (1 / np.sin(2 * theta_l_check))
      ) / (
        (N_c(0, 0) - N_c(0, 90))**2 *
        np.sqrt((N_c(90, 90) - N_c(0, 90)) / (N_c(0, 0) - N_c(0, 90))) *
        (1 + (N_c(90, 90) - N_c(0, 90)) / (N_c(0, 0) - N_c(0, 90)))
      )
    )**2
  )
  # Fourth term: Contribution from N_c(0, 90)
  + N_c(0, 90) * (
    (
      (
        (8 * (N_c(45, 45) - N_c(0, 90))) /
        (N_c(0, 0) - 2 * N_c(0, 90) + N_c(90, 90))**2
        - 4 / (N_c(0, 0) - 2 * N_c(0, 90) + N_c(90, 90))
      ) * (1 / np.sin(2 * theta_l_check))
      - (
        (-1 + (4 * (N_c(45, 45) - N_c(0, 90))) /
        (N_c(0, 0) - 2 * N_c(0, 90) + N_c(90, 90)))
        * (-1 / (N_c(0, 0) - N_c(0, 90)) +
        (N_c(90, 90) - N_c(0, 90)) / (N_c(0, 0) - N_c(0, 90))**2)
        * (1 / np.tan(2 * theta_l_check))
        * (1 / np.sin(2 * theta_l_check))
      ) / (
        np.sqrt((N_c(90, 90) - N_c(0, 90)) / (N_c(0, 0) - N_c(0, 90))) *
        (1 + (N_c(90, 90) - N_c(0, 90)) / (N_c(0, 0) - N_c(0, 90)))
      )
    )**2
  )
)

##########################################
# USE 16 MEASUREMENTS TO FIT STATE MODEL #
##########################################

# Fit model to data
def fit_residuals(params):
  return expected_coincidences(alpha, beta, *params) - N
initial_fit_parameters = [1.0, 45.0, 0.0, 0.0]  # [A, theta_l, phi_m, C]
fit_result = least_squares(fit_residuals, initial_fit_parameters)

# Calculate fit errors
s_sq = np.sum(fit_result.fun**2) / (len(N) - len(fit_result.x))
covariance_matrix = np.linalg.inv(fit_result.jac.T @ fit_result.jac) * s_sq
fit_errors = np.sqrt(np.diag(covariance_matrix))

##################
# PRINT AND PLOT #
##################

if __name__ == '__main__':
  print(f'Analytic parameters:')
  print(f'\tA = {A_check:.4f} +- {A_check_error:.4f}')
  print(f'\ttheta_l = {np.degrees(theta_l_check):.4f} +- {np.degrees(theta_l_check_error):.4f}')
  print(f'\tcos(phi_m) = {cos_phi_m_check:.4f} +- {cos_phi_m_check_error:.4f}')
  print(f'\tC = {C_check:.4f} +- {C_check_error:.4f}')

  print(f'Fit parameters:')
  print(f'\tA = {fit_result.x[0]:.4f} +- {fit_errors[0]:.4f}')
  print(f'\ttheta_l = {fit_result.x[1]:.4f} +- {fit_errors[1]:.4f}')
  print(f'\tphi_m = {fit_result.x[2]:.4f} +- {fit_errors[2]:.4f}')
  print(f'\tC = {fit_result.x[3]:.4f} +- {fit_errors[3]:.4f}')

  # Set up plot
  fig = plt.figure(figsize=(14, 7))
  ax1 = fig.add_subplot(131, projection='3d')
  ax2 = fig.add_subplot(132, projection='3d')
  ax3 = fig.add_subplot(133, projection='3d')

  # Surface plot
  alpha_grid = np.linspace(-120, 120, 200)
  beta_grid = np.linspace(-120, 120, 200)
  alpha_grid, beta_grid = np.meshgrid(alpha_grid, beta_grid)
  A_fit, theta_l_fit, phi_m_fit, C_fit = fit_result.x
  N_analytic = expected_coincidences(alpha_grid, beta_grid, A_check, theta_l_check, 0, C_check)
  N_fit = expected_coincidences(alpha_grid, beta_grid, A_fit, theta_l_fit, 0, C_fit)
  N_corrected = expected_coincidences(alpha_grid, beta_grid, A_fit, 45, 0, C_fit)
  ax1.plot_surface(alpha_grid, beta_grid, N_analytic, cmap='coolwarm', alpha=0.6, label='Analytic check')
  ax2.plot_surface(alpha_grid, beta_grid, N_fit, cmap='coolwarm', alpha=0.6, label='Fit model')
  ax3.plot_surface(alpha_grid, beta_grid, N_corrected, cmap='coolwarm', alpha=0.6, label='Corrected model')

  # Scatter plot of data
  ax1.errorbar(alpha_check, beta_check, N_check, zerr=np.sqrt(N_check), fmt='o', capsize=3, color='black', label='5 measurements')
  ax2.errorbar(alpha, beta, N, zerr=np.sqrt(N), fmt='o', capsize=3, color='black', label='16 measurements')

  for ax in (ax1, ax2, ax3):
    ax.view_init(elev=20, azim=-110)

    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')
    ax.set_zlabel(r'$N(\alpha,\beta)$')
    ax.legend()
  
  for ax, label in zip((ax1, ax2, ax3), ['(a)', '(b)', '(c)']):
    ax.text2D(0.5, 0, label, transform=ax.transAxes, ha='center', va='center')

  # fig.set_size_inches(12,6)
  plt.tight_layout()

  fig.savefig(f'state.pdf', format='pdf', bbox_inches='tight')

  plt.show()
