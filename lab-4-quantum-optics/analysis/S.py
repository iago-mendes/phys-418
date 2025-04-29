import numpy as np

from coincidences import N as N_data, alpha, beta
from state import fit_result, expected_coincidences

def measured_N(a, b):
  for i in range(len(N_data)):
    if alpha[i] == a and beta[i] == b:
      return N_data[i]
  return None

def corrected_N(a, b):
  A = fit_result.x[0]
  theta_l = 45  # degrees
  phi_m = fit_result.x[2]
  C = fit_result.x[3]
  return expected_coincidences(a, b, A, theta_l, phi_m, C)

def E(a, b, N):
  return (
    N(a, b) + N(a + 90, b + 90) - N(a, b + 90) - N(a + 90, b)
  ) / (
    N(a, b) + N(a + 90, b + 90) + N(a, b + 90) + N(a + 90, b)
  )

def S(N):
  return E(-45, -22.5, N) - E(-45, 22.5, N) + E(0, -22.5, N) + E(0, 22.5, N)

def S_error(N):
  n = []
  for i in range(len(N_data)):
    n.append(N(alpha[i], beta[i]))
  # Expression derived in the Mathematica notebook `calculations.nb`
  return np.sqrt(
    n[11] * ((-n[9] + n[11] + n[1] - n[3]) / (n[9] + n[11] + n[1] + n[3])**2 - 1 / (n[9] + n[11] + n[1] + n[3]))**2 +
    n[1]  * ((-n[9] + n[11] + n[1] - n[3]) / (n[9] + n[11] + n[1] + n[3])**2 - 1 / (n[9] + n[11] + n[1] + n[3]))**2 +
    n[9]  * ((-n[9] + n[11] + n[1] - n[3]) / (n[9] + n[11] + n[1] + n[3])**2 + 1 / (n[9] + n[11] + n[1] + n[3]))**2 +
    n[3]  * ((-n[9] + n[11] + n[1] - n[3]) / (n[9] + n[11] + n[1] + n[3])**2 + 1 / (n[9] + n[11] + n[1] + n[3]))**2 +
    n[12] * (-((-n[12] + n[14] + n[4] - n[6]) / (n[12] + n[14] + n[4] + n[6])**2) - 1 / (n[12] + n[14] + n[4] + n[6]))**2 +
    n[6]  * (-((-n[12] + n[14] + n[4] - n[6]) / (n[12] + n[14] + n[4] + n[6])**2) - 1 / (n[12] + n[14] + n[4] + n[6]))**2 +
    n[14] * (-((-n[12] + n[14] + n[4] - n[6]) / (n[12] + n[14] + n[4] + n[6])**2) + 1 / (n[12] + n[14] + n[4] + n[6]))**2 +
    n[4]  * (-((-n[12] + n[14] + n[4] - n[6]) / (n[12] + n[14] + n[4] + n[6])**2) + 1 / (n[12] + n[14] + n[4] + n[6]))**2 +
    n[13] * (-((-n[13] + n[15] + n[5] - n[7]) / (n[13] + n[15] + n[5] + n[7])**2) - 1 / (n[13] + n[15] + n[5] + n[7]))**2 +
    n[7]  * (-((-n[13] + n[15] + n[5] - n[7]) / (n[13] + n[15] + n[5] + n[7])**2) - 1 / (n[13] + n[15] + n[5] + n[7]))**2 +
    n[15] * (-((-n[13] + n[15] + n[5] - n[7]) / (n[13] + n[15] + n[5] + n[7])**2) + 1 / (n[13] + n[15] + n[5] + n[7]))**2 +
    n[5]  * (-((-n[13] + n[15] + n[5] - n[7]) / (n[13] + n[15] + n[5] + n[7])**2) + 1 / (n[13] + n[15] + n[5] + n[7]))**2 +
    n[2]  * (-((n[0] + n[10] - n[2] - n[8]) / (n[0] + n[10] + n[2] + n[8])**2) - 1 / (n[0] + n[10] + n[2] + n[8]))**2 +
    n[8]  * (-((n[0] + n[10] - n[2] - n[8]) / (n[0] + n[10] + n[2] + n[8])**2) - 1 / (n[0] + n[10] + n[2] + n[8]))**2 +
    n[0]  * (-((n[0] + n[10] - n[2] - n[8]) / (n[0] + n[10] + n[2] + n[8])**2) + 1 / (n[0] + n[10] + n[2] + n[8]))**2 +
    n[10] * (-((n[0] + n[10] - n[2] - n[8]) / (n[0] + n[10] + n[2] + n[8])**2) + 1 / (n[0] + n[10] + n[2] + n[8]))**2
  )


measured_S = S(measured_N)
measured_S_error = S_error(measured_N)
print(f'S (measured) = {measured_S:.4f} +- {measured_S_error:.4f}')

corrected_S = S(corrected_N)
corrected_S_error = S_error(corrected_N)
print(f'S (corrected) = {corrected_S:.4f} +- {corrected_S_error:.4f}')
