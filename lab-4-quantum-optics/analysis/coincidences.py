import numpy as np


COINCIDENCE_WINDOW = 25e-9 # in seconds
BACKGROUND_NA = 443042. / (8. * 60.) # in counts / second
BACKGROUND_NB = 299624. / (8. * 60)  # in counts / second


def filter_data(fname, duration):
  data = np.loadtxt(fname, dtype=float, comments='#')
  alpha = data[:,0]
  beta = data[:,1]
  raw_NA = data[:,2]
  raw_NB = data[:,3]
  raw_N = data[:,4]

  NA = raw_NA - BACKGROUND_NA * duration
  NB = raw_NB - BACKGROUND_NA * duration
  Nac = NA * NB * COINCIDENCE_WINDOW / duration
  N = raw_N - Nac

  return alpha, beta, N


# Filter 5 measurements taken to check state
alpha_check, beta_check, N_check = filter_data('../data/5.txt', 60.)

if __name__ == '__main__':
  # Print filtered data
  print(f'5 measurements taken to check state:')
  print(f"\t{'alpha':<10}{'beta':<10}{'N':<10}")
  for a, b, n in zip(alpha_check, beta_check, N_check):
    print(f"\t{a:<10}{b:<10}{n:<10.0f}")
  print()


# Filter 16 measurements taken to test Bell's inequality
alpha, beta, N = filter_data('../data/16.txt', 8. * 60.)

if __name__ == '__main__':
  # Print filtered data
  print(f"16 measurements taken test Bell's inequality:")
  print(f"\t{'alpha':<10}{'beta':<10}{'N':<10}")
  for a, b, n in zip(alpha, beta, N):
    print(f"\t{a:<10}{b:<10}{n:<10.0f}")
