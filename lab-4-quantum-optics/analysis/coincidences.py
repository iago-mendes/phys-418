import numpy as np

# Read data
data_fname = '../data/16.txt'
data = np.loadtxt(data_fname, dtype=float, comments='#')
alpha = data[:,0]
beta = data[:,1]
raw_NA = data[:,2]
raw_NB = data[:,3]
raw_N = data[:,4]

# Subtract background and accidental coincidences
background_NA = 443042.
NA = raw_NA - background_NA
background_NB = 299624.
NB = raw_NB - background_NB
coincidence_window = 25e-9 # in seconds
measurement_duration = 8. * 60. # in seconds
Nac = NA * NB * coincidence_window / measurement_duration
N = raw_N - Nac
