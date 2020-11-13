# https://github.com/raphaelvallat/yasa/blob/master/notebooks/08_bandpower.ipynb
import mne
raw = mne.io.read_raw_edf("chb01_01.edf",preload=True, verbose=0)
raw.pick_types(eeg=True)
channel_number, sample_number = raw._data.shape
channel_names = raw.ch_names
sampling_rate = int(raw.info['sfreq'])
session_time = sample_number/sampling_rate



from mir_eval.separation import bss_eval_sources
import pyroomacoustics as pra
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

# Callback function to monitor the convergence of the algorithm
# def convergence_callback(Y):
#     global SDR, SIR
#     y = pra.transform.stft.synthesis(Y, L, hop, win=win_s)
#     y = y[L - hop:, :].T
#     m = np.minimum(y.shape[1], ref.shape[1])
#     sdr, sir, sar, perm = bss_eval_sources(ref[:,:m], y[:,:m])
#     SDR.append(sdr)
#     SIR.append(sir)
def check(Y):
    print(Y.shape)
# Simulate
# The premix contains the signals before mixing at the microphones
# shape=(n_sources, n_mics, n_samples)
# separate_recordings = raw._data

# Mix down the recorded signals (n_mics, n_samples)
# i.e., just sum the array over the sources axis
# print(raw._data.shape)
# mics_signals = np.sum(separate_recordings, axis=0)
mics_signals = raw._data[:, :25600]

print(mics_signals.shape)

# STFT parameters
L = 2048
hop = L // 4
win_a = pra.hamming(L)
win_s = pra.transform.stft.compute_synthesis_window(win_a, hop)

# Observation vector in the STFT domain
X = pra.transform.stft.analysis(mics_signals.T, L, hop, win=win_a)
print(X.shape)
# Reference signal to calculate performance of BSS
# ref = separate_recordings[:, 0, :]
ref = mics_signals[0, :].reshape(1,1,mics_signals.shape[1])
# print(ref.shape)
SDR, SIR = [], []

# Run AuxIVA
Y = pra.bss.auxiva(X, n_iter=20)
# Y = pra.bss.ilrma(X, n_iter=30, n_components=23, proj_back=True, callback=convergence_callback)
# Ys = [0]*41
# for i in range(1,41):
#     try:
#         # Ys[i] = pra.bss.ilrma(X, n_iter=30)
#     except:
#         print(i, "fail")
#         Ys[i] = False
print(Y.shape)