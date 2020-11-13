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


# read multichannel wav file
fs = sampling_rate
# raw._data.shape => (23, 921600)
# audio.shape == (nsamples, nchannels)
audio = raw._data.T

# STFT analysis parameters
# https://www.kfs.oeaw.ac.at/manual/3.8/html/userguide/464.htm
# RT60 https://www.roomeqwizard.com/help/help_en-GB/html/graph_rt60.html#:~:text=RT60%20is%20a%20measure%20of,directions%20at%20the%20same%20level.
# fft_size = 4096  # `fft_size / fs` should be ~RT60
fft_size = 45
hop = fft_size // 2  # half-overlap
win_a = pra.hann(fft_size)  # analysis window
# optimal synthesis window
win_s = pra.transform.compute_synthesis_window(win_a, hop)

# STFT
# X.shape == (nframes, nfrequencies, nchannels)
X = pra.transform.analysis(audio, fft_size, hop, win=win_a)

# Separation
Y = pra.bss.auxiva(X, n_iter=20)

# iSTFT (introduces an offset of `hop` samples)
# y contains the time domain separated signals
# y.shape == (new_nsamples, nchannels)
y = pra.transform.synthesis(Y, fft_size, hop, win=win_s)