# https://github.com/raphaelvallat/yasa/blob/master/notebooks/08_bandpower.ipynb
import mne
raw = mne.io.read_raw_edf("chb01_01.edf",preload=True, verbose=0)
raw.pick_types(eeg=True)
channel_number, sample_number = raw._data.shape
channel_names = raw.ch_names
sampling_rate = int(raw.info['sfreq'])
session_time = sample_number/sampling_rate

# 0 V
# 1 uV
unit = 1
# set window to be first 10 second
window = 10
# set channels to see
v_channel = {0,1}

# raw._data.shape 
#   (23, 921600)
# raw.ch_names 
#   ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 
#   'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2', 
#   'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8-1']

# # Apply a bandpass filter between 0.5 - 45 Hz
# raw.filter(0.5, 45)

# # Extract the data and convert from V to uV
# data = raw._data * 1e6
# sf = raw.info['sfreq']
# chan = raw.ch_names

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
# plot raw data
# raw._data
fig, ax = plt.subplots(len(v_channel))
for index in v_channel:
    data = raw._data[index]
    if(unit): data = data * 1e6
    # NFFT=256, Fs=sampling_rate, vmin=-20, vmax=30
    ax[index].specgram(data.astype(np.float32), Fs=sampling_rate, NFFT=256)
    
    # ax[index].plot(range(0,window*sampling_rate), data[:window*sampling_rate])
    # ax[index].set_ylabel(channel_names[index] + (" (uV)"*unit))
plt.show()


# print("==== STFT ====")
# # # https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.transform.stft.html
# import pyroomacoustics as pra
# # parameters
# h_len = sampling_rate
# h = np.ones(h_len)
# h /= np.linalg.norm(h)

# block_size = 512 - h_len + 1  # make sure the FFT size is a power of 2
# hop = block_size // 2  # half overlap
# slide_window = pra.hann(
#     block_size, flag="asymmetric", length="full"
# )  # analysis slide_window (no synthesis slide_window)
# # Create the STFT object
# stft = pra.transform.STFT(
#     block_size, hop=hop, analysis_window=None, channels=23, streaming=True
# )
# # set the filter and the appropriate amount of zero padding (back)
# if h_len > 1:
#     stft.set_filter(h, zb=h.shape[0] - 1)

# # collect the processed blocks
# processed_audio = np.zeros((window*sampling_rate,23))

# # process the signals while full blocks are available
# n = 0
# audio = raw._data.T
# print(audio.shape, processed_audio.shape)
# while window*sampling_rate - n > hop:
#     print(n)
#     # go to frequency domain
#     stft.analysis( audio[n : n + hop, :] )

#     stft.process()  # apply the filter

#     # copy processed block in the output buffer
#     # print(stft.synthesis())
#     processed_audio[n : n + hop] = stft.synthesis()

#     n += hop


# plt.figure()
# # plt.subplot(2, 1, 1)
# plt.specgram(audio[: n - hop,1].astype(np.float32), NFFT=256, Fs=sampling_rate, vmin=-20, vmax=30)
# plt.show()


# def convergence_callback(Y):
#     global SDR, SIR
#     from mir_eval.separation import bss_eval_sources
#     ref = np.moveaxis(separate_recordings, 1, 2)
#     y = pra.transform.stft.synthesis(Y, L, hop, win=win_s)
#     y = y[L-hop: , :].T
#     m = np.minimum(y.shape[1], ref.shape[1])
#     sdr, sir, sar, perm = bss_eval_sources(ref[:, :m, 0], y[:, :m])
#     SDR.append(sdr)
#     SIR.append(sir)


# # filter to apply
# h_len = 99
# h = np.ones(h_len)
# h /= np.linalg.norm(h)

# # parameters
# block_size = 512 - h_len + 1  # make sure the FFT size is a power of 2
# hop = block_size // 2  # half overlap
# window = pra.hann(
#     block_size, flag="asymmetric", length="full"
# )  # analysis window (no synthesis window)

# stft = pra.transform.STFT(
#     block_size, hop=hop, analysis_window=window, channels=23, streaming=True
# )
# Y = pra.bss.ilrma(X, n_iter=30, n_components=23, proj_back=True,
#                           callback=convergence_callback)