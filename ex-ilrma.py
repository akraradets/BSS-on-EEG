import time
import numpy as np
from scipy.io import wavfile

from mir_eval.separation import bss_eval_sources
import pyroomacoustics as pra

wav_files = [
        ['examples/1-guitar.wav'],
        ['examples/1-vocal.wav']
        ]

## Prepare one-shot STFT
L = 2048
hop = L // 2
win_a = pra.hann(L)
win_s = pra.transform.stft.compute_synthesis_window(win_a, hop)

# get signals
fs = 44100
sig_L = np.array(wavfile.read('examples/2-L.wav')[1],dtype='float32')
sig_M = np.array(wavfile.read('examples/2-M.wav')[1],dtype='float32')
sig_R = np.array(wavfile.read('examples/2-R.wav')[1],dtype='float32')
# sig_pad = np.concatenate([sig_M,np.zeros(sig_L.shape[0] - sig_M.shape[0])])
# print(sig_L,sig_)
signals = np.array([sig_L,sig_M,sig_R])
# (2, 1881143)

mics_signals = signals
## STFT ANALYSIS
# (n_samples, n_channels)
X = pra.transform.stft.analysis(mics_signals.T, L, hop, win=win_a)
# print(X.shape)
# Run ILRMA
Y = pra.bss.ilrma(X, n_iter=30, n_components=2, proj_back=True) #callback=convergence_callback)
# print(Y.shape)
y = pra.transform.stft.synthesis(Y, L, hop, win=win_s)
print(y.shape)
for i, sig in enumerate(y.T):
    print(sig)
    wavfile.write('bss_iva_source{}.wav'.format(i+1), 44100,
            pra.normalize(sig, bits=16).astype(np.int16))
# for i, sig in enumerate(y):
    # wavfile.write('bss_iva_source{}.wav'.format(i+1), 44100,
    #         pra.normalize(sig, bits=16).astype(np.int16))