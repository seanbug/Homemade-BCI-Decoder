import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt

data = np.load('data/mi_dataset_S01_20251121_194235.npz')
print(data.files)
for key in data.files:
    print(f"K is : {key}")
    print(f"Shape is : {data[key].shape}")
    print(f"Type is : {type(data[key])}")

print(f"First 10 labels:{data['labels'][:10]}")
print(f"First 10 conditions:{data['conditions'][:10]}")

# MI condition = label 1
# REST condition = label 0

fs = data['fs']
time_raw = data['timestamps']
time = []

for k in time_raw:
    time.append(k.split('T')[1])

print(f"{fs}")

eeg_data = data['eeg']
print(f"EEG data shape: {eeg_data.shape}")
print(f"EEG data type: {eeg_data[:10,:10]}")

#welch filter of eeg_data
f, Pxx = welch(eeg_data, fs=fs, window='hann', nperseg=48000, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')

print(f"frequencies: {f[:5]}")
print(f"PSD: {Pxx[:5]}")

print(f"Shape of frequencies (f): {f.shape}")
print(f"Shape of PSD (Pxx): {Pxx.shape}")
print(f"Bin width: {f[1] - f[0]}")
print(f"Frequency with maximum PSD: {f[np.argmax(Pxx.mean(axis=0))]}")

labels = data['labels']
psd_mi = Pxx[labels == 1]
psd_rest = Pxx[labels == 0]
print(f"Shape of PSD for MI condition: {psd_mi.shape}")
print(f"Shape of PSD for REST condition: {psd_rest.shape}")

fig, axes = plt.subplots(2, 1, figsize=(10, 12))
axes[0].axvspan(8, 30, alpha=0.15, color='gray')
axes[0].set_xlim(0, 100)
axes[0].set_yscale('log')
axes[0].plot(f, Pxx.mean(axis=0), label='PSD', color='black')
axes[0].set_title('Average PSD across all trials')
axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_ylabel('Power Spectral Density (a.u./Hz)')
axes[0].legend()

axes[1].set_xlim(0, 100)
axes[1].set_yscale('log')
axes[1].plot(f, psd_rest.mean(axis=0), label='Rest Condition PSD', color='blue')
axes[1].plot(f, psd_mi.mean(axis=0), label='MI Condition PSD', color='orange')
axes[1].set_title('Average PSD difference between MI vs. Rest trials')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Power Spectral Density (a.u./Hz)')
axes[1].axvspan(8, 30, alpha=0.15, color='gray')
axes[1].legend()
plt.show()