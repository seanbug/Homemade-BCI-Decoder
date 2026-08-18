import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
# MI condition = label 1
# REST condition = label 0
data = np.load('data/mi_dataset_S01_20251121_194235.npz')
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

labels = data['labels']

thru_30 = (f >= 8) & (f <= 30)
print(f"Shape of frequencies (f): {f.shape}")
print(f"Shape of interest bands (0-30 Hz): {thru_30.shape}")
print(f"Shape of Pxx: {Pxx.shape}")
psd30_mi_band   = Pxx[labels == 1][:, thru_30]
psd30_rest_band = Pxx[labels == 0][:, thru_30]
print(f"Shape of PSD for MI condition: {psd30_mi_band.shape}")
print(f"Shape of PSD for REST condition: {psd30_rest_band.shape}")

from numpy import trapezoid

bp_mi = trapezoid(psd30_mi_band, f[thru_30], axis=1)
print(f"Shape of band power for MI condition: {bp_mi.shape}")
bp_rest = trapezoid(psd30_rest_band, f[thru_30], axis=1)
print(f"Shape of band power for REST condition: {bp_rest.shape}")   

bp_mi_log = np.log10(bp_mi)
bp_rest_log = np.log10(bp_rest)

from scipy.stats import ttest_ind

t, p = ttest_ind(bp_mi_log, bp_rest_log)

bp_mi_log.std(ddof=1)
mi_se = bp_mi_log.std(ddof=1) / np.sqrt(len(bp_mi_log))
bp_rest_log.std(ddof=1)
rest_se = bp_rest_log.std(ddof=1) / np.sqrt(len(bp_rest_log))

print(f"T-test results: t={t:.3f}, p={p:.3e}.")
print(f"MI band power: mean={bp_mi_log.mean():.3f}, std={bp_mi_log.std(ddof=1):.3f}, se={mi_se:.3f}")
print(f"REST band power: mean={bp_rest_log.mean():.3f}, std={bp_rest_log.std(ddof=1):.3f}, se={rest_se:.3f}")

fig, ax = plt.subplots(figsize=(6, 5))

rng = np.random.default_rng(0)
x_mi = np.ones(len(bp_mi_log)) + rng.normal(0, 0.04, len(bp_mi_log))
ax.scatter(x_mi, bp_mi_log,color='orange', alpha=0.5, label='MI trials')
x_rest = np.zeros(len(bp_rest_log)) + rng.normal(0, 0.04, len(bp_rest_log))
ax.scatter(x_rest, bp_rest_log, color='blue', alpha=0.5, label='Rest trials')
ax.set_ylabel('log10 band power, 8-30 Hz (a.u.)')
ax.hlines(bp_rest_log.mean(), -0.2, 0.2, colors='k', linewidth=2)
ax.hlines(bp_mi_log.mean(), 0.8, 1.2, colors='k', linewidth=2)
ax.errorbar(0, bp_rest_log.mean(), yerr=rest_se, color='k', linewidth=2, capsize=5)
ax.errorbar(1, bp_mi_log.mean(), yerr=mi_se, color='k', linewidth=2, capsize=5)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Rest', 'MI'])
ax.set_xlim(-0.5, 1.5)
ax.set_title(f"8-30 Hz band power by condition (t={t:.2f}, p={p:.2f})")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "band_power_by_condition.png", dpi=150, bbox_inches="tight")
print(f"Saved to {FIG_DIR / 'band_power_by_condition.png'}")


psd_mean = Pxx.mean(axis=0)
bw = f[1] - f[0]
total = psd_mean.sum() * bw
inband = psd_mean[(f >= 8) & (f <= 30)].sum() * bw

harm = np.zeros_like(f, dtype=bool)
for h in range(60, int(f[-1]), 60):
    harm |= (f >= h - 2) & (f <= h + 2)
harm_power = psd_mean[harm].sum() * bw

print(f"in-band {100*inband/total:.4f}%  harmonics {100*harm_power/total:.2f}%")