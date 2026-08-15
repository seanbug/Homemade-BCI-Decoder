import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

fs = 48000
secondWindow = 2.0
Nsamp = int(fs * secondWindow)

buffer = np.zeros(Nsamp, dtype=float)

def callback(indata, frames, time_info, status):
    global buffer
    if status:
        print("Stream status:", status)
    new = indata[:, 0].astype(float)
    buffer = np.concatenate([buffer, new])[-Nsamp:]

plt.ion()
fig, (ax_time, ax_fft) = plt.subplots(2, 1, figsize=(10, 6))
plt.tight_layout()

t = np.linspace(0, secondWindow, Nsamp)
line_time, = ax_time.plot(t, np.zeros_like(t))
ax_time.set_ylim(-1.0, 1.0)
ax_time.set_title("EEG Signal")
ax_time.set_xlabel("Time (s)")
ax_time.set_ylabel("Normalized Amplitude")

#fft plot
freqs = np.fft.rfftfreq(Nsamp, d=1.0/fs)
line_fft, = ax_fft.plot(freqs, np.zeros_like(freqs))
ax_fft.set_xlim(0, 60)
ax_fft.set_title("EEG Frequency Spectrum (FFT)")
ax_fft.set_xlabel("Frequency (Hz)")
ax_fft.set_ylabel("Power (dB)")


stream = sd.InputStream(
    channels=1,
    samplerate=fs,
    callback=callback
)
stream.start()

print("Streaming")

try:
    while True:
        sig = buffer.copy()

        raw_sig = sig
        line_time.set_ydata(raw_sig)

        fft_power = np.abs(np.fft.rfft(raw_sig))**2
        fft_db = 10 * np.log10(fft_power + 1e-12)

        line_fft.set_ydata(fft_db)
        ax_fft.set_ylim(fft_db.min() - 3, fft_db.max() + 3)

        dom_idx = np.argmax(fft_power)
        dom_freq = freqs[dom_idx]
        print(f"Dominant freq: {dom_freq:.1f} Hz")

        plt.pause(0.01)

except KeyboardInterrupt:
    print("Stop")
    stream.stop()
    stream.close()
