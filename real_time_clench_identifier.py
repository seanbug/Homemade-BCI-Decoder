import numpy as np
import sounddevice as sd
from joblib import load
from pathlib import Path
import time
import keyboard

from features import extract_features

model_path = Path(__file__).resolve().parent / "mi_clf.joblib"
clf = load(model_path)

fs = 48000
secondsPerWindow = 1.0
iterWindow = 0.25
rawSampleN = int(fs * secondsPerWindow)

MIthreshold = 0.7
MIpersistence = 3
keypressBCI = "a"

buffer = np.zeros(rawSampleN, dtype=float)
consecutive_hits = 0

def callback(indata, frames, time_info, status):
    global buffer
    if status:
        print("Stream status:", status)
    new = indata[:, 0].astype(float)
    buffer = np.concatenate([buffer, new])[-rawSampleN:]

sd.default.samplerate = fs
sd.default.channels = 1

stream = sd.InputStream(callback=callback)
stream.start()

print("BCI On")

try:
    last_classify_time = time.time()
    while True:
        now = time.time()
        if now - last_classify_time >= iterWindow:
            last_classify_time = now
            window = buffer.copy()

            if not np.any(window):
                continue

            try:
                feats = extract_features(window, fs_in=fs).reshape(1, -1)
            except ValueError as e:
                print("Skipping window:", e)
                continue

            proba = clf.predict_proba(feats)[0]
            p_mi = proba[1]

            print(f"P(MI) = {p_mi:.2f}")

            if p_mi > MIthreshold:
                consecutive_hits += 1
            else:
                consecutive_hits = 0

            if consecutive_hits >= MIpersistence:
                print(f"MI detected, pressing {keypressBCI}")
                keyboard.press_and_release(keypressBCI)
                consecutive_hits = 0

        time.sleep(0.01)

except KeyboardInterrupt:
    print("Stopping…")
    stream.stop()
    stream.close()
