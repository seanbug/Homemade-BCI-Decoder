#loading as is wont work (psychopy requires older ver (I downloaded python 3.10.11)) so run from terminal from the psychopy env in scripts... .txt


import numpy as np
import sounddevice as sd
from psychopy import visual, core, event
import random, datetime, os, json

# eeg and psycoopy params
subjID = "Sean"
Fs = 48000
secondsPerTrial = 3.0
NindivExperiment = 40
breakTime = 1.0
slightDelayJitter = (1.0, 2.0)

#pseudorandomize
conditions = ["MI"] * NindivExperiment + ["REST"] * NindivExperiment
random.shuffle(conditions)

# psychopy params
win = visual.Window(fullscr=False, color="black", units="height")
fix = visual.TextStim(win, text="+", color="white", height=0.1)
ready = visual.TextStim(win, text="Get Ready", color="white", height=0.07)
mi_text = visual.TextStim(win, text="CLENCH", color="white", height=0.045)
rest_text = visual.TextStim(win, text="REST", color="white", height=0.10)

# load eeg mic stream
sd.default.samplerate = Fs
sd.default.channels = 1

# preloads data arrays into memory
eeg_segments = []
labels = []
timestamps = []

print(f"Starting MI dataset collection for {subjID}")
print(f"{len(conditions)} total trials ({NindivExperiment} MI, {NindivExperiment} REST)")

for i, cond in enumerate(conditions):
    if "escape" in event.getKeys():
        break

    # random fixation baseline before trial
    iti = random.uniform(*slightDelayJitter)
    fix.draw()
    win.flip()
    core.wait(iti)

    # get-ready cue
    ready.draw()
    win.flip()
    core.wait(breakTime)

    if cond == "MI":
        mi_text.draw()
    else:
        rest_text.draw()
    win.flip()

    n_samples = int(secondsPerTrial * Fs)
    eeg_chunk = sd.rec(n_samples, dtype="float32")
    sd.wait()

    eeg_segments.append(eeg_chunk.flatten())
    labels.append(1 if cond == "MI" else 0)
    timestamps.append(datetime.datetime.now().isoformat())

    print(f"Trial {i+1}/{len(conditions)} — {cond}")

win.close()

# saves eeg data as array with above labels
eeg_segments = np.array(eeg_segments)
labels = np.array(labels)

date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"mi_dataset_{subjID}_{date_str}.npz"
np.savez(
    filename,
    eeg=eeg_segments,
    labels=labels,
    fs=Fs,
    conditions=np.array(conditions),
    timestamps=np.array(timestamps),
    trial_sec=secondsPerTrial,
    subject=subjID
)

print(f"\nSaved dataset to {os.path.abspath(filename)}")
print("Recording session complete.")
