# End to end Motor Imagery BCI

A single-channel motor imagery BCI built end to end with an analog front end (AD620 instrumentation amp into TL084CN filter stages), PsychoPy acquisition task, offline classifier training, and a live decoder that emits a keystroke.

**Current state.** The software stages work and are verified against test signals. Acquisition is still wip: in the recorded dataset, 60 Hz and its harmonics account for roughly 75% of total power while the 8–30 Hz band holds about 0.01%, so there is almost nothing to train on. The likeliest candidate is the laptop line-in on the soundcard is used as the initial ADC which has several known issues (audio crackles on playback, mic capture, and over Bluetooth).

## Installation

To install, clone the reposity into your working directory and run `uv sync` to install the dependencies, or `uv sync --extra paradigm` if you also want to install the psycopy dependenci

## Files

The `src/homemade_bci_decoder/` directory has the following python scripts:

**`features.py`** is the shared feature extraction, imported by both training and inference. Resamples to 250 Hz, bandpass filters 8–30 Hz, returns log mean band power. Raises on degenerate input rather than substituting a fallback.

The `scripts` directory has the following python scripts:

**`Clench_motor_imagery_training.py`** is the PsychoPy task. 40 motor imagery (imagined left fist clench) and 40 rest trials, shuffled, 3 s each at 48 kHz, written to a timestamped `.npz`.

**`train_MI_classifier.py`** fits a logistic regression under stratified 5-fold CV, exports the model.

**`real_time_clench_identifier.py`** is the live decoder. Rolling 1 s buffer, feature extracted at a fixed update interval, keystroke emitted after several consecutive above-threshold windows as a debounce.

**`EEG_visualizer.py`** is a debug visualizer of the wave input and FFT.

The `tests` directory has the following python scripts:

**`test_features.py`** asserts distinct inputs give distinct features, in-band and out-of-band tones separate, and degenerate inputs raise.

## Future Directions

- Test the EEG acquisition hardware on another computer to see if the issue persists 
- Depending on that result, either swap in a different audio interface or utilize a dedicated biosignal ADC. Add a contralateral recording channel, since motor imagery ERD is lateralized, and generalize feature extraction to operate over a channel axis rather than a single trace.
- Collect a new dataset once acquisition is validated since the previously collected data was full of noisy data
- Split the single broadband feature into separate mu and beta bands, and add recording channels to make spatial filtering possible.
