import numpy as np

from features import extract_features

fs = 48000


def test_distinguishes_noise_from_tone():
    rng = np.random.default_rng(0)
    noise = extract_features(rng.normal(0, 1, fs), fs_in=fs)
    tone = extract_features(np.sin(2 * np.pi * 10 * np.arange(fs) / fs), fs_in=fs)

    assert np.all(np.isfinite(noise)), noise
    assert np.all(np.isfinite(tone)), tone
    assert not np.isclose(noise[0], tone[0]), (noise[0], tone[0])


def test_rejects_dead_signal():
    try:
        extract_features(np.zeros(fs), fs_in=fs)
    except ValueError:
        pass
    else:
        raise AssertionError("silence should raise, not return a fallback feature")


if __name__ == "__main__":
    test_distinguishes_noise_from_tone()
    test_rejects_dead_signal()

    rng = np.random.default_rng(0)
    noise = extract_features(rng.normal(0, 1, fs), fs_in=fs)[0]
    tone = extract_features(np.sin(2 * np.pi * 10 * np.arange(fs) / fs), fs_in=fs)[0]
    print(f"white noise    : {noise:.4f}")
    print(f"10 Hz sine     : {tone:.4f}")
    print(f"separation     : {abs(tone - noise):.4f} nats")
    print("OK")
