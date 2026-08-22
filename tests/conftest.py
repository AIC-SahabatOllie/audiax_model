import numpy as np
import pytest


class DummyEncoder:
    "Embedding dari energi per pita frekuensi, sinyal beda frekuensi jadi beda embedding."

    def __init__(self, n_bands: int = 8) -> None:
        self.n_bands = n_bands

    def embed(self, windows: np.ndarray) -> np.ndarray:
        windows = np.asarray(windows, dtype=np.float32)
        n_freq = windows.shape[1] // 2 + 1
        edges = np.linspace(0, n_freq, self.n_bands + 1, dtype=int)
        out = np.zeros((windows.shape[0], self.n_bands), dtype=np.float32)
        for i, w in enumerate(windows):
            spec = np.abs(np.fft.rfft(w))
            for b in range(self.n_bands):
                lo, hi = edges[b], max(edges[b + 1], edges[b] + 1)
                out[i, b] = float(spec[lo:hi].mean())
        norm = np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
        return out / norm


class DummyAdapter:
    "Logic tester."

    def project(self, embeddings: np.ndarray) -> np.ndarray:
        return np.asarray(embeddings, dtype=np.float32)


@pytest.fixture
def dummy_encoder() -> DummyEncoder:
    return DummyEncoder()


@pytest.fixture
def dummy_adapter() -> DummyAdapter:
    return DummyAdapter()


@pytest.fixture
def sine_wave():
    "Bikin sinyal sine_wave(freq_hz, seconds) buat test audio"

    def _make(freq: float, seconds: float, sr: int = 16000, amplitude: float = 0.5) -> np.ndarray:
        t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
        return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float64)

    return _make
