"""Test calibrate() -> inspect() pakai encoder/adapter palsu dari conftest.py.
Part ini di-skip kalau ai/ belum ada isinya,
otomatis nyala lagi begitu ai.calibrate/inspect/MachineBaseline tersedia.
"""

import numpy as np
import pytest
import soundfile as sf

ai = pytest.importorskip("ai")
if not all(hasattr(ai, name) for name in ("calibrate", "inspect", "MachineBaseline")):
    pytest.skip(
        "ai/ belum menyediakan calibrate()/inspect()/MachineBaseline -- lihat docs/PROGRESS.md.",
        allow_module_level=True,
    )


def _write_wav(path, samples: np.ndarray, sr: int = 16000) -> str:
    sf.write(str(path), samples.astype(np.float32), sr)
    return str(path)


def test_calibrate_then_inspect_returns_valid_health_card(tmp_path, dummy_encoder, dummy_adapter, sine_wave):
    calib_path = _write_wav(tmp_path / "calib.wav", sine_wave(440.0, 30.0))
    baseline = ai.calibrate(
        calib_path,
        machine_label="Test Machine",
        _encoder_override=dummy_encoder,
        _adapter_override=dummy_adapter,
    )
    assert baseline.n_windows > 0
    assert baseline.calibration_quality in ("baik", "rendah")
    assert baseline.embeddings.shape[0] == baseline.n_windows

    test_path = _write_wav(tmp_path / "test.wav", sine_wave(440.0, 3.0))
    card = ai.inspect(
        test_path,
        baseline,
        _encoder_override=dummy_encoder,
        _adapter_override=dummy_adapter,
    )
    assert card.status in ("NORMAL", "WARNING", "CRITICAL", "KALIBRASI_KURANG")
    assert card.disclaimer 
    assert card.dominant_indicator is None  


def test_baseline_roundtrips_through_json(tmp_path, dummy_encoder, dummy_adapter, sine_wave):
    calib_path = _write_wav(tmp_path / "calib.wav", sine_wave(440.0, 30.0))
    baseline = ai.calibrate(
        calib_path,
        machine_label="Test Machine",
        _encoder_override=dummy_encoder,
        _adapter_override=dummy_adapter,
    )

    restored = ai.MachineBaseline.from_json_dict(baseline.to_json_dict())

    assert restored.machine_label == baseline.machine_label
    assert restored.model_fingerprint == baseline.model_fingerprint
    assert restored.n_windows == baseline.n_windows
    assert restored.calibration_quality == baseline.calibration_quality
    np.testing.assert_allclose(restored.embeddings, baseline.embeddings, atol=1e-2)


def test_shifted_signal_scores_worse_than_matching_signal(tmp_path, dummy_encoder, dummy_adapter, sine_wave):
    calib_path = _write_wav(tmp_path / "calib.wav", sine_wave(440.0, 30.0))
    baseline = ai.calibrate(
        calib_path,
        machine_label="Test Machine",
        _encoder_override=dummy_encoder,
        _adapter_override=dummy_adapter,
    )

    matching_path = _write_wav(tmp_path / "matching.wav", sine_wave(440.0, 3.0))
    shifted_path = _write_wav(tmp_path / "shifted.wav", sine_wave(3000.0, 3.0))

    card_matching = ai.inspect(
        matching_path, baseline, _encoder_override=dummy_encoder, _adapter_override=dummy_adapter
    )
    card_shifted = ai.inspect(
        shifted_path, baseline, _encoder_override=dummy_encoder, _adapter_override=dummy_adapter
    )

    assert card_shifted.z_score > card_matching.z_score


def test_calibrate_rejects_silent_audio(tmp_path, dummy_encoder, dummy_adapter):
    silence = np.zeros(16000 * 5, dtype=np.float64)
    calib_path = _write_wav(tmp_path / "silence.wav", silence)
    with pytest.raises(ValueError):
        ai.calibrate(
            calib_path,
            machine_label="Test Machine",
            _encoder_override=dummy_encoder,
            _adapter_override=dummy_adapter,
        )


def test_inspect_rejects_baseline_from_different_model_fingerprint(
    tmp_path, dummy_encoder, dummy_adapter, sine_wave
):
    calib_path = _write_wav(tmp_path / "calib.wav", sine_wave(440.0, 30.0))
    baseline = ai.calibrate(
        calib_path,
        machine_label="Test Machine",
        _encoder_override=dummy_encoder,
        _adapter_override=dummy_adapter,
    )
    baseline.model_fingerprint = "stale-fingerprint-from-old-model"

    test_path = _write_wav(tmp_path / "test.wav", sine_wave(440.0, 3.0))
    card = ai.inspect(
        test_path, baseline, _encoder_override=dummy_encoder, _adapter_override=dummy_adapter
    )
    assert card.status == "KALIBRASI_KURANG"
