import pytest

try:
    import service.main as service_main  # type: ignore[import-not-found], akan works saat merge sama servce
    from fastapi.testclient import TestClient
except ImportError as e:
    pytest.skip(
        f"service.main tidak bisa diimpor ({e}) -- lihat docs/PROGRESS.md.",
        allow_module_level=True,
    )


class _FakeBaseline:
    def to_json_dict(self) -> dict:
        return {"schema_version": "1", "machine_label": "Fake Machine", "n_windows": 5}


class _FakeHealthCard:
    status = "NORMAL"
    z_score = 0.5
    calibration_quality = "baik"
    dominant_indicator = None
    disclaimer = "Alat bantu triase, bukan diagnosis mengikat -- tetap perlu inspeksi teknisi."


@pytest.fixture(autouse=True)
def _patch_model_layer(monkeypatch):
    monkeypatch.setattr(service_main, "warm_up", lambda: None)
    monkeypatch.setattr(service_main, "calibrate", lambda *a, **k: _FakeBaseline())
    monkeypatch.setattr(service_main, "inspect", lambda *a, **k: _FakeHealthCard())
    monkeypatch.setattr(
        service_main.MachineBaseline, "from_json_dict", classmethod(lambda cls, d: _FakeBaseline())
    )
    monkeypatch.setattr(service_main, "_model_ready", True)


@pytest.fixture
def client():
    with TestClient(service_main.app) as c:
        yield c


def _wav_upload():
    return {"audio": ("clip.wav", b"RIFF....WAVEfmt ", "audio/wav")}


def test_healthz_ok_when_model_ready(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_healthz_503_when_model_not_ready(client, monkeypatch):
    monkeypatch.setattr(service_main, "_model_ready", False)
    resp = client.get("/healthz")
    assert resp.status_code == 503


def test_calibrate_returns_baseline_json(client):
    resp = client.post("/v1/calibrate", files=_wav_upload(), data={"machine_label": "Blower Oven 1"})
    assert resp.status_code == 200
    assert resp.json()["machine_label"] == "Fake Machine"


def test_calibrate_missing_machine_label_returns_422(client):
    resp = client.post("/v1/calibrate", files=_wav_upload())
    assert resp.status_code == 422


def test_calibrate_missing_audio_returns_422(client):
    resp = client.post("/v1/calibrate", data={"machine_label": "Blower Oven 1"})
    assert resp.status_code == 422


def test_calibrate_value_error_from_ai_returns_400(client, monkeypatch):
    def _raise(*a, **k):
        raise ValueError("Audio tidak lolos quality gate.")

    monkeypatch.setattr(service_main, "calibrate", _raise)
    resp = client.post("/v1/calibrate", files=_wav_upload(), data={"machine_label": "Blower Oven 1"})
    assert resp.status_code == 400


def test_inspect_returns_health_card_contract(client):
    resp = client.post(
        "/v1/inspect", files=_wav_upload(), data={"baseline_json": '{"schema_version": "1"}'}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body.keys() == {"status", "z_score", "calibration_quality", "dominant_indicator", "disclaimer"}
    assert body["status"] == "NORMAL"
    assert body["disclaimer"]


def test_inspect_invalid_json_baseline_returns_400(client):
    resp = client.post("/v1/inspect", files=_wav_upload(), data={"baseline_json": "not valid json"})
    assert resp.status_code == 400


def test_inspect_missing_baseline_json_returns_422(client):
    resp = client.post("/v1/inspect", files=_wav_upload())
    assert resp.status_code == 422


def test_inspect_value_error_from_ai_returns_400(client, monkeypatch):
    def _raise(*a, **k):
        raise ValueError("Audio inspeksi tidak menghasilkan window valid.")

    monkeypatch.setattr(service_main, "inspect", _raise)
    resp = client.post(
        "/v1/inspect", files=_wav_upload(), data={"baseline_json": '{"schema_version": "1"}'}
    )
    assert resp.status_code == 400
