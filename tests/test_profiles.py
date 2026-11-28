"""Tests for the FaceScape-C degradation profiles (synthetic image, no real face data)."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from src.robustness import profiles as P

FACE_PX = 384.0
SEED = 1234
IDS = list(P.PROFILES)
SMALL = {"C01", "C08", "C09", "R1", "R2"}


@pytest.fixture(scope="module")
def img() -> np.ndarray:
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:768, 0:768]
    base = np.stack([xx / 3.0, yy / 3.0, (xx + yy) / 6.0], -1)
    base += (((xx // 32 + yy // 32) % 2) * 60)[..., None]
    base += rng.normal(0, 5, base.shape)  # fine texture, like skin detail
    base[300:340, 100:600] = 255
    base[500:520, :] = 10
    cv2.circle(base, (384, 384), 150, (200, 120, 90), 6)
    return np.clip(base, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def outputs(img):
    """Every (profile, severity) applied once with a fixed seed."""
    return {(p, s): P.apply(img, p, s, face_px=FACE_PX, seed=SEED)
            for p in IDS for s in P.SEVERITIES}


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def same_size(out: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if out.shape == ref.shape:
        return out
    return cv2.resize(out, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_CUBIC)


def test_registry():
    assert IDS == [f"C{i:02d}" for i in range(1, 11)] + ["R1", "R2", "R3", "R4"]
    assert P.SEVERITIES == (1, 2, 3)
    for spec in P.PROFILES.values():
        assert len(spec.severities) == 3
        assert spec.restorer_aligned == spec.id.startswith("R")
        assert spec.name and spec.scenario
    assert len(P.conditions()) == 43
    assert len(P.conditions(include_restorer=False)) == 31
    assert P.conditions()[0] == ("clean", 0)
    assert all(not P.PROFILES[p].restorer_aligned for p, _ in P.conditions(False)[1:])


def test_condition_seed_stable():
    assert P.condition_seed(1, "case", "C01", 2) == P.condition_seed(1, "case", "C01", 2)
    assert P.condition_seed(1, "case", "C01", 2) != P.condition_seed(1, "case", "C01", 3)
    assert P.condition_seed(1, "a", "C01", 2) != P.condition_seed(1, "b", "C01", 2)
    assert 0 <= P.condition_seed(1, "case", "C01", 1) < 2 ** 32


@pytest.mark.parametrize("pid", IDS)
def test_output_and_meta(outputs, img, pid):
    for s in P.SEVERITIES:
        out, meta = outputs[(pid, s)]
        assert out.dtype == np.uint8 and out.ndim == 3 and out.shape[2] == 3
        for key in ("profile_id", "severity", "params", "seed", "input_size", "output_size",
                    "clean_to_degraded"):
            assert key in meta
        assert meta["profile_id"] == pid and meta["severity"] == s and meta["seed"] == SEED
        assert meta["input_size"] == [768, 768]
        assert meta["output_size"] == [out.shape[1], out.shape[0]]
        assert (out.shape == img.shape) != (pid in SMALL)
        json.dumps(meta)  # JSON-serialisable


@pytest.mark.parametrize("pid", sorted(SMALL))
def test_transform_maps_corners(outputs, pid):
    for s in P.SEVERITIES:
        out, meta = outputs[(pid, s)]
        m = np.array(meta["clean_to_degraded"])
        assert m.shape == (3, 3)
        end = m @ np.array([768.0, 768.0, 1.0])
        assert np.allclose(end[:2], [out.shape[1], out.shape[0]], atol=0.5 + 1e-6 * 768)
        assert np.allclose(m @ np.array([0.0, 0.0, 1.0]), [0, 0, 1])


def test_transform_identity_for_same_size(outputs):
    for pid in IDS:
        if pid not in SMALL:
            assert np.allclose(outputs[(pid, 1)][1]["clean_to_degraded"], np.eye(3))


def test_clean_is_copy(img):
    out, meta = P.apply(img, "clean", 0, face_px=FACE_PX, seed=1)
    assert np.array_equal(out, img) and out is not img
    assert np.allclose(meta["clean_to_degraded"], np.eye(3))
    assert meta["output_size"] == meta["input_size"]


def test_input_not_modified(img):
    before = img.copy()
    for pid in ("C06", "C10", "R4"):
        P.apply(img, pid, 3, face_px=FACE_PX, seed=3)
    assert np.array_equal(before, img)


@pytest.mark.parametrize("pid", IDS)
def test_deterministic(img, outputs, pid):
    again, meta2 = P.apply(img, pid, 2, face_px=FACE_PX, seed=SEED)
    first, meta1 = outputs[(pid, 2)]
    assert again.tobytes() == first.tobytes()
    assert meta1 == meta2


@pytest.mark.parametrize("pid", ["C03", "C04", "C05", "C09", "R1", "R2", "R3", "R4"])
def test_seed_changes_stochastic(img, outputs, pid):
    other, _ = P.apply(img, pid, 2, face_px=FACE_PX, seed=SEED + 1)
    assert other.shape != outputs[(pid, 2)][0].shape or other.tobytes() != outputs[(pid, 2)][0].tobytes()


@pytest.mark.parametrize("pid", IDS)
def test_severity_monotonic(outputs, img, pid):
    scores = [psnr(same_size(outputs[(pid, s)][0], img), img) for s in P.SEVERITIES]
    assert scores[0] > scores[1] > scores[2], scores


@pytest.mark.parametrize("pid", ["C02", "C03", "C10", "R3"])
def test_scales_with_face_px(img, pid):
    """Same relative strength at half the face size: params scale, output size stays."""
    half = cv2.resize(img, (384, 384), interpolation=cv2.INTER_AREA)
    a, ma = P.apply(img, pid, 2, face_px=384.0, seed=7)
    b, mb = P.apply(half, pid, 2, face_px=192.0, seed=7)
    a_small = cv2.resize(a, (384, 384), interpolation=cv2.INTER_AREA)
    assert psnr(a_small, b) > 22.0


def high_freq(x: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())


def test_beauty_filter_smooths(outputs, img):
    base = high_freq(img)
    energies = [high_freq(outputs[("C10", s)][0]) for s in P.SEVERITIES]
    assert energies[0] < base and energies[0] > energies[1] > energies[2]


def test_overexposure_brightens(outputs, img):
    base = img.mean()
    means = [outputs[("C06", s)][0].mean() for s in P.SEVERITIES]
    assert all(m > base for m in means) and means[0] < means[2]


def test_low_light_params_and_brightness(outputs, img):
    for s in P.SEVERITIES:
        out, meta = outputs[("C05", s)]
        p = meta["params"]
        assert p["exposure"] < 1.0 and p["gain"] > 1.0
        assert abs(p["exposure"] * p["gain"] - p["target_brightness"]) < 1e-9
        assert out.mean() < img.mean()
    exps = [outputs[("C05", s)][1]["params"]["exposure"] for s in P.SEVERITIES]
    assert exps == [0.35, 0.2, 0.1]


def test_c08_clamp_keeps_min_size():
    tiny = np.full((40, 40, 3), 128, np.uint8)
    out, meta = P.apply(tiny, "C08", 3, face_px=20.0, seed=0)
    assert meta["params"]["round_factors"][1] == 1.0  # second round clamped
    assert min(out.shape[:2]) >= 32


def test_realesrgan_quarter_size(outputs, img):
    for s in P.SEVERITIES:
        out, meta = outputs[("R2", s)]
        assert out.shape == (192, 192, 3)
        assert meta["output_size"] == [192, 192]
        assert np.allclose(meta["clean_to_degraded"], np.diag([0.25, 0.25, 1.0]))
        assert meta["params"]["severity_rank"] == [16, 32, 47][s - 1]


def test_downsizing_profiles_dont_upsample(outputs):
    assert outputs[("C01", 3)][0].shape[:2] == (round(768 * 16 / 384),) * 2
    assert outputs[("C01", 1)][0].shape[:2] == (128, 128)


def test_haze_transmission_at_face(outputs):
    for s, t in zip(P.SEVERITIES, (0.75, 0.55, 0.35)):
        got = outputs[("R3", s)][1]["params"]["face_transmission_realised"]
        assert abs(got - t) < 0.08


def test_invalid_inputs(img):
    with pytest.raises(ValueError):
        P.apply(img, "nope", 1, face_px=FACE_PX, seed=0)
    with pytest.raises(ValueError):
        P.apply(img, "C01", 4, face_px=FACE_PX, seed=0)
    with pytest.raises(ValueError):
        P.apply(img.astype(np.float32), "C01", 1, face_px=FACE_PX, seed=0)
