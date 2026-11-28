"""FaceScape-C degradation benchmark: 14 profiles x 3 severities for real face crops.

Every profile is a pure function of (image, severity params, face_px, rng); randomness comes
only from ``np.random.default_rng(seed)``, so a (profile, severity, seed) triple is reproducible
and nothing touches global RNG state. Parameters that depend on size scale with ``face_px``
(jaw-to-jaw width in pixels of the crop; reference 384 px on a 768 px frame).

Profiles (S1 / S2 / S3)
-----------------------
C01 low_resolution      face downsampled to 64 / 32 / 16 px wide (INTER_AREA), output small
C02 defocus             disk blur, radius 0.012 / 0.024 / 0.048 * face_px (linear light)
C03 motion_blur         linear blur, length 0.03 / 0.06 / 0.10 * face_px, random angle
C04 sensor_noise        shot (400/120/40 e- at white) + read (2/5/10 /255) + chroma noise
C05 low_light           exposure 0.35/0.2/0.1, noise, gain back to 0.6/0.5/0.4, JPEG q85
C06 overexposure        flash gain 1.6/2.2/3.0, clip, bloom, contrast loss 0.1/0.2/0.3
C07 jpeg                JPEG quality 30 / 15 / 5
C08 messaging_chain     two rounds of (downscale 0.5/0.35/0.25, JPEG q70/50/30), output small
C09 cheap_webcam        downscale 0.5/0.35/0.25, chromatic aberration, vignette, noise,
                        over-sharpening, JPEG q75/55/35; output small
C10 beauty_filter       bilateral smoothing x1/2/3 passes, brighten +4/8/12 %, warm tint
R1  bfr_classical       CodeFormer blind-face-restoration degradation at 25/50/75th pct
R2  realesrgan_second_order  Real-ESRGAN x4plus second-order pipeline, 25/50/75th pct by PSNR
R3  haze                atmospheric scattering, face transmission 0.75 / 0.55 / 0.35
R4  shadow              hard-edged cast shadow, factor 0.65 / 0.45 / 0.30 (linear), blue tint

``R*`` are restorer-aligned: they mimic the training degradations of face/image restorers and
serve as a control. Profiles that model a low-resolution capture return the smaller image
(never upsampled back); ``meta["clean_to_degraded"]`` maps clean pixel coordinates (continuous,
pixel i covering [i, i+1)) to output coordinates.

The kernel samplers and noise generators of R2 are numpy ports of BasicSR
(``basicsr/data/degradations.py``, Apache-2.0, Copyright XPixelGroup) and the Real-ESRGAN
training pipeline (``realesrgan/models/realesrgan_model.py``, ``realesrgan/data/
realesrgan_dataset.py``, BSD-3-Clause, Xintao Wang). Constants come from
``options/train_realesrgan_x4plus.yml``.
"""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image
from scipy import special

REF_FACE_PX = 384.0
SEVERITIES = (1, 2, 3)
_CODEFORMER_FACE_PX = 300.0  # ~0.6 * 512, see R1

# --- Real-ESRGAN x4plus constants (third_party/Real-ESRGAN/options/train_realesrgan_x4plus.yml) ---
_R2 = {
    "scale": 4,
    "resize_prob": (0.2, 0.7, 0.1), "resize_range": (0.15, 1.5),
    "gaussian_noise_prob": 0.5, "noise_range": (1, 30), "poisson_scale_range": (0.05, 3),
    "gray_noise_prob": 0.4, "jpeg_range": (30, 95),
    "second_blur_prob": 0.8,
    "resize_prob2": (0.3, 0.4, 0.3), "resize_range2": (0.3, 1.2),
    "gaussian_noise_prob2": 0.5, "noise_range2": (1, 25), "poisson_scale_range2": (0.05, 2.5),
    "gray_noise_prob2": 0.4, "jpeg_range2": (30, 95),
    "kernel_list": ("iso", "aniso", "generalized_iso", "generalized_aniso", "plateau_iso",
                    "plateau_aniso"),
    "kernel_prob": (0.45, 0.25, 0.12, 0.03, 0.12, 0.03),
    "sinc_prob": 0.1, "blur_sigma": (0.2, 3), "betag_range": (0.5, 4), "betap_range": (1, 2),
    "sinc_prob2": 0.1, "blur_sigma2": (0.2, 1.5),
    "final_sinc_prob": 0.8,
    "kernel_sizes": (7, 9, 11, 13, 15, 17, 19, 21),
    "n_candidates": 64, "proxy_size": 256, "quantiles": (0.25, 0.5, 0.75),
}


@dataclass(frozen=True)
class ProfileSpec:
    id: str
    name: str
    scenario: str
    severities: tuple
    restorer_aligned: bool


# --------------------------------------------------------------------------- colour helpers
def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """sRGB in [0, 1] -> linear light (exact piecewise transfer function)."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(
        np.float32)


def _to_lin(img: np.ndarray) -> np.ndarray:
    return srgb_to_linear(img.astype(np.float32) / 255.0)


def _to_u8(lin: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(linear_to_srgb(lin) * 255.0), 0, 255).astype(np.uint8)


def _f_to_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)


def jpeg_roundtrip(img: np.ndarray, quality: int) -> np.ndarray:
    """JPEG encode/decode with Pillow (default 4:2:0 chroma subsampling)."""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=int(quality))
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), dtype=np.uint8)


def _scaled_size(img: np.ndarray, factor: float) -> tuple[int, int]:
    h, w = img.shape[:2]
    return max(1, int(round(w * factor))), max(1, int(round(h * factor)))


def _filter(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return cv2.filter2D(img, -1, kernel.astype(np.float32), borderType=cv2.BORDER_REFLECT)


# --------------------------------------------------------------------------- blur kernels
def disk_kernel(radius: float) -> np.ndarray:
    """Normalised pillbox kernel with an anti-aliased (1 px linear ramp) edge."""
    half = int(math.ceil(radius)) + 2
    ax = np.arange(-half, half + 1, dtype=np.float32)
    dist = np.hypot(*np.meshgrid(ax, ax))
    k = np.clip(radius + 0.5 - dist, 0.0, 1.0)
    return k / k.sum()


def motion_kernel(length: float, angle_deg: float) -> np.ndarray:
    """Normalised line kernel of ``length`` px at ``angle_deg``, bilinearly splatted."""
    half = int(math.ceil(length / 2)) + 2
    size = 2 * half + 1
    k = np.zeros((size, size), np.float32)
    n = max(int(length * 4), 8)
    t = np.linspace(-length / 2, length / 2, n)
    a = math.radians(angle_deg)
    xs, ys = half + t * math.cos(a), half - t * math.sin(a)
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    fx, fy = xs - x0, ys - y0
    for dx, dy, wgt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                        (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        np.add.at(k, (y0 + dy, x0 + dx), wgt)
    return k / k.sum()


# --------------------------------------------------------------------------- noise helpers
def _lowfreq_chroma(shape: tuple, sigma: float, blur_sigma: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Per-channel chroma noise (zero luma), blurred, rescaled to std ``sigma``."""
    n = rng.standard_normal(shape).astype(np.float32)
    n -= n.mean(axis=2, keepdims=True)
    n = cv2.GaussianBlur(n, (0, 0), max(blur_sigma, 0.3))
    std = float(n.std())
    return n * (sigma / std) if std > 0 else n


def _sensor_noise(lin: np.ndarray, photons: float, read_sigma: float, chroma_blur: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Poisson shot + Gaussian read + low-frequency chroma noise on linear [0, 1] data."""
    shot = rng.poisson(np.clip(lin, 0, None) * photons).astype(np.float32) / photons
    read = rng.standard_normal(lin.shape).astype(np.float32) * read_sigma
    chroma = _lowfreq_chroma(lin.shape, 0.5 * read_sigma, chroma_blur, rng)
    return np.clip(shot + read + chroma, 0.0, 1.0)


# --------------------------------------------------------------------------- C-profiles
def _c01(img, p, face_px, rng):
    f = p["face_target_px"] / face_px
    out = cv2.resize(img, _scaled_size(img, f), interpolation=cv2.INTER_AREA)
    return out, {"factor": f}


def _c02(img, p, face_px, rng):
    r = face_px * p["radius_frac"]
    out = _to_u8(_filter(_to_lin(img), disk_kernel(r)))
    return out, {"radius_px": r}


def _c03(img, p, face_px, rng):
    length = face_px * p["length_frac"]
    angle = float(rng.uniform(0.0, 180.0))
    out = _to_u8(_filter(_to_lin(img), motion_kernel(length, angle)))
    return out, {"length_px": length, "angle_deg": angle}


def _c04(img, p, face_px, rng):
    blur = face_px * 0.01
    out = _to_u8(_sensor_noise(_to_lin(img), p["photons"], p["read_sigma"], blur, rng))
    return out, {"chroma_sigma": 0.5 * p["read_sigma"], "chroma_blur_px": blur}


def _c05(img, p, face_px, rng):
    blur = face_px * 0.01
    dark = _to_lin(img) * p["exposure"]
    noisy = _sensor_noise(dark, p["photons"], p["read_sigma"], blur, rng)
    gain = p["target_brightness"] / p["exposure"]
    out = jpeg_roundtrip(_to_u8(noisy * gain), p["jpeg_quality"])
    return out, {"gain": gain, "chroma_blur_px": blur}


def _c06(img, p, face_px, rng):
    g = _to_lin(img) * p["gain"]
    clipped = (g.max(axis=2, keepdims=True) > 1.0).astype(np.float32)
    bloom = cv2.GaussianBlur(clipped[..., 0], (0, 0), face_px * 0.03)[..., None]
    lin = np.clip(np.clip(g, 0, 1) + p["bloom"] * bloom, 0, 1)
    s = linear_to_srgb(lin)
    mean = float(s.mean())
    s = s + p["contrast_loss"] * (mean - s)
    return _f_to_u8(s), {"bloom_sigma_px": face_px * 0.03, "clipped_fraction": float(clipped.mean())}


def _c07(img, p, face_px, rng):
    return jpeg_roundtrip(img, p["quality"]), {}


def _c08(img, p, face_px, rng):
    out, factors = img, []
    for _ in range(2):
        f = max(p["factor"], 32.0 / min(out.shape[:2]))  # clamp: keep >= 32 px
        f = min(f, 1.0)
        out = cv2.resize(out, _scaled_size(out, f), interpolation=cv2.INTER_AREA)
        out = jpeg_roundtrip(out, p["quality"])
        factors.append(f)
    return out, {"round_factors": factors, "total_factor": float(np.prod(factors))}


def _chromatic_aberration(img: np.ndarray, frac: float) -> np.ndarray:
    """Radially magnify R by (1+2*frac) and shrink B by (1-2*frac) about the centre,
    giving a +-frac*width shift at the left/right edge midpoints."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    out = img.copy()
    for ch, s in ((0, 1 + 2 * frac), (2, 1 - 2 * frac)):
        mx, my = cx + (xx - cx) / s, cy + (yy - cy) / s
        out[..., ch] = cv2.remap(img[..., ch], mx, my, cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
    return out


def _vignette(lin: np.ndarray, strength: float) -> np.ndarray:
    h, w = lin.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - (w - 1) / 2) ** 2 + (yy - (h - 1) / 2) ** 2) / (((w - 1) / 2) ** 2 + ((h - 1) / 2) ** 2)
    return lin * (1.0 - strength * r2)[..., None]


def _c09(img, p, face_px, rng):
    small = cv2.resize(img, _scaled_size(img, p["factor"]), interpolation=cv2.INTER_AREA)
    small = _chromatic_aberration(small, p["ca_frac"])
    small = _to_u8(_vignette(_to_lin(small), p["vignette"]))
    f = small.astype(np.float32)
    f += rng.standard_normal(f.shape).astype(np.float32) * p["noise_sigma"]
    f = np.clip(f, 0, 255)
    usm_sigma = max(0.8, face_px * p["factor"] * 0.01)
    f = np.clip(f + p["sharpen"] * (f - cv2.GaussianBlur(f, (0, 0), usm_sigma)), 0, 255)
    out = jpeg_roundtrip(np.rint(f).astype(np.uint8), p["jpeg_quality"])
    return out, {"usm_sigma_px": usm_sigma}


def _c10(img, p, face_px, rng):
    d = int(face_px * 0.03) | 1
    sigma_space = face_px * 0.03
    out = img
    for _ in range(p["passes"]):
        out = cv2.bilateralFilter(out, d, p["sigma_color"], sigma_space)
    lin = _to_lin(out) * (1.0 + p["brighten"])
    tint = 0.01 * p["passes"]
    lin = lin * np.array([1 + tint, 1.0, 1 - tint], np.float32)
    return _to_u8(lin), {"diameter_px": d, "sigma_space": sigma_space, "warm_tint": tint}


# --------------------------------------------------------------------------- R1 CodeFormer
def _r1(img, p, face_px, rng):
    sigma_px = p["blur_sigma"] * face_px / _CODEFORMER_FACE_PX
    out = cv2.GaussianBlur(img, (0, 0), sigma_px)
    out = cv2.resize(out, _scaled_size(out, 1.0 / p["downsample"]), interpolation=cv2.INTER_LINEAR)
    noisy = out.astype(np.float32) + rng.standard_normal(out.shape).astype(np.float32) * p["noise"]
    out = jpeg_roundtrip(np.clip(np.rint(noisy), 0, 255).astype(np.uint8), p["jpeg_quality"])
    return out, {"blur_sigma_px": sigma_px}


# --------------------------------------------------------------------------- R2 Real-ESRGAN
def _grid(k: int) -> np.ndarray:
    ax = np.arange(-k // 2 + 1.0, k // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    return np.stack([xx, yy], axis=-1)


def _sigma_matrix(sx: float, sy: float, theta: float) -> np.ndarray:
    d = np.diag([sx ** 2, sy ** 2])
    u = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return u @ d @ u.T


def _quad_form(k: int, sx: float, sy: float, theta: float, iso: bool) -> np.ndarray:
    sm = np.diag([sx ** 2, sx ** 2]) if iso else _sigma_matrix(sx, sy, theta)
    g = _grid(k)
    return np.sum((g @ np.linalg.inv(sm)) * g, axis=2)


def random_mixed_kernel(rng: np.random.Generator, kernel_size: int, kernel_list, kernel_prob,
                        sigma_range, beta_g, beta_p) -> tuple[np.ndarray, dict]:
    """Port of basicsr random_mixed_kernels (noise_range=None) with a Generator."""
    ktype = str(rng.choice(kernel_list, p=np.asarray(kernel_prob) / np.sum(kernel_prob)))
    iso = ktype.endswith("iso") and not ktype.endswith("aniso")
    sx = float(rng.uniform(*sigma_range))
    if iso:
        sy, rot = sx, 0.0
    else:
        sy = float(rng.uniform(*sigma_range))
        rot = float(rng.uniform(-math.pi, math.pi))
    q = _quad_form(kernel_size, sx, sy, rot, iso)
    desc: dict[str, Any] = {"type": ktype, "size": kernel_size, "sigma_x": sx, "sigma_y": sy,
                            "rotation": rot}
    if ktype == "iso" or ktype == "aniso":
        k = np.exp(-0.5 * q)
    elif ktype.startswith("generalized"):
        beta = float(rng.uniform(beta_g[0], 1) if rng.uniform() < 0.5 else rng.uniform(1, beta_g[1]))
        k, desc["beta"] = np.exp(-0.5 * np.power(q, beta)), beta
    else:  # plateau
        beta = float(rng.uniform(beta_p[0], 1) if rng.uniform() < 0.5 else rng.uniform(1, beta_p[1]))
        k, desc["beta"] = 1.0 / (np.power(q, beta) + 1.0), beta
    return k / k.sum(), desc


def circular_lowpass_kernel(cutoff: float, kernel_size: int, pad_to: int = 0) -> np.ndarray:
    """2D sinc low-pass kernel (port of basicsr circular_lowpass_kernel)."""
    c = (kernel_size - 1) / 2
    x, y = np.meshgrid(np.arange(kernel_size), np.arange(kernel_size), indexing="ij")
    r = np.sqrt((x - c) ** 2 + (y - c) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = cutoff * special.j1(cutoff * r) / (2 * np.pi * r)
    k[int(c), int(c)] = cutoff ** 2 / (4 * np.pi)
    k = k / k.sum()
    if pad_to > kernel_size:
        pw = (pad_to - kernel_size) // 2
        k = np.pad(k, ((pw, pw), (pw, pw)))
    return k


def _pad21(k: np.ndarray) -> np.ndarray:
    pw = (21 - k.shape[0]) // 2
    return np.pad(k, ((pw, pw), (pw, pw)))


def _r2_blur_kernel(rng, sinc_prob: float, sigma_range) -> tuple[np.ndarray, dict]:
    size = int(rng.choice(_R2["kernel_sizes"]))
    if rng.uniform() < sinc_prob:
        lo = np.pi / 3 if size < 13 else np.pi / 5
        omega = float(rng.uniform(lo, np.pi))
        return _pad21(circular_lowpass_kernel(omega, size)), {"type": "sinc", "size": size,
                                                               "omega_c": omega}
    k, d = random_mixed_kernel(rng, size, _R2["kernel_list"], _R2["kernel_prob"], sigma_range,
                               _R2["betag_range"], _R2["betap_range"])
    return _pad21(k), d


def _r2_draw(rng: np.random.Generator) -> dict:
    """Sample one complete second-order degradation (kernels, resizes, noise, ordering)."""
    c = _R2
    k1, d1 = _r2_blur_kernel(rng, c["sinc_prob"], c["blur_sigma"])
    k2, d2 = _r2_blur_kernel(rng, c["sinc_prob2"], c["blur_sigma2"])
    if rng.uniform() < c["final_sinc_prob"]:
        size = int(rng.choice(c["kernel_sizes"]))
        omega = float(rng.uniform(np.pi / 3, np.pi))
        ks = circular_lowpass_kernel(omega, size, pad_to=21)
        dsinc: dict[str, Any] = {"type": "sinc", "size": size, "omega_c": omega}
    else:
        ks = np.zeros((21, 21)); ks[10, 10] = 1.0
        dsinc = {"type": "pulse"}

    def stage(prefix: str, resize_p, resize_r, gn_p, nr, pr, gray, jr) -> dict:
        kind = str(rng.choice(["up", "down", "keep"], p=resize_p))
        scale = {"up": lambda: rng.uniform(1, resize_r[1]), "down": lambda: rng.uniform(resize_r[0], 1),
                 "keep": lambda: 1.0}[kind]()
        interp = str(rng.choice(["area", "bilinear", "bicubic"]))
        gaussian = bool(rng.uniform() < gn_p)
        return {"resize": kind, "scale": float(scale), "interp": interp, "gaussian": gaussian,
                "noise_level": float(rng.uniform(*(nr if gaussian else pr))),
                "gray": bool(rng.uniform() < gray), "jpeg": float(rng.uniform(*jr))}

    s1 = stage("1", c["resize_prob"], c["resize_range"], c["gaussian_noise_prob"], c["noise_range"],
               c["poisson_scale_range"], c["gray_noise_prob"], c["jpeg_range"])
    blur2 = bool(rng.uniform() < c["second_blur_prob"])
    s2 = stage("2", c["resize_prob2"], c["resize_range2"], c["gaussian_noise_prob2"],
               c["noise_range2"], c["poisson_scale_range2"], c["gray_noise_prob2"], c["jpeg_range2"])
    resize_first = bool(rng.uniform() < 0.5)
    final_interp = str(rng.choice(["area", "bilinear", "bicubic"]))
    desc = {"kernel1": d1, "stage1": s1, "second_blur": blur2, "kernel2": d2 if blur2 else None,
            "stage2": s2, "resize_then_sinc_before_jpeg": resize_first, "final_interp": final_interp,
            "final_sinc": dsinc}
    return {"desc": desc, "k1": k1, "k2": k2, "ks": ks, "noise_seed": int(rng.integers(2 ** 31))}


_INTERP = {"area": cv2.INTER_AREA, "bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}


def _usm_sharp(img: np.ndarray, weight: float = 0.5, radius: int = 50, threshold: int = 10):
    """Real-ESRGAN USMSharp applied to the ground truth before degrading (float [0, 1])."""
    ks = radius + 1 if radius % 2 == 0 else radius
    blur = cv2.GaussianBlur(img, (ks, ks), 0)
    resid = img - blur
    mask = (np.abs(resid) * 255 > threshold).astype(np.float32)
    soft = cv2.GaussianBlur(mask, (ks, ks), 0)
    sharp = np.clip(img + weight * resid, 0, 1)
    return soft * sharp + (1 - soft) * img


def _r2_noise(img: np.ndarray, st: dict, rng: np.random.Generator) -> np.ndarray:
    """Gaussian or Poisson noise as in basicsr *_noise_pt (gray noise shared across channels)."""
    h, w = img.shape[:2]
    if st["gaussian"]:
        noise = rng.standard_normal(img.shape).astype(np.float32) * st["noise_level"] / 255.0
        if st["gray"]:
            noise = np.repeat(rng.standard_normal((h, w, 1)).astype(np.float32), 3, 2) \
                * st["noise_level"] / 255.0
        return np.clip(img + noise, 0, 1)

    def poisson(x: np.ndarray) -> np.ndarray:
        x = np.clip(np.rint(x * 255.0), 0, 255) / 255.0
        vals = 2 ** np.ceil(np.log2(len(np.unique(x))))
        return (rng.poisson(x * vals) / vals).astype(np.float32) - x

    noise = poisson(img)
    if st["gray"]:
        g = img @ np.array([0.299, 0.587, 0.114], np.float32)
        noise = np.repeat(poisson(g)[..., None], 3, 2)
    return np.clip(img + noise * st["noise_level"], 0, 1)


def _r2_jpeg(img: np.ndarray, quality: float) -> np.ndarray:
    u8 = _f_to_u8(np.clip(img, 0, 1))
    return jpeg_roundtrip(u8, int(round(quality))).astype(np.float32) / 255.0


def _r2_run(img: np.ndarray, draw: dict) -> np.ndarray:
    """Run one drawn pipeline on a uint8 RGB image -> uint8 RGB at 1/4 size."""
    d, sc = draw["desc"], _R2["scale"]
    rng = np.random.default_rng(draw["noise_seed"])
    h, w = img.shape[:2]
    x = _usm_sharp(img.astype(np.float32) / 255.0)
    # first degradation
    s = d["stage1"]
    x = _filter(x, draw["k1"])
    x = cv2.resize(x, (max(1, int(w * s["scale"])), max(1, int(x.shape[0] * s["scale"]))),
                   interpolation=_INTERP[s["interp"]])
    x = _r2_jpeg(_r2_noise(x, s, rng), s["jpeg"])
    # second degradation
    s = d["stage2"]
    if d["second_blur"]:
        x = _filter(x, draw["k2"])
    x = cv2.resize(x, (max(1, int(w / sc * s["scale"])), max(1, int(h / sc * s["scale"]))),
                   interpolation=_INTERP[s["interp"]])
    x = _r2_noise(x, s, rng)
    size = (w // sc, h // sc)
    if d["resize_then_sinc_before_jpeg"]:
        x = _filter(cv2.resize(x, size, interpolation=_INTERP[d["final_interp"]]), draw["ks"])
        x = _r2_jpeg(x, s["jpeg"])
    else:
        x = _r2_jpeg(x, s["jpeg"])
        x = _filter(cv2.resize(x, size, interpolation=_INTERP[d["final_interp"]]), draw["ks"])
    return _f_to_u8(x)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 0 else 10 * math.log10(255.0 ** 2 / mse)


def _r2_pick(img: np.ndarray, seed: int, severity: int) -> tuple[dict, dict]:
    """Draw candidates from the seed, rank by proxy PSNR, return the 25/50/75th pct one."""
    n, ps = _R2["n_candidates"], _R2["proxy_size"]
    h, w = img.shape[:2]
    ph, pw = min(ps, h) // 4 * 4, min(ps, w) // 4 * 4
    y0, x0 = (h - ph) // 2, (w - pw) // 2
    proxy = np.ascontiguousarray(img[y0:y0 + ph, x0:x0 + pw])  # native pixel scale preserved
    draws, scores = [], []
    for i in range(n):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0xFFFFFFFF, 0x52324, i]))
        draw = _r2_draw(rng)
        lq = _r2_run(proxy, draw)
        up = cv2.resize(lq, (pw, ph), interpolation=cv2.INTER_CUBIC)
        draws.append(draw)
        scores.append(_psnr(up, proxy))
    order = np.argsort(-np.asarray(scores), kind="stable")  # mildest first
    rank = int(round(_R2["quantiles"][severity - 1] * (n - 1)))
    idx = int(order[rank])
    return draws[idx], {"candidate_index": idx, "severity_rank": rank, "n_candidates": n,
                        "proxy_psnr_db": float(scores[idx])}


def _r2(img, p, face_px, rng, *, seed=0, severity=1):
    draw, info = _r2_pick(img, seed, severity)
    out = _r2_run(img, draw)
    return out, {**info, "pipeline": draw["desc"]}


# --------------------------------------------------------------------------- R3 haze / R4 shadow
def _smooth_field(h: int, w: int, rng: np.random.Generator, grid: int = 5) -> np.ndarray:
    f = cv2.resize(rng.standard_normal((grid, grid)).astype(np.float32), (w, h),
                   interpolation=cv2.INTER_CUBIC)
    return f / (np.abs(f).max() + 1e-6)


def _r3(img, p, face_px, rng):
    h, w = img.shape[:2]
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    depth = 0.6 + 0.4 * (0.5 - yy) + 0.25 * _smooth_field(h, w, rng)
    depth = np.clip(depth, 0.05, None)
    half = int(face_px / 2)
    cy, cx = h // 2, w // 2
    d_face = float(depth[max(cy - half, 0):cy + half, max(cx - half, 0):cx + half].mean())
    beta = -math.log(p["face_transmission"]) / d_face
    t = np.exp(-beta * depth)[..., None]
    airlight = float(rng.uniform(0.85, 0.95))
    tint = np.array([1.0, 1.0, 1.0], np.float32) * airlight
    out = img.astype(np.float32) / 255.0 * t + tint * (1 - t)
    return _f_to_u8(out), {"beta": beta, "airlight": airlight, "face_depth": d_face,
                           "face_transmission_realised": float(
                               t[max(cy - half, 0):cy + half, max(cx - half, 0):cx + half].mean())}


def _r4(img, p, face_px, rng):
    h, w = img.shape[:2]
    ang = float(rng.uniform(0, math.pi))
    offset = rng.uniform(-0.15, 0.15, 2) * face_px
    curved = bool(rng.random() < 0.5)
    curv = float(rng.uniform(-0.6, 0.6)) if curved else 0.0
    side = float(rng.choice([-1.0, 1.0]))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    px, py = xx - (w / 2 + offset[0]), yy - (h / 2 + offset[1])
    n = (math.cos(ang), math.sin(ang))
    tvec = (-n[1], n[0])
    dist = side * (px * n[0] + py * n[1] + curv * (px * tvec[0] + py * tvec[1]) ** 2 / face_px)
    feather = face_px * 0.01
    m = np.clip(0.5 - dist / (2 * feather), 0.0, 1.0)[..., None]
    gain = p["factor"] * np.array([0.94, 0.98, 1.06], np.float32)
    lin = _to_lin(img)
    out = lin * (1 - m) + lin * gain * m
    return _to_u8(out), {"angle_deg": math.degrees(ang), "offset_px": [float(v) for v in offset],
                         "curved": curved, "curvature": curv, "feather_px": feather,
                         "shadow_fraction": float(m.mean()), "tint_gain": [0.94, 0.98, 1.06]}


# --------------------------------------------------------------------------- registry
def _sev(**cols: tuple) -> tuple:
    return tuple({k: v[i] for k, v in cols.items()} for i in range(3))


def _spec(pid, name, scenario, sev, restorer=False) -> ProfileSpec:
    return ProfileSpec(pid, name, scenario, sev, restorer)


_SPECS = [
    _spec("C01", "low_resolution", "Distant or tiny face; whole frame box-downsampled",
          _sev(face_target_px=(64, 32, 16))),
    _spec("C02", "defocus", "Out-of-focus capture, disk (pillbox) blur",
          _sev(radius_frac=(0.012, 0.024, 0.048))),
    _spec("C03", "motion_blur", "Camera or subject motion, linear blur at a random angle",
          _sev(length_frac=(0.03, 0.06, 0.10))),
    _spec("C04", "sensor_noise", "High-ISO sensor: shot, read and chroma noise",
          _sev(photons=(400, 120, 40), read_sigma=(2 / 255, 5 / 255, 10 / 255))),
    _spec("C05", "low_light", "Dim scene: underexposed, noisy, gained back, JPEG q85",
          _sev(exposure=(0.35, 0.2, 0.1), photons=(120, 40, 40 / 1.5),
               read_sigma=(5 / 255, 10 / 255, 15 / 255), target_brightness=(0.6, 0.5, 0.4),
               jpeg_quality=(85, 85, 85))),
    _spec("C06", "overexposure", "Harsh flash: gain, clipping, bloom and contrast loss",
          _sev(gain=(1.3, 1.6, 2.0), bloom=(0.3, 0.3, 0.3), contrast_loss=(0.1, 0.2, 0.3))),
    _spec("C07", "jpeg", "Heavy JPEG compression", _sev(quality=(30, 15, 5))),
    _spec("C08", "messaging_chain", "Messaging app re-share: two rounds of downscale + JPEG",
          _sev(factor=(0.5, 0.35, 0.25), quality=(70, 50, 30))),
    _spec("C09", "cheap_webcam", "Cheap webcam: low-res, CA, vignette, noise, sharpening, JPEG",
          _sev(factor=(0.5, 0.35, 0.25), ca_frac=(0.002, 0.004, 0.007),
               vignette=(0.15, 0.3, 0.45), noise_sigma=(3, 6, 10), sharpen=(0.5, 1.0, 1.5),
               jpeg_quality=(75, 55, 35))),
    _spec("C10", "beauty_filter", "Beauty filter: edge-preserving smoothing, brighten, warm tint",
          _sev(sigma_color=(20, 35, 55), passes=(1, 2, 3), brighten=(0.04, 0.08, 0.12))),
    _spec("R1", "bfr_classical", "CodeFormer/GFPGAN blind-face-restoration degradation",
          _sev(blur_sigma=(4.5, 8.0, 11.5), downsample=(8.25, 15.5, 22.75), noise=(5, 10, 15),
               jpeg_quality=(75, 60, 45)), True),
    _spec("R2", "realesrgan_second_order", "Real-ESRGAN x4plus second-order degradation (1/4 size)",
          _sev(percentile=(0.25, 0.5, 0.75), n_candidates=(64, 64, 64)), True),
    _spec("R3", "haze", "Atmospheric scattering haze over the face",
          _sev(face_transmission=(0.75, 0.55, 0.35)), True),
    _spec("R4", "shadow", "Hard-edged cast shadow across the face",
          _sev(factor=(0.65, 0.45, 0.30)), True),
]
PROFILES: dict[str, ProfileSpec] = {s.id: s for s in _SPECS}

_IMPL: dict[str, Callable] = {
    "C01": _c01, "C02": _c02, "C03": _c03, "C04": _c04, "C05": _c05, "C06": _c06, "C07": _c07,
    "C08": _c08, "C09": _c09, "C10": _c10, "R1": _r1, "R2": _r2, "R3": _r3, "R4": _r4,
}


def conditions(include_restorer: bool = True) -> list[tuple[str, int]]:
    """[("clean", 0)] followed by every (profile_id, severity) pair."""
    out = [("clean", 0)]
    for pid, spec in PROFILES.items():
        if spec.restorer_aligned and not include_restorer:
            continue
        out.extend((pid, s) for s in SEVERITIES)
    return out


def condition_seed(base_seed: int, case_key: str, profile_id: str, severity: int) -> int:
    """Stable 32-bit seed from (base_seed, case, profile, severity); independent of PYTHONHASHSEED."""
    key = f"{int(base_seed)}|{case_key}|{profile_id}|{int(severity)}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def apply(image: np.ndarray, profile_id: str, severity: int, *, face_px: float,
          seed: int) -> tuple[np.ndarray, dict]:
    """Degrade a uint8 RGB crop. Returns (image, meta); input is never modified."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be uint8 with shape (H, W, 3)")
    h, w = image.shape[:2]
    meta: dict[str, Any] = {"profile_id": profile_id, "severity": int(severity), "seed": int(seed),
                            "face_px": float(face_px), "input_size": [w, h]}
    if profile_id == "clean":
        out, params, scale = image.copy(), {}, (1.0, 1.0)
        meta["name"] = "clean"
    else:
        if profile_id not in PROFILES:
            raise ValueError(f"unknown profile {profile_id!r}")
        if severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}")
        spec = PROFILES[profile_id]
        p = dict(spec.severities[severity - 1])
        rng = np.random.default_rng(int(seed))
        kw = {"seed": int(seed), "severity": int(severity)} if profile_id == "R2" else {}
        out, extra = _IMPL[profile_id](image, p, float(face_px), rng, **kw)
        params = {**p, **extra}
        meta["name"] = spec.name
    ow, oh = out.shape[1], out.shape[0]
    sx, sy = ow / w, oh / h
    meta.update(params=_jsonable(params), output_size=[ow, oh],
                clean_to_degraded=[[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])
    return out, meta
