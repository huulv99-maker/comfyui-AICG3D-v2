# Copyright (C) 2026 AICG3D
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Audio helpers that keep working when ``torchaudio`` is not installed.
#
# Recent ComfyUI releases no longer depend on torchaudio, and PyTorch wheels newer
# than the last torchaudio release cannot be paired with it.  Every function here
# uses torchaudio when it imports and works, and otherwise falls back to a
# pure torch / scipy implementation.
#
# The fallback algorithms (windowed-sinc resampling, Slaney mel filterbank,
# biquad high-pass) follow the reference implementations in torchaudio
# (BSD-2-Clause, Copyright (c) 2017 Facebook Inc. (Soumith Chintala)).
# See third_party/LICENSE-torchaudio-BSD-2-Clause.txt.

from __future__ import annotations

import math

import torch

try:  # torchaudio is optional on current ComfyUI hosts
    import torchaudio as _torchaudio
except Exception:  # pragma: no cover - depends on the host environment
    _torchaudio = None

HAS_TORCHAUDIO = _torchaudio is not None


# --------------------------------------------------------------------------- resample
def _sinc_resample_kernel(orig_freq: int, new_freq: int, gcd: int,
                          lowpass_filter_width: int = 6, rolloff: float = 0.99):
    orig_freq //= gcd
    new_freq //= gcd
    base_freq = min(orig_freq, new_freq) * rolloff
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    idx = torch.arange(-width, width + orig_freq, dtype=torch.float64)[None, None] / orig_freq
    t = torch.arange(0, -new_freq, -1, dtype=torch.float64)[:, None, None] / new_freq + idx
    t = (t * base_freq).clamp_(-lowpass_filter_width, lowpass_filter_width)
    window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    t = t * math.pi
    scale = base_freq / orig_freq
    kernels = torch.where(t == 0, torch.ones_like(t), t.sin() / t)
    kernels = kernels * window * scale
    return kernels, width


def _resample_torch(waveform: torch.Tensor, orig_freq: int, new_freq: int) -> torch.Tensor:
    gcd = math.gcd(orig_freq, new_freq)
    kernel, width = _sinc_resample_kernel(orig_freq, new_freq, gcd)
    kernel = kernel.to(device=waveform.device, dtype=waveform.dtype)
    orig = orig_freq // gcd
    new = new_freq // gcd
    shape = waveform.shape
    flat = waveform.reshape(-1, shape[-1])
    length = flat.shape[-1]
    flat = torch.nn.functional.pad(flat, (width, width + orig))
    out = torch.nn.functional.conv1d(flat[:, None], kernel, stride=orig)
    out = out.transpose(1, 2).reshape(flat.shape[0], -1)
    target = int(math.ceil(new * length / orig))
    out = out[..., :target]
    return out.reshape(shape[:-1] + out.shape[-1:])


def resample(waveform: torch.Tensor, orig_freq: int, new_freq: int) -> torch.Tensor:
    """Drop-in for ``torchaudio.functional.resample`` (last dim is time)."""
    orig_freq = int(orig_freq)
    new_freq = int(new_freq)
    if orig_freq == new_freq or orig_freq <= 0 or new_freq <= 0:
        return waveform
    if _torchaudio is not None:
        try:
            return _torchaudio.functional.resample(waveform, orig_freq, new_freq)
        except Exception:
            pass
    if not waveform.is_floating_point():
        waveform = waveform.float()
    return _resample_torch(waveform, orig_freq, new_freq)


# ------------------------------------------------------------------ highpass biquad
def highpass_biquad(waveform: torch.Tensor, sample_rate: int, cutoff_freq: float,
                    Q: float = 0.707) -> torch.Tensor:
    """Drop-in for ``torchaudio.functional.highpass_biquad`` (output clamped to [-1, 1])."""
    if _torchaudio is not None:
        try:
            return _torchaudio.functional.highpass_biquad(waveform, sample_rate, cutoff_freq, Q)
        except Exception:
            pass
    import numpy as np
    from scipy.signal import lfilter

    w0 = 2.0 * math.pi * cutoff_freq / sample_rate
    alpha = math.sin(w0) / 2.0 / Q
    cos_w0 = math.cos(w0)
    b0 = (1.0 + cos_w0) / 2.0
    b = np.array([b0, -1.0 - cos_w0, b0], dtype=np.float64)
    a = np.array([1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha], dtype=np.float64)
    data = waveform.detach().cpu().double().numpy()
    filtered = lfilter(b / a[0], a / a[0], data, axis=-1)
    out = torch.from_numpy(np.ascontiguousarray(np.clip(filtered, -1.0, 1.0)))
    return out.to(device=waveform.device, dtype=waveform.dtype)


# ------------------------------------------------------------------------ mel spectrogram
def _hz_to_mel_slaney(freq: torch.Tensor) -> torch.Tensor:
    f_sp = 200.0 / 3
    mels = freq / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    log_region = freq >= min_log_hz
    return torch.where(log_region, min_log_mel + torch.log(freq.clamp_min(1e-10) / min_log_hz) / logstep, mels)


def _mel_to_hz_slaney(mels: torch.Tensor) -> torch.Tensor:
    f_sp = 200.0 / 3
    freqs = f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    log_region = mels >= min_log_mel
    return torch.where(log_region, min_log_hz * torch.exp(logstep * (mels - min_log_mel)), freqs)


def _melscale_fbanks_slaney(n_freqs: int, f_min: float, f_max: float, n_mels: int,
                            sample_rate: int) -> torch.Tensor:
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs, dtype=torch.float64)
    m_pts = torch.linspace(
        float(_hz_to_mel_slaney(torch.tensor(float(f_min), dtype=torch.float64))),
        float(_hz_to_mel_slaney(torch.tensor(float(f_max), dtype=torch.float64))),
        n_mels + 2, dtype=torch.float64,
    )
    f_pts = _mel_to_hz_slaney(m_pts)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
    down = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    fb = torch.max(torch.zeros(1, dtype=torch.float64), torch.min(down, up))
    fb = fb * (2.0 / (f_pts[2:n_mels + 2] - f_pts[:n_mels])).unsqueeze(0)
    return fb.float()


class _TorchMelSpectrogram(torch.nn.Module):
    """Minimal ``torchaudio.transforms.MelSpectrogram`` (slaney scale and norm)."""

    def __init__(self, sample_rate, n_fft, win_length, hop_length, f_min, f_max, n_mels, power):
        super().__init__()
        self.n_fft = int(n_fft)
        self.win_length = int(win_length) if win_length else self.n_fft
        self.hop_length = int(hop_length) if hop_length else self.win_length // 2
        self.power = float(power)
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)
        fmax = float(f_max) if f_max is not None else float(sample_rate // 2)
        self.register_buffer(
            "fb",
            _melscale_fbanks_slaney(self.n_fft // 2 + 1, float(f_min), fmax, int(n_mels), int(sample_rate)),
            persistent=False,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        shape = waveform.shape
        flat = waveform.reshape(-1, shape[-1])
        spec = torch.stft(
            flat, self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=self.window.to(flat.device, flat.dtype), center=True, pad_mode="reflect",
            normalized=False, onesided=True, return_complex=True,
        ).abs()
        if self.power != 1.0:
            spec = spec.pow(self.power)
        mel = torch.matmul(spec.transpose(-1, -2), self.fb.to(spec.device, spec.dtype)).transpose(-1, -2)
        return mel.reshape(shape[:-1] + mel.shape[-2:])


def MelSpectrogram(sample_rate=16000, n_fft=400, win_length=None, hop_length=None,
                   f_min=0.0, f_max=None, n_mels=128, power=1.0, norm="slaney",
                   mel_scale="slaney"):
    """Drop-in for ``torchaudio.transforms.MelSpectrogram``.

    The fallback only implements the Slaney scale/norm used by Spark-TTS; any other
    combination requires torchaudio.
    """
    if _torchaudio is not None:
        try:
            return _torchaudio.transforms.MelSpectrogram(
                sample_rate, n_fft, win_length, hop_length, f_min, f_max,
                n_mels=n_mels, power=power, norm=norm, mel_scale=mel_scale,
            )
        except Exception:
            pass
    if norm != "slaney" or mel_scale != "slaney":
        raise RuntimeError(
            "torchaudio is not installed and the built-in mel fallback only supports "
            "norm='slaney' / mel_scale='slaney'."
        )
    return _TorchMelSpectrogram(sample_rate, n_fft, win_length, hop_length, f_min, f_max, n_mels, power)
