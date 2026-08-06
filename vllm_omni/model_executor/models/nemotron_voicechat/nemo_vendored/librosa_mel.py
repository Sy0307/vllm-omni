# SPDX-License-Identifier: ISC
# SPDX-FileCopyrightText: Copyright (c) 2013--2023, librosa development team
#
# Vendored verbatim from librosa 0.11.0 (librosa/core/convert.py and
# librosa/filters.py, ISC license) so the NemotronVoiceChat vendored NeMo
# modules do not depend on the librosa package (undeclared in this repo and
# banned by its lint rules). The arithmetic is IDENTICAL to librosa's —
# the mel filterbank values feed the parity-verified perception/codec paths,
# so this must never be "simplified" or re-derived.
#
# Only the norm values the NeMo configs use are supported (None / "slaney");
# numeric norms would need librosa.util.normalize and are rejected loudly.
"""Exact librosa mel-filterbank math (hz/mel conversion + triangular filters)."""

import warnings
from typing import Any

import numpy as np


def hz_to_mel(frequencies: Any, *, htk: bool = False) -> np.ndarray:
    """Convert Hz to Mels (librosa.core.convert.hz_to_mel, verbatim)."""
    frequencies = np.asanyarray(frequencies)

    if htk:
        mels: np.ndarray = 2595.0 * np.log10(1.0 + frequencies / 700.0)
        return mels

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if frequencies.ndim:
        # If we have array data, vectorize
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels


def mel_to_hz(mels: Any, *, htk: bool = False) -> np.ndarray:
    """Convert mel bin numbers to frequencies (librosa verbatim)."""
    mels = np.asanyarray(mels)

    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    if mels.ndim:
        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs


def mel_frequencies(n_mels: int = 128, *, fmin: float = 0.0, fmax: float = 11025.0, htk: bool = False) -> np.ndarray:
    """Mel-scale center frequencies (librosa verbatim)."""
    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)

    hz: np.ndarray = mel_to_hz(mels, htk=htk)
    return hz


def fft_frequencies(*, sr: float = 22050, n_fft: int = 2048) -> np.ndarray:
    """rfft bin center frequencies (librosa verbatim)."""
    return np.fft.rfftfreq(n=n_fft, d=1.0 / sr)


def mel(
    *,
    sr: float,
    n_fft: int,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    htk: bool = False,
    norm: str | None = "slaney",
    dtype: Any = np.float32,
) -> np.ndarray:
    """Create a Mel filter-bank (librosa.filters.mel, verbatim math).

    Deviation from librosa: numeric ``norm`` values (which need
    ``librosa.util.normalize``) are rejected — the NeMo configs only use
    ``"slaney"`` (the default) or ``None``.
    """
    if fmax is None:
        fmax = float(sr) / 2

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if isinstance(norm, str):
        if norm == "slaney":
            # Slaney-style mel is scaled to be approx constant energy per channel
            enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
            weights *= enorm[:, np.newaxis]
        else:
            raise ValueError(f"Unsupported norm={norm}")
    elif norm is not None:
        raise NotImplementedError(
            f"Numeric mel norms are not supported in the vendored filterbank (got norm={norm!r}); "
            "the NemotronVoiceChat configs use 'slaney' or None."
        )

    # Only check weights if f_mel[0] is positive
    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        # This means we have an empty channel somewhere
        warnings.warn(
            "Empty filters detected in mel frequency basis. "
            "Some channels will produce empty responses. "
            "Try increasing your sampling rate (and fmax) or "
            "reducing n_mels.",
            stacklevel=2,
        )

    return weights
