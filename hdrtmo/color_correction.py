from __future__ import annotations

import numpy as np

from .colorspace import (
    convert_ipt_to_ich,
    convert_rgb_to_xyz,
    convert_xyz_to_ipt,
    luminance,
    saturation_pouli,
)
from .core import remove_specials, same_image


def color_correction(image: np.ndarray, saturation: float = 0.5) -> np.ndarray:
    saturation = 0.5 if saturation < 0 else saturation
    luma = luminance(image)
    ratio = np.divide(image, luma[..., None], out=np.zeros_like(image), where=luma[..., None] != 0)
    return remove_specials(np.power(ratio, saturation) * luma[..., None])


def color_correction_linear(image: np.ndarray, saturation: float = 0.5) -> np.ndarray:
    saturation = 0.5 if saturation <= 0 else saturation
    luma = luminance(image)
    ratio = np.divide(image, luma[..., None], out=np.zeros_like(image), where=luma[..., None] != 0)
    return remove_specials(((ratio - 1.0) * saturation + 1.0) * luma[..., None])


def color_correction_sigmoid(
    cone_luminance: np.ndarray,
    exponent: float,
    sigma: float,
    bleaching: float,
) -> np.ndarray:
    numerator = exponent * bleaching * cone_luminance**exponent * sigma**exponent
    return numerator / (cone_luminance**exponent + sigma**exponent) ** 2


def color_correction_pouli(
    hdr: np.ndarray,
    tone_mapped: np.ndarray,
    clamp_tmo: bool = False,
) -> np.ndarray:
    if not same_image(hdr, tone_mapped):
        raise ValueError("HDR and tone-mapped images must have the same shape")
    maximum_tmo = float(np.max(tone_mapped))
    maximum_hdr = float(np.max(hdr))
    hdr_normalized = hdr / maximum_hdr
    if clamp_tmo and maximum_tmo > 1:
        maximum_tmo = 1.0
    tmo_normalized = tone_mapped / maximum_tmo
    tmo_ich = convert_ipt_to_ich(convert_xyz_to_ipt(convert_rgb_to_xyz(tmo_normalized)))
    hdr_ich = convert_ipt_to_ich(convert_xyz_to_ipt(convert_rgb_to_xyz(hdr_normalized)))
    intensity = tmo_ich[..., 0].copy()
    chroma_prime = tmo_ich[..., 1] * hdr_ich[..., 0] / (tmo_ich[..., 0] + 1e-5)
    scale = saturation_pouli(hdr_ich[..., 1], hdr_ich[..., 0]) / saturation_pouli(
        chroma_prime, tmo_ich[..., 0]
    )
    tmo_ich[..., 1] = scale * chroma_prime
    tmo_ich[..., 2] = hdr_ich[..., 2]
    tmo_ich[..., 0] = intensity
    corrected = convert_rgb_to_xyz(
        convert_xyz_to_ipt(convert_ipt_to_ich(tmo_ich, True), True),
        True,
    )
    return np.maximum(remove_specials(corrected) * maximum_tmo, 0.0)


ColorCorrection = color_correction
ColorCorrectionLinear = color_correction_linear
ColorCorrectionSigmoid = color_correction_sigmoid
ColorCorrectionPouli = color_correction_pouli

