# TMO input and output domains

The MATLAB toolbox `lum.m` assumes linear RGB values with sRGB/Rec.709
primaries. Rec.2020 inputs must therefore be converted to linear Rec.709
before MATLAB-compatible luminance-based tone mapping. Transfer decoding
(PQ, gamma, and so on) must happen before that conversion.

## Gamma and logarithmic operators

- `GammaTMO` does not normalize by the image maximum. It applies an exposure
  multiplier, raises the result to `1/gamma`, and clamps RGB to `[0,1]`.
  HDR values therefore require a suitable `fstop`, `BestExposureTMO`, or
  another tone mapper first.
- `LogarithmicTMO` normalizes its mapped luminance by
  `log10(1 + Lmax * k)`. With the default `q=k=1`, the maximum mapped
  luminance is one. It does not clamp RGB after restoring chroma, so an RGB
  channel may exceed one even though mapped luminance is in range.
- `NormalizeTMO` is the explicit maximum/robust-normalization operator.

## Input domain

All single-image HDR TMO entry points accept linear-light input. The
following models additionally depend on absolute or display luminance and
should not be pre-normalized:

- `BanterleTMO`, `PattanaikTMO`, `VanHaterenTMO`
- `FerwerdaTMO`, `TumblinTMO`, `WardGlobalTMO`, `WardHistAdjTMO`
- `YPFerwerdaTMO`, `YPTumblinTMO`, `YPWardGlobalTMO`
- `KuangTMO` in calibrated mode

`BanterleTMO` and `KuangTMO` provide explicit rescaling/unknown-calibration
modes. Pre-normalizing absolute-luminance inputs changes the intended visual
adaptation behavior.

## Output processing

Already display/gamma encoded:

- `GammaTMO`
- `BruceExpoBlendTMO`
- `MertensTMO`
- `RamanTMO`

Special display response:

- `DragoTMO` should use `GammaDrago`.
- `VanHaterenTMO` states that no additional gamma correction is required.
- `ReinhardDevlinTMO` recommends gamma 1.6.

The remaining operators produce linear output and normally require display
encoding such as `GammaTMO(output, 2.2)` before writing an SDR image. This is
also the workflow used by MATLAB `ConvHDRtoLDR.m`.

An RGB maximum above one does not by itself mean that luminance normalization
failed. Operators using `ChangeLuminance` preserve chromatic ratios, which
can produce out-of-range RGB channels after a bounded luminance mapping.
For SDR previews, the batch tester compresses out-of-gamut RGB by scaling all
three channels together instead of clipping each channel independently. This
preserves hue and avoids white highlights caused only by gamut clipping.
