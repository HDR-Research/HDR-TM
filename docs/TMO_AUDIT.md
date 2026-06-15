# Single-image TMO source audit

Audit basis: the 31 `*TMO.m` files in
`HDR_Toolbox-master/source_code/Tmo`, compared with `hdrtmo/tmo.py`,
`hdrtmo/tmo_local.py`, and `hdrtmo/runner.py`.

`formula` means that the principal equations, default parameters, and return
domain follow the MATLAB source. It does not claim floating-point or
pixel-identical parity. `approximate` identifies a known backend or
algorithmic approximation.

| MATLAB operator | Python entry | Absolute luminance | Output | Status |
|---|---|---:|---|---|
| AshikhminTMO | `ashikhmin_tmo` | No | Linear | formula |
| BanterleTMO | `banterle_tmo` | Yes/rescaled | Linear | formula |
| BestExposureTMO | `best_exposure_tmo` | No | Linear `[0,1]` | formula |
| BruceExpoBlendTMO | `bruce_expo_blend_tmo` | No | Gamma encoded | approximate local entropy |
| ChiuTMO | `chiu_tmo` | No | Linear | formula |
| DragoTMO | `drago_tmo` | No | Linear; GammaDrago | formula |
| DurandTMO | `durand_tmo` | No | Linear | formula |
| ExponentialTMO | `exponential_tmo` | No | Linear | formula |
| FerwerdaTMO | `ferwerda_tmo` | Yes | Linear `[0,1]` | formula |
| GammaTMO | `gamma_tmo` | No | Gamma encoded `[0,1]` | formula |
| KimKautzConsistentTMO | `kim_kautz_consistent_tmo` | No | Linear | formula |
| KrawczykTMO | `krawczyk_tmo` | No | Linear | approximate bilateral backend |
| KuangTMO | `kuang_tmo` | Calibrated mode | Linear `[0,1]` | approximate local white/filtering |
| LischinskiTMO | `lischinski_tmo` | No | Linear | formula, SciPy sparse solver |
| LogarithmicTMO | `logarithmic_tmo` | No | Linear | formula |
| MertensTMO | `mertens_tmo` | No | Gamma encoded `[0,1]` | formula, Python pyramids |
| NormalizeTMO | `normalize_tmo` | No | Linear `[0,1]` | formula |
| PattanaikTMO | `pattanaik_tmo` | Yes | Linear visual response | formula |
| RamanTMO | `raman_tmo` | No | Gamma encoded `[0,1]` | approximate bilateral backend |
| ReinhardDevlinTMO | `reinhard_devlin_tmo` | No | Linear; gamma 1.6 suggested | formula |
| ReinhardRobustTMO | `reinhard_robust_tmo` | No | Linear | formula |
| ReinhardTMO | `reinhard_tmo` | No | Linear | global formula |
| SchlickTMO | `schlick_tmo` | No | Linear | formula |
| SelectOverexposedTMO | `select_overexposed_tmo` | No | Linear `[0,1]` | equivalent quantile solution |
| TumblinTMO | `tumblin_tmo` | Yes | Linear display-relative | formula |
| VanHaterenTMO | `van_hateren_tmo` | Yes | Perceptual response | formula, numerical root solver |
| WardGlobalTMO | `ward_global_tmo` | Yes | Linear `[0,1]` | formula |
| WardHistAdjTMO | `ward_hist_adj_tmo` | Yes | Linear | formula |
| YPFerwerdaTMO | `yp_ferwerda_tmo` | Yes | Linear `[0,1]` | formula |
| YPTumblinTMO | `yp_tumblin_tmo` | Yes | Linear display-relative | formula |
| YPWardGlobalTMO | `yp_ward_global_tmo` | Yes | Linear `[0,1]` | formula |

## Important differences

1. MATLAB `lum.m` assumes linear sRGB/Rec.709 primaries. The CLI therefore
   converts linear Rec.2020 input to linear Rec.709 by default.
2. MATLAB MEX and Image Processing Toolbox filters are replaced by
   SciPy/OpenCV implementations. Boundary behavior and kernel sampling can
   differ slightly.
3. Bruce local entropy is approximated using local log-domain variance.
4. SelectOverexposed uses a deterministic quantile solution instead of
   MATLAB `fminsearch` on a discontinuous objective. It targets the same
   requested overexposed fraction and avoids dark local minima.
5. SDR gamut handling is outside the original TMO formula. The CLI scales
   out-of-gamut RGB triplets together before display encoding to preserve
   hue. `--output-mode raw` skips this display step.
6. The CLI can auto-expose GammaTMO, but the function itself retains the
   MATLAB default `fstop=0`.

## Verification

```bash
python tools/test_all_tmos.py --input hdrimage --output tmo_test_results --max-side 256
python -m unittest discover -s tests -q
```

The current functional check covers three differently encoded images and 31
operators, for 93 runs. MATLAB/Octave golden outputs are still required for
strict numerical parity certification.
