# HDR Toolbox Python

Python port of the MATLAB HDR Toolbox, including all 31 single-image tone
mapping operators under `HDR_Toolbox-master/source_code/Tmo`.

> **Reference implementation only:** The paper's traditional TMO images and
> reported benchmark results were generated in MATLAB. This Python translation
> is not numerically identical and may differ in defaults, I/O, interpolation,
> color processing, and floating-point behavior. Use the MATLAB implementation
> and the original deep-learning repositories to reproduce manuscript results.

The original toolbox is Copyright Francesco Banterle and GPL v3 licensed.
Please cite the original HDR Toolbox and *Advanced High Dynamic Range
Imaging (2nd Edition)* when using this port.

## Install

```bash
python -m pip install -e .
```

List available operators:

```bash
hdrtmo --list-algorithms
```

Legacy TOML reader/output configuration remains supported:

```bash
hdrtmo input.exr output.png --config configs/linear_times_100.toml
```

For mixed PNG/EXR/HDR folders, command-line `--transfer auto` is preferred
because one TOML reader transfer applies to every file.

## Input Pipeline

The processing order is:

```text
file values
-> transfer decoding and absolute-luminance scaling
-> linear input RGB
-> input-primary to working-primary conversion
-> TMO
-> gamut compression and display encoding
-> PNG
```

The MATLAB function `lum.m` assumes **linear sRGB/Rec.709 primaries**.
Therefore, Rec.2020 images should normally use:

```text
--input-primaries rec2020 --working-primaries rec709
```

Both Rec.2020 -> Rec.709 and Rec.709 -> Rec.2020 mappings are supported.
Set `--input-primaries` to the original image gamut and
`--working-primaries` to the gamut required before the TMO.

The primary conversion does not remove PQ or gamma. Transfer decoding must
happen first.

### Transfer choices

| Option | Meaning |
|---|---|
| `--transfer pq` | Normalize integer PNG and decode ST 2084/PQ to cd/m² |
| `--transfer linear` | Stored values are already linear |
| `--transfer linear_times_100` | Stored linear value `1.0` represents `100 cd/m²` |
| `--transfer auto` | PNG uses PQ; EXR/HDR/PFM use `linear_times_100` |

For the images in `hdrimage`, use `--transfer auto`: `black_032.png` is PQ
Rec.2020, while `507.exr` and `indoors_1.hdr` are linear Rec.2020 values
multiplied by 100 before tone mapping.

Do not pre-normalize absolute-luminance inputs for `Banterle`, `Ferwerda`,
`Kuang` calibrated mode, `Pattanaik`, `Tumblin`, `VanHateren`, `Ward`, or the
Yee-Pattanaik variants.

## Single Image

Run one algorithm:

```bash
hdrtmo hdrimage/507.exr output/507_reinhard.png \
  --algorithm ReinhardTMO \
  --transfer linear_times_100 \
  --input-primaries rec2020 \
  --working-primaries rec709 \
  --overwrite
```

Names are case-insensitive; the `TMO` suffix is optional:

```bash
hdrtmo hdrimage/507.exr output/507_log.png -a logarithmic \
  --transfer linear_times_100 \
  --input-primaries rec2020 \
  --working-primaries rec709
```

PQ PNG:

```bash
hdrtmo hdrimage/black_032.png output/black_reinhard.png \
  -a reinhard \
  --transfer pq \
  --input-primaries rec2020 \
  --working-primaries rec709
```

GammaTMO is exposure plus gamma encoding, not HDR normalization. Use an
explicit f-stop or automatic exposure:

```bash
hdrtmo hdrimage/507.exr output/507_gamma.png -a gamma \
  --transfer linear_times_100 \
  --input-primaries rec2020 \
  --working-primaries rec709 \
  --auto-expose-gamma
```

## Algorithm Parameters

Pass Python function parameters with repeatable `--param key=value`.
Use this when running one algorithm:

```bash
hdrtmo hdrimage/507.exr output/507_durand.png -a durand \
  --transfer linear_times_100 \
  --input-primaries rec2020 \
  --working-primaries rec709 \
  --param target_contrast=8
```

```bash
hdrtmo hdrimage/indoors_1.hdr output/indoors_reinhard_devlin.png \
  -a reinharddevlin \
  --transfer linear_times_100 \
  --input-primaries rec2020 \
  --working-primaries rec709 \
  --param intensity=-2 \
  --param normalize=true
```

```bash
hdrtmo hdrimage/black_032.png output/black_select.png \
  -a selectoverexposed \
  --transfer pq \
  --input-primaries rec2020 \
  --working-primaries rec709 \
  --param percent=5
```

## Folder Processing

Run one algorithm on all supported images in a folder:

```bash
hdrtmo hdrimage output/reinhard \
  -a ReinhardTMO \
  --transfer auto \
  --input-primaries rec2020 \
  --working-primaries rec709 \
  --overwrite
```

Run selected algorithms:

```bash
hdrtmo hdrimage output/selected \
  -a ReinhardTMO \
  -a LogarithmicTMO \
  -a MertensTMO \
  --transfer auto \
  --input-primaries rec2020 \
  --working-primaries rec709 \
  --overwrite
```

Comma-separated selection is also accepted:

```bash
hdrtmo hdrimage output/selected \
  -a reinhard,drago,durand \
  --transfer auto \
  --input-primaries rec2020 \
  --working-primaries rec709
```

Run all 31 algorithms:

```bash
hdrtmo hdrimage output/all_tmos \
  --all \
  --transfer auto \
  --input-primaries rec2020 \
  --working-primaries rec709 \
  --auto-expose-gamma \
  --overwrite
```

Process subdirectories:

```bash
hdrtmo dataset output/tmo --all --recursive \
  --transfer auto \
  --input-primaries rec2020 \
  --working-primaries rec709
```

Large local operators can require substantial time and memory. For previews:

```bash
hdrtmo hdrimage output/preview --all \
  --transfer auto \
  --input-primaries rec2020 \
  --working-primaries rec709 \
  --max-side 512
```

`--max-side 0` is the default and preserves original resolution.

## Output Modes

SDR PNG output is the default:

```bash
--output-mode sdr --bit-depth 8 --gamma 2.2
```

Linear/raw output:

```bash
hdrtmo hdrimage/507.exr output/507_reinhard.exr \
  -a reinhard \
  --transfer linear_times_100 \
  --input-primaries rec2020 \
  --working-primaries rec709 \
  --output-mode raw
```

Raw mode requires `.exr`, `.hdr`, or `.pfm` output. It skips display gamma
encoding. Algorithms such as Mertens, Raman, Bruce, and GammaTMO already
return display/gamma-encoded values; raw mode does not make those linear.

## Output Encoding

- `MertensTMO`, `RamanTMO`, `BruceExpoBlendTMO`, and `GammaTMO` already
  return display/gamma-encoded values.
- `DragoTMO` uses the toolbox `GammaDrago` curve.
- `VanHaterenTMO` requires no additional gamma according to the MATLAB code.
- `ReinhardDevlinTMO` uses the recommended display gamma 1.6.
- Other operators produce linear output and receive the selected display
  gamma when `--output-mode sdr` is used.

RGB values above one are gamut-compressed by scaling all three channels
together. This preserves hue better than independent channel clipping.

## Reproduce The 31-TMO Check

The validation script uses resized images by default to keep local methods
practical:

```bash
python tools/test_all_tmos.py \
  --input hdrimage \
  --output tmo_test_results \
  --max-side 256
```

Use original resolution:

```bash
python tools/test_all_tmos.py --input hdrimage --output tmo_full --max-side 0
```

It writes PNG previews, `results.csv`, `inputs.json`, and `summary.json`.
The current three-image check passes all `93/93` combinations.

## Validation Status

- 336/336 MATLAB source files have matching Python symbols.
- All 31 single-image TMO entry points are registered in the CLI.
- Core formulas and defaults were checked against their MATLAB source.
- Some backend-dependent operations are numerical approximations rather
  than pixel-identical ports. See [docs/TMO_AUDIT.md](docs/TMO_AUDIT.md).
- Formula correspondence does not replace MATLAB/Octave golden-image tests.

More detail:

- [TMO input/output domains](docs/TMO_INPUT_OUTPUT.md)
- [TMO source audit](docs/TMO_AUDIT.md)
- [Migration inventory](docs/MIGRATION.md)
