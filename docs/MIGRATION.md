# HDR Toolbox Python migration

The source toolbox contains 336 MATLAB files with 353 declared functions,
plus five C/C++ files used for EXR and bilateral filtering.

Migration is tracked per original source file in
`docs/migration_inventory.csv`. Regenerate it with:

```bash
python tools/build_migration_inventory.py
```

Current implementation status:

- 336 MATLAB files have matching Python implementations.
- 0 MATLAB files are partially implemented.
- 0 MATLAB files are pending.
- 95 Python regression tests pass.

Implemented module groups currently include the color-space and format layer,
analysis helpers, color correction, basic metrics, global tone-mapping
operators, expansion operators, bilateral-filter compatibility, Laplacian
pyramids, configurable LDR stacks, HDR merging, Debevec CRF estimation,
veiling-glare/PSF estimation, environment-map conversion, foundational IBL
sampling, PFM/RGBE compatibility I/O and shared image-processing utilities.
Alignment, motion deghosting, TMQI and multiple-exposure PSNR are also
available, together with adaptive IBL sampling, configurable HDR/LDR video
I/O, batch conversion, HDR JPEG2000/Boschetti compression, multiple CRF
estimators, RAW/RAW-stack support and scriptable toolbox utilities.
Local tone-mapping support now also includes visual adaptation and sigmoid
response models, dynamic-range segmentation, Krawczyk partitioning,
Ashikhmin/Reinhard filters, and sparse Poisson/Lischinski solvers.
The corresponding local and exposure-fusion operators include Ashikhmin,
Banterle, Bruce, Chiu, Durand, Kim-Kautz, Krawczyk, Lischinski, Mertens,
Pattanaik, Raman, Van Hateren, Ward histogram adjustment, and the
Yee-Pattanaik adaptation variants.
Video tone mapping and HDR-video generation are available as configurable
array/`VideoStream` APIs. PU2 perceptual encoding and a cross-platform
software `HDRMonitorDriver` replacement complete the source-file inventory.

Status meanings:

- `implemented`: every declared MATLAB function in the file has a Python symbol.
- `partial`: at least one declared function has been ported.
- `pending`: no declared function has been ported yet.

Having a Python symbol is only the first gate. Numerical parity tests against
MATLAB/Octave are required before a module is considered verified.

## Dependency order

1. Core utilities, color spaces, formats, analysis and image I/O.
2. Tone mapping utilities and global operators.
3. Local tone mapping, pyramids, filters and Poisson solvers.
4. HDR generation, CRF estimation, stacks, alignment and deghosting.
5. Metrics, expansion operators, environment maps and IBL.
6. Video, compression, batch tools and interactive visualization.
