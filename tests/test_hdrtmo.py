import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from hdrtmo.io import (
    ReaderConfig,
    hdrimread,
    hdrimwrite,
    ldrimread,
    pq_to_linear,
    read_hdr,
    read_pfm,
    read_raw,
    read_raw_info,
    write_pfm,
)
from hdrtmo.operators import gamma_tmo, luminance, reinhard_tmo
from hdrtmo.colorspace import (
    convert_rgb2020_to_rgb709,
    convert_rgb_to_srgb,
    convert_rgb_to_xyz,
    convert_xyz_to_cielab,
)
from hdrtmo.formats import float_to_rgbe, rgbe_to_float
from hdrtmo.tmo import (
    best_exposure_tmo,
    drago_tmo,
    exponential_tmo,
    ferwerda_tmo,
    logarithmic_tmo,
    normalize_tmo,
    reinhard_devlin_tmo,
    reinhard_robust_tmo,
    schlick_tmo,
    select_overexposed_tmo,
    tumblin_tmo,
    ward_global_tmo,
)
from hdrtmo.eo import (
    akyuz_eo,
    kovaleski_oliveira_eo,
    kuo_eo,
    landis_eo,
    masia_eo,
)
from hdrtmo.pyramids import (
    gaussian_pyramid,
    laplacian_pyramid,
    pyramid_blend,
    reconstruct_pyramid,
)
from hdrtmo.generation import (
    apply_crf,
    build_hdr,
    compute_glare_image,
    debevec_crf,
    estimate_psf,
    gsolve,
    remove_crf,
    simulate_spatial_exposure,
    weight_function,
)
from hdrtmo.stacks import (
    compute_stack_histogram,
    read_ldr_stack,
    read_raw_stack,
    read_raw_stack_info,
    sort_stack,
    write_ldr_stack,
)
from hdrtmo.environment import (
    align_ll_panoramas,
    angular_mask,
    change_mapping,
    cross_mask,
    direction_to_ll,
    ll_to_direction,
    rotate_y_ll,
)
from hdrtmo.ibl import (
    create_1d_distribution,
    diffuse_convolution_sh,
    export_lights,
    importance_sampling,
    median_cut,
    sample_1d_distribution,
    uniform_sampling,
    variance_minimization_sampling,
)
from hdrtmo.utilities import (
    bitblit,
    compute_connected_components,
    estimate_homography,
    filter_firefly,
    image_warp,
)
from hdrtmo.tmo_local_utils import (
    ashikhmin_filtering,
    ciecam02_chromatic_adaptation,
    create_segments,
    generate_masks,
    histogram_ceiling,
    krawczyk_image_partition,
    krawczyk_kmeans,
    lischinski_minimization,
    poisson_solver,
    reinhard_filtering,
    saturation_parameters,
    sigmoid_response,
    tvi_ashikhmin,
)
from hdrtmo.tmo_local import (
    ashikhmin_tmo,
    bruce_expo_blend_tmo,
    durand_tmo,
    kim_kautz_consistent_tmo,
    kuang_tmo,
    lischinski_tmo,
    mertens_tmo,
    pattanaik_tmo,
    raman_tmo,
    ward_hist_adj_tmo,
    van_hateren_tmo,
    yp_ferwerda_tmo,
    yp_tumblin_tmo,
    yp_ward_global_tmo,
)
from hdrtmo.video_tmo import gamma_exposure_tmov, kiser_tmov, static_tmov
from hdrtmo.generation_video import (
    banterle_enhance_ldr_frame,
    create_hdrv_from_image,
    find_hdr_ldr_crf,
    find_hdr_ldr_scale,
)
from hdrtmo.native_visualization import HDRMonitorDriver
from hdrtmo.runner import (
    TMO_SPECS,
    convert_working_primaries,
    get_tmo,
    resolve_reader,
    run_tmo,
)
from hdrtmo.alignment import (
    sift_image_alignment,
    ward_compute_threshold,
    ward_get_exposure_shift,
    ward_image_alignment,
)
from hdrtmo.deghosting import gallo_reference_image, pece_kautz_merge, pece_kautz_move_mask
from hdrtmo.metrics import multiple_exposure_psnr, pu2_encode, tmqi, tmqi_statistical_naturalness
from hdrtmo.video import (
    check_video_resolution,
    hdrv_analysis,
    hdrv_get_frame,
    hdrvread,
    ldrv_analysis,
    ldrv_get_frame,
    ldrvread,
)
from hdrtmo.batch import convert_hdr_to_hdr, convert_hdr_to_ldr, convert_ldr_to_ldr
from hdrtmo.compression import (
    boschetti_decode,
    boschetti_encode,
    hdr_jpeg2000_decode,
    hdr_jpeg2000_encode,
)
from hdrtmo.generation_crf import (
    akyuz_ldr_stack_denoise,
    mann_picard_crf,
    mitsunaga_nayar_crf,
    raw_crf,
    robertson_crf,
)
from hdrtmo.tools import (
    automatic_exposure,
    false_color,
    hdr_image_crop,
    image_color_calibration,
    image_white_balance,
    rotate_ll_gui,
)


class ReaderTests(unittest.TestCase):
    def test_developed_raw_fallback_and_stack(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for index, value in enumerate((1000, 2000)):
                frame = np.full((6, 8, 3), value, dtype=np.uint16)
                path = Path(directory) / f"{index:03d}.tiff"
                self.assertTrue(cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)))
            image, metadata, saturation = read_raw(Path(directory) / "000.tiff")
            info = read_raw_info(Path(directory) / "000.tiff")
            stack = read_raw_stack(directory, "tiff")
            exposures = read_raw_stack_info(directory, "tiff")
        self.assertEqual(image.shape, (6, 8, 3))
        self.assertEqual(metadata["Width"], 8)
        self.assertEqual(info["Height"], 6)
        self.assertEqual(saturation, 2**12 - 1)
        self.assertEqual(stack.shape, (6, 8, 3, 2))
        self.assertEqual(exposures.shape, (2,))

    def test_pfm_round_trip(self) -> None:
        source = np.linspace(0, 10, 4 * 5 * 3).reshape(4, 5, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.pfm"
            self.assertTrue(write_pfm(source, path))
            restored = read_pfm(path)
        np.testing.assert_allclose(restored, source, rtol=1e-6, atol=1e-6)

    def test_hdr_compatibility_round_trip(self) -> None:
        source = np.linspace(0.01, 10, 8 * 9 * 3).reshape(8, 9, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.hdr"
            self.assertTrue(hdrimwrite(source, path))
            restored, metadata = hdrimread(path)
        self.assertTrue(metadata["loaded"])
        np.testing.assert_allclose(restored, source, rtol=0.02, atol=0.04)

    def test_ldr_compatibility_reader_normalizes_bit_depth(self) -> None:
        source = np.array([[0, 65535]], dtype=np.uint16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            self.assertTrue(cv2.imwrite(str(path), source))
            restored = ldrimread(path)
        np.testing.assert_allclose(restored, [[0, 1]])

    def test_png_raw_mode_preserves_integer_values(self) -> None:
        source = np.array([[0, 1024], [32768, 65535]], dtype=np.uint16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "linear.png"
            self.assertTrue(cv2.imwrite(str(path), source))
            image, metadata = read_hdr(
                path,
                ReaderConfig(
                    preset="custom",
                    png_mode="raw",
                    transfer="linear",
                    primaries="rec709",
                ),
            )

        np.testing.assert_array_equal(image, source.astype(np.float64))
        self.assertEqual(metadata["png_mode"], "raw")

    def test_png_normalized_mode_scales_values(self) -> None:
        source = np.array([[0, 65535]], dtype=np.uint16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "linear.png"
            self.assertTrue(cv2.imwrite(str(path), source))
            image, _ = read_hdr(
                path,
                ReaderConfig(
                    preset="custom",
                    png_mode="normalized",
                    transfer="linear",
                    primaries="rec709",
                ),
            )

        np.testing.assert_allclose(image, [[0.0, 1.0]])

    def test_pq_reference_points(self) -> None:
        decoded = pq_to_linear(np.array([0.0, 1.0]))
        np.testing.assert_allclose(decoded, [0.0, 10000.0], atol=1e-8)

    def test_hdtv1k_preset_normalizes_then_decodes_pq(self) -> None:
        source = np.array([[0, 65535]], dtype=np.uint16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pq.png"
            self.assertTrue(cv2.imwrite(str(path), source))
            image, metadata = read_hdr(path)

        np.testing.assert_allclose(image, [[0.0, 10000.0]], atol=1e-8)
        self.assertEqual(metadata["primaries"], "rec2020")


class OperatorTests(unittest.TestCase):
    def test_luminance_uses_srgb_primaries(self) -> None:
        image = np.array([[[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]]])
        np.testing.assert_allclose(luminance(image), [[1.0, 0.2126]])

    def test_gamma_tmo_applies_exposure_and_clamps(self) -> None:
        image = np.array([[0.0, 0.25, 1.0]])
        np.testing.assert_allclose(
            gamma_tmo(image, gamma=2.0, fstop=0.0),
            [[0.0, 0.5, 1.0]],
        )

    def test_reinhard_returns_finite_output(self) -> None:
        image = np.array([[[0.1, 0.2, 0.3], [2.0, 1.0, 0.5]]])
        mapped, alpha, white_point = reinhard_tmo(image)
        self.assertTrue(np.isfinite(mapped).all())
        self.assertGreater(alpha, 0.0)
        self.assertGreater(white_point, 0.0)


class FoundationTests(unittest.TestCase):
    def test_rgb_xyz_round_trip(self) -> None:
        image = np.array([[[0.1, 0.4, 0.8], [1.0, 0.5, 0.2]]])
        restored = convert_rgb_to_xyz(convert_rgb_to_xyz(image), inverse=True)
        np.testing.assert_allclose(restored, image, atol=1e-12)

    def test_srgb_round_trip(self) -> None:
        image = np.linspace(0, 1, 30).reshape(2, 5, 3)
        restored = convert_rgb_to_srgb(convert_rgb_to_srgb(image), inverse=True)
        np.testing.assert_allclose(restored, image, atol=1e-12)

    def test_cielab_round_trip(self) -> None:
        xyz = np.array([[[0.2, 0.3, 0.4], [0.8, 0.7, 0.6]]])
        restored = convert_xyz_to_cielab(convert_xyz_to_cielab(xyz), inverse=True)
        np.testing.assert_allclose(restored, xyz, atol=1e-12)

    def test_rgbe_round_trip_has_expected_quantization(self) -> None:
        image = np.array([[[0.1, 1.0, 10.0], [50.0, 20.0, 2.0]]])
        restored = rgbe_to_float(float_to_rgbe(image))
        np.testing.assert_allclose(restored, image, rtol=0.07, atol=0.13)


class GlobalTMOTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.array(
            [
                [[0.01, 0.02, 0.04], [0.2, 0.4, 0.8]],
                [[1.0, 2.0, 4.0], [10.0, 20.0, 40.0]],
            ],
            dtype=np.float64,
        )

    def test_global_tmos_return_finite_images(self) -> None:
        operators = (
            normalize_tmo,
            exponential_tmo,
            logarithmic_tmo,
            schlick_tmo,
            ward_global_tmo,
            drago_tmo,
            tumblin_tmo,
            reinhard_devlin_tmo,
            ferwerda_tmo,
        )
        for operator in operators:
            with self.subTest(operator=operator.__name__):
                output = operator(self.image)
                self.assertEqual(output.shape, self.image.shape)
                self.assertTrue(np.isfinite(output).all())
                self.assertGreaterEqual(float(np.min(output)), 0.0)

    def test_reinhard_robust_returns_parameters(self) -> None:
        output, alpha, average = reinhard_robust_tmo(self.image)
        self.assertTrue(np.isfinite(output).all())
        self.assertGreater(alpha, 0.0)
        self.assertGreater(average, 0.0)

    def test_gamma_clamps_without_maximum_normalization(self) -> None:
        image = np.array([[[0.25, 1.0, 4.0]]])
        output = gamma_tmo(image)
        np.testing.assert_allclose(output, [[[0.25 ** (1 / 2.2), 1.0, 1.0]]])

    def test_logarithmic_normalizes_luminance_not_rgb_channels(self) -> None:
        image = np.array([[[4.0, 1.0, 0.0]], [[1.0, 1.0, 1.0]]])
        output = logarithmic_tmo(image)
        self.assertAlmostEqual(float(np.max(luminance(output))), 1.0)
        self.assertGreater(float(np.max(output)), 1.0)

    def test_rec2020_hdr_conversion_can_preserve_values_above_one(self) -> None:
        image = np.full((1, 1, 3), 4.0)
        converted = convert_rgb2020_to_rgb709(image, clip=False)
        self.assertGreater(float(np.max(converted)), 1.0)

    def test_exposure_operators_return_positive_exposure(self) -> None:
        for operator, arguments in (
            (best_exposure_tmo, {"method": "mean"}),
            (select_overexposed_tmo, {"percent": 10.0}),
        ):
            output, exposure = operator(self.image, **arguments)
            self.assertTrue(np.isfinite(output).all())
            self.assertGreater(exposure, 0.0)
            self.assertLessEqual(float(np.max(output)), 1.0)

    def test_selected_overexposure_matches_requested_fraction(self) -> None:
        image = np.geomspace(0.001, 100, 32 * 32 * 3).reshape(32, 32, 3)
        output, _ = select_overexposed_tmo(image, percent=5.0)
        fraction = np.mean(np.power(output, 1 / 2.2) >= 0.95)
        self.assertAlmostEqual(float(fraction), 0.05, places=2)


class ExpansionOperatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.linspace(0.05, 0.95, 4 * 4 * 3).reshape(4, 4, 3)

    def test_direct_expansion_operators(self) -> None:
        outputs = (
            akyuz_eo(self.image, 1000, 1.0, 2.2),
            landis_eo(self.image, 2.0, 0.5, 1000, 2.2),
            masia_eo(self.image, 1000, noise_removal=False, gamma_removal=2.2)[0],
            kuo_eo(self.image, 1000, 2.2),
        )
        for output in outputs:
            self.assertEqual(output.shape, self.image.shape)
            self.assertTrue(np.isfinite(output).all())
            self.assertGreaterEqual(float(np.min(output)), 0.0)

    def test_kovaleski_returns_expansion_map(self) -> None:
        output, expansion_map = kovaleski_oliveira_eo(
            self.image,
            "image",
            sigma_spatial=1.0,
            sigma_range=0.1,
            display_min=0.3,
            display_max=1000,
            gamma_removal=2.2,
        )
        self.assertEqual(output.shape, self.image.shape)
        self.assertEqual(expansion_map.shape, self.image.shape[:2])
        self.assertTrue(np.isfinite(output).all())


class PyramidTests(unittest.TestCase):
    def test_laplacian_round_trip(self) -> None:
        image = np.arange(7 * 9, dtype=np.float64).reshape(7, 9) / 63
        restored = reconstruct_pyramid(laplacian_pyramid(image))
        np.testing.assert_allclose(restored, image, atol=1e-12)

    def test_gaussian_pyramid_reduces_resolution(self) -> None:
        pyramid = gaussian_pyramid(np.ones((8, 10)), max_levels=2)
        self.assertEqual(len(pyramid.details), 2)
        self.assertEqual(pyramid.base.shape, (2, 3))

    def test_blend_respects_constant_weights(self) -> None:
        first = np.ones((8, 8, 3))
        second = np.zeros_like(first)
        np.testing.assert_allclose(
            pyramid_blend(first, second, np.ones((8, 8))),
            first,
            atol=1e-12,
        )


class GenerationTests(unittest.TestCase):
    def test_gamma_crf_round_trip(self) -> None:
        image = np.linspace(0, 1, 30).reshape(2, 5, 3)
        restored = remove_crf(apply_crf(image, "gamma", 2.2), "gamma", 2.2)
        np.testing.assert_allclose(restored, image, atol=1e-12)

    def test_weight_functions_stay_in_unit_interval(self) -> None:
        image = np.linspace(0, 1, 30).reshape(2, 5, 3)
        for kind in ("all", "identity", "reverse", "box", "Robertson", "hat", "Deb97"):
            with self.subTest(kind=kind):
                weight = weight_function(image, kind)
                self.assertGreaterEqual(float(np.min(weight)), 0.0)
                self.assertLessEqual(float(np.max(weight)), 1.0)

    def test_linear_build_hdr_recovers_radiance(self) -> None:
        radiance = np.linspace(0.2, 0.4, 4 * 5 * 3).reshape(4, 5, 3)
        exposures = np.array([0.5, 1.0, 2.0])
        stack = np.stack([radiance * exposure for exposure in exposures], axis=-1)
        restored, _ = build_hdr(
            stack,
            exposures,
            linearization="linear",
            weight_type="all",
            merge_type="linear",
        )
        np.testing.assert_allclose(restored, radiance, atol=1e-12)

    def test_spatial_exposure_pattern(self) -> None:
        image = np.ones((4, 4, 3))
        output = simulate_spatial_exposure(image, np.array([-2, -1, 0, 1]))
        self.assertEqual(output.shape, image.shape)
        self.assertTrue(np.isfinite(output).all())

    def test_gsolve_recovers_monotonic_response(self) -> None:
        exposures = np.array([0.25, 0.5, 1.0, 2.0])
        radiance = np.linspace(0.05, 0.45, 64)
        samples = np.rint(
            np.clip((radiance[:, None] * exposures[None, :]) ** (1 / 2.2), 0, 1) * 255
        )
        weights = weight_function(np.linspace(0, 1, 256), "Deb97")
        response = np.exp(gsolve(samples, np.log(exposures), 64, weights))
        self.assertGreater(np.corrcoef(response, np.linspace(0, 1, 256) ** 2.2)[0, 1], 0.99)

    def test_debevec_crf_and_automatic_lut_merge(self) -> None:
        radiance = np.linspace(0.02, 0.45, 12 * 12 * 3).reshape(12, 12, 3)
        exposures = np.array([0.25, 0.5, 1.0, 2.0])
        stack = np.stack(
            [np.clip(radiance * exposure, 0, 1) ** (1 / 2.2) for exposure in exposures],
            axis=-1,
        )
        response, maximum = debevec_crf(stack, exposures, samples=48)
        self.assertEqual(response.shape, (256, 3))
        self.assertEqual(maximum.shape, (3,))
        self.assertTrue(np.all(np.diff(response, axis=0) >= -1e-8))

        restored, estimated = build_hdr(
            stack,
            exposures,
            linearization="LUT",
            function=None,
            weight_type="Deb97",
            merge_type="linear",
        )
        self.assertEqual(np.asarray(estimated).shape, (256, 3))
        scale = np.sum(restored * radiance) / np.sum(restored**2)
        np.testing.assert_allclose(restored * scale, radiance, rtol=0.15, atol=0.02)

    def test_glare_model_is_bounded_by_source(self) -> None:
        image = np.full((9, 9, 3), 0.05)
        image[4, 4, :] = 10
        glare = compute_glare_image(
            image,
            np.ones((3, 3)),
            np.array([[4], [4]]),
            np.array([0, 0.01, 0.01, 0.01]),
        )
        self.assertEqual(glare.shape, image.shape)
        self.assertTrue(np.all(glare <= image + 1e-12))
        self.assertGreater(float(np.sum(glare)), 0)

    def test_estimate_psf_returns_kernel_and_hot_pixels(self) -> None:
        yy, xx = np.mgrid[:32, :32]
        radius = np.maximum(np.hypot(xx - 16, yy - 16), 1)
        image = np.repeat((0.001 + 1 / radius**2)[..., None], 3, axis=2)
        image[16, 16, :] = 100
        psf, coefficients, hot_pixels = estimate_psf(image, working_width=32)
        self.assertEqual(psf.shape, (33, 33))
        self.assertEqual(coefficients.shape, (4,))
        self.assertEqual(hot_pixels.shape[0], 2)
        self.assertTrue(np.isfinite(psf).all())


class EnvironmentMapTests(unittest.TestCase):
    def test_ll_direction_coordinates_round_trip(self) -> None:
        rows, columns = 16, 32
        directions = ll_to_direction(rows, columns)
        x, y = direction_to_ll(directions, rows, columns)
        expected_x, expected_y = np.meshgrid(np.arange(columns), np.arange(rows))
        np.testing.assert_allclose(x[1:], expected_x[1:], atol=1e-10)
        np.testing.assert_allclose(y, expected_y, atol=1e-10)
        np.testing.assert_allclose(directions[0, :, 1], 1, atol=1e-12)

    def test_mapping_masks_have_expected_shapes(self) -> None:
        angular = angular_mask(16, 16)
        cross = cross_mask(16, 12)
        self.assertEqual(angular.shape, (16, 16, 3))
        self.assertEqual(cross.shape, (16, 12, 3))
        self.assertGreater(float(np.sum(angular)), 0)
        self.assertEqual(int(np.sum(cross[..., 0])), 6 * 4 * 4)

    def test_ll_rotation_and_alignment(self) -> None:
        image = np.zeros((8, 16, 3))
        image[:, 2:5, :] = 1
        shifted = rotate_y_ll(image, 90)
        restored, rotation, error = align_ll_panoramas(shifted, image)
        np.testing.assert_allclose(restored, image)
        self.assertEqual(rotation, 12)
        self.assertAlmostEqual(error, 0)

    def test_ll_to_angular_mapping(self) -> None:
        image = np.ones((16, 32, 3))
        angular = change_mapping(image, "LL", "Angular")
        self.assertEqual(angular.shape, (16, 16, 3))
        self.assertTrue(np.isfinite(angular).all())
        np.testing.assert_allclose(angular[8, 8], 1, atol=1e-8)
        np.testing.assert_allclose(angular[0, 0], 0, atol=1e-8)


class IBLTests(unittest.TestCase):
    def test_distribution_sampling(self) -> None:
        distribution = create_1d_distribution(np.array([1, 2, 1]))
        np.testing.assert_allclose(distribution.pdf, [0.25, 0.5, 0.25])
        index, probability = sample_1d_distribution(distribution, 0.7)
        self.assertEqual(index, 1)
        self.assertEqual(probability, 0.5)

    def test_uniform_sampling_preserves_total_energy(self) -> None:
        image = np.ones((16, 32, 3))
        light_map, lights = uniform_sampling(image, light_count=16)
        self.assertEqual(len(lights), 16)
        np.testing.assert_allclose(np.sum(light_map, axis=(0, 1)), np.sum(image, axis=(0, 1)))

    def test_importance_sampling_returns_requested_count(self) -> None:
        image = np.ones((8, 16, 3))
        sample_map, samples = importance_sampling(
            image,
            sample_count=32,
            rng=np.random.default_rng(1),
        )
        self.assertEqual(len(samples), 32)
        self.assertEqual(float(np.sum(sample_map)), 32)
        self.assertTrue(all(np.isfinite(sample.direction).all() for sample in samples))

    def test_diffuse_spherical_harmonics_is_finite(self) -> None:
        image = np.ones((12, 24, 3))
        convolved, coefficients = diffuse_convolution_sh(image)
        self.assertEqual(convolved.shape, image.shape)
        self.assertEqual(coefficients.shape, (3, 9))
        self.assertTrue(np.isfinite(convolved).all())

    def test_adaptive_sampling_preserves_energy(self) -> None:
        image = np.full((16, 32, 3), 0.1)
        image[3:7, 20:25] = 10
        for sampler in (median_cut, variance_minimization_sampling):
            with self.subTest(sampler=sampler.__name__):
                light_map, lights = sampler(image, light_count=8)
                self.assertEqual(len(lights), 8)
                np.testing.assert_allclose(
                    np.sum(light_map, axis=(0, 1)),
                    np.sum(image, axis=(0, 1)),
                    rtol=1e-12,
                )

    def test_export_lights_writes_directions_and_colors(self) -> None:
        image = np.ones((8, 16, 3))
        _, lights = median_cut(image, light_count=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lights.txt"
            export_lights(lights, path)
            content = path.read_text()
        self.assertIn("Num: 4", content)
        self.assertEqual(content.count("Dir:"), 4)
        self.assertEqual(content.count("Col:"), 4)


class UtilityTests(unittest.TestCase):
    def test_homography_recovers_translation(self) -> None:
        first = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        second = first + np.array([3, 2])
        homography = estimate_homography(first, second)
        homogeneous = np.column_stack((first, np.ones(4)))
        transformed = (homography @ homogeneous.T).T
        transformed = transformed[:, :2] / transformed[:, 2:]
        np.testing.assert_allclose(transformed, second, atol=1e-10)

    def test_dense_warp_uses_relative_offsets(self) -> None:
        image = np.arange(25, dtype=np.float32).reshape(5, 5)
        offsets = np.zeros((5, 5, 2), dtype=np.float32)
        offsets[..., 0] = 1
        warped = image_warp(image, offsets)
        np.testing.assert_array_equal(warped[:, 1:], image[:, :-1])

    def test_connected_components_separates_equal_regions(self) -> None:
        image = np.array([[1, 0, 1], [1, 0, 0], [0, 0, 1]])
        labels, values, shifts = compute_connected_components(image, 4)
        self.assertEqual(set(values), {0, 1})
        self.assertGreater(len(np.unique(labels)), 2)
        self.assertEqual(shifts[0], 0)

    def test_bitblit_and_firefly_filter(self) -> None:
        image = np.zeros((5, 5, 1))
        sprite = np.ones((3, 3, 1))
        composited = bitblit(image, sprite, np.array([[2, 2]]))
        self.assertEqual(float(np.sum(composited)), 9)
        composited[2, 2, 0] = np.nan
        repaired = filter_firefly(composited)
        self.assertTrue(np.isfinite(repaired).all())


class AlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(4)
        self.reference = rng.random((96, 128, 3))
        self.reference[20:70, 30:35] = 1
        self.reference[45:50, 15:100] = 0

    def test_ward_threshold_is_exposure_invariant(self) -> None:
        first, first_mask = ward_compute_threshold(self.reference)
        second, second_mask = ward_compute_threshold(self.reference * 0.4)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_mask.shape, first.shape)
        self.assertEqual(second_mask.shape, second.shape)

    def test_ward_alignment_recovers_translation(self) -> None:
        matrix = np.float32(((1, 0, 5), (0, 1, -3)))
        shifted = cv2.warpAffine(
            self.reference,
            matrix,
            (self.reference.shape[1], self.reference.shape[0]),
            borderMode=cv2.BORDER_REPLICATE,
        )
        shift = ward_get_exposure_shift(self.reference, shifted, shift_bits=5)
        np.testing.assert_array_equal(shift, [-5, 3])
        aligned, information = ward_image_alignment(self.reference, shifted, rotation=False)
        np.testing.assert_array_equal(information[0], shift)
        np.testing.assert_allclose(aligned[8:-8, 8:-8], self.reference[8:-8, 8:-8], atol=1e-12)

    def test_sift_alignment_recovers_translation(self) -> None:
        image = np.zeros((160, 200, 3), dtype=np.float32)
        rng = np.random.default_rng(8)
        for _ in range(40):
            x, y = rng.integers(10, 190), rng.integers(10, 150)
            cv2.circle(image, (int(x), int(y)), int(rng.integers(2, 7)), tuple(rng.random(3).tolist()), -1)
        matrix = np.float32(((1, 0, 7), (0, 1, 4)))
        shifted = cv2.warpAffine(image, matrix, (200, 160))
        aligned, homography = sift_image_alignment(image, shifted, max_iterations=200)
        np.testing.assert_allclose(homography[:2, 2], [-7, -4], atol=0.6)
        np.testing.assert_allclose(aligned[12:-12, 12:-12], image[12:-12, 12:-12], atol=0.08)


class DeghostingTests(unittest.TestCase):
    def test_reference_selects_least_clipped_frame(self) -> None:
        stack = np.empty((8, 8, 3, 3))
        stack[..., 0] = 0
        stack[..., 1] = 0.5
        stack[..., 2] = 1
        self.assertEqual(gallo_reference_image(stack), 1)

    def test_movement_mask_detects_changed_object(self) -> None:
        stack = np.full((32, 32, 3, 3), 0.2)
        stack[8:16, 5:13, :, 0] = 0.9
        stack[8:16, 18:26, :, 1] = 0.9
        mask, count = pece_kautz_move_mask(
            stack,
            iterations=1,
            erosion_size=1,
            dilation_size=2,
        )
        self.assertGreaterEqual(count, 1)
        self.assertTrue(np.any(mask >= 1))
        self.assertTrue(np.any(mask == -1))

    def test_deghosting_merge_returns_finite_image(self) -> None:
        base = np.linspace(0.1, 0.8, 32 * 32 * 3).reshape(32, 32, 3)
        stack = np.stack((base * 0.6, base, np.clip(base * 1.4, 0, 1)), axis=-1)
        stack[10:18, 5:13, :, 0] = 1
        stack[10:18, 19:27, :, 2] = 1
        output = pece_kautz_merge(
            stack,
            iterations=1,
            erosion_size=1,
            dilation_size=2,
        )
        self.assertEqual(output.shape, base.shape)
        self.assertTrue(np.isfinite(output).all())
        self.assertGreaterEqual(float(np.min(output)), 0)
        self.assertLessEqual(float(np.max(output)), 1)


class AdvancedMetricTests(unittest.TestCase):
    def test_tmqi_scores_identical_tone_mapping(self) -> None:
        hdr = np.linspace(0.01, 20, 64 * 64 * 3).reshape(64, 64, 3)
        ldr = np.clip(hdr / np.max(hdr), 0, 1)
        quality, structural, naturalness, maps, local = tmqi(hdr, ldr)
        self.assertTrue(0 <= quality <= 1)
        self.assertTrue(0 <= structural <= 1)
        self.assertTrue(0 <= naturalness <= 1)
        self.assertEqual(len(maps), 5)
        self.assertEqual(local.shape, (5,))

    def test_tmqi_naturalness_accepts_normalized_input(self) -> None:
        image = np.linspace(0, 1, 32 * 32).reshape(32, 32)
        self.assertAlmostEqual(
            tmqi_statistical_naturalness(image),
            tmqi_statistical_naturalness(image * 255),
        )

    def test_multiple_exposure_psnr_identical_is_infinite_sentinel(self) -> None:
        image = np.linspace(0.01, 4, 16 * 16 * 3).reshape(16, 16, 3)
        score, maximum, minimum = multiple_exposure_psnr(image, image)
        self.assertEqual(score, 1000)
        self.assertGreaterEqual(maximum, minimum)


class VideoIOTests(unittest.TestCase):
    def test_resolution_table(self) -> None:
        self.assertTrue(check_video_resolution(1080, 1920))
        self.assertFalse(check_video_resolution(123, 456))

    def test_ldr_directory_video_read_and_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for index, value in enumerate((0.2, 0.5, 0.8)):
                frame = np.full((12, 16, 3), round(value * 255), dtype=np.uint8)
                self.assertTrue(
                    cv2.imwrite(
                        str(Path(directory) / f"{index:03d}.png"),
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                    )
                )
            video = ldrvread(directory)
            frame, video = ldrv_get_frame(video, 1)
            np.testing.assert_allclose(frame, 0.5, atol=1 / 255)
            video, statistics, histograms = ldrv_analysis(video, histogram=True)
        self.assertEqual(statistics.shape, (3, 7))
        self.assertEqual(histograms.shape, (3, 256))
        self.assertFalse(video.stream_open)

    def test_hdr_directory_video_uses_reader_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for index, value in enumerate((1.0, 2.0)):
                write_pfm(np.full((12, 16, 3), value), Path(directory) / f"{index:03d}.pfm")
            video = hdrvread(
                directory,
                ReaderConfig(
                    preset="custom",
                    png_mode="normalized",
                    transfer="linear",
                    primaries="rec709",
                ),
            )
            frame, video = hdrv_get_frame(video, 1)
            np.testing.assert_allclose(frame, 2)
            video, statistics, histograms = hdrv_analysis(video, histogram=True)
        self.assertEqual(statistics.shape, (2, 7))
        self.assertEqual(histograms.shape, (2, 4096))
        self.assertFalse(video.stream_open)


class BatchTests(unittest.TestCase):
    def test_hdr_and_ldr_directory_conversions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hdr_input = root / "hdr_input"
            hdr_output = root / "hdr_output"
            ldr_output = root / "ldr_output"
            converted_output = root / "converted"
            for path in (hdr_input, hdr_output, ldr_output, converted_output):
                path.mkdir()
            source = np.linspace(0.01, 4, 16 * 16 * 3).reshape(16, 16, 3)
            write_pfm(source, hdr_input / "source.pfm")

            hdr_files = convert_hdr_to_hdr(
                "pfm",
                "hdr",
                hdr_input,
                hdr_output,
                ReaderConfig(preset="custom", transfer="linear", primaries="rec709"),
            )
            ldr_files = convert_hdr_to_ldr(
                "pfm",
                "png",
                directory=hdr_input,
                output_directory=ldr_output,
                reader_config=ReaderConfig(preset="custom", transfer="linear", primaries="rec709"),
            )
            converted_files = convert_ldr_to_ldr("png", "jpg", ldr_output, converted_output)

            self.assertEqual(len(hdr_files), 1)
            self.assertEqual(len(ldr_files), 1)
            self.assertEqual(len(converted_files), 1)
            self.assertTrue(all(path.exists() for path in hdr_files + ldr_files + converted_files))


class CompressionTests(unittest.TestCase):
    def test_hdr_jpeg2000_round_trip(self) -> None:
        image = np.linspace(0.01, 100, 64 * 64 * 3).reshape(64, 64, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.jp2"
            hdr_jpeg2000_encode(image, path, compression_ratio=1)
            restored = hdr_jpeg2000_decode(path)
        np.testing.assert_allclose(restored, image, rtol=5e-3, atol=2e-3)

    def test_boschetti_round_trip(self) -> None:
        image = np.linspace(0.01, 20, 64 * 64 * 3).reshape(64, 64, 3)
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "image"
            boschetti_encode(image, prefix, rate_e=1, rate_rgb=1)
            restored = boschetti_decode(prefix)
        self.assertEqual(restored.shape, image.shape)
        self.assertTrue(np.isfinite(restored).all())
        np.testing.assert_allclose(restored, image, rtol=0.03, atol=0.03)


class CRFTests(unittest.TestCase):
    def setUp(self) -> None:
        self.exposures = np.array([0.25, 0.5, 1.0, 2.0])
        radiance = np.linspace(0.03, 0.45, 24 * 24 * 3).reshape(24, 24, 3)
        self.stack = np.stack(
            [np.clip(radiance * exposure, 0, 1) ** (1 / 2.2) for exposure in self.exposures],
            axis=-1,
        )

    def test_robertson_response_is_monotonic(self) -> None:
        response, maximum = robertson_crf(
            self.stack,
            self.exposures,
            max_iterations=8,
            normalize=True,
        )
        self.assertEqual(response.shape, (256, 3))
        self.assertGreater(maximum, 0)
        self.assertTrue(np.all(np.diff(response, axis=0) >= -1e-12))

    def test_mann_picard_recovers_gamma_shape(self) -> None:
        response, parameters = mann_picard_crf(self.stack, self.exposures)
        self.assertEqual(parameters.shape, (2, 3))
        self.assertGreater(np.corrcoef(response[:, 0], np.linspace(0, 1, 256) ** 2.2)[0, 1], 0.98)

    def test_mitsunaga_nayar_returns_polynomial(self) -> None:
        response, polynomial = mitsunaga_nayar_crf(
            self.stack,
            self.exposures,
            degree=3,
            samples=64,
            sampling_strategy="RegularSpatial",
        )
        self.assertEqual(response.shape, (256, 3))
        self.assertEqual(polynomial.shape, (4, 3))
        self.assertTrue(np.isfinite(response).all())

    def test_raw_crf_fits_known_polynomial(self) -> None:
        jpeg = np.linspace(0.05, 0.95, 20 * 20 * 3).reshape(20, 20, 3)
        raw = jpeg**2
        response, polynomial = raw_crf(raw, jpeg, degree=2)
        self.assertEqual(polynomial.shape, (3, 3))
        np.testing.assert_allclose(response[:, 0], np.linspace(0, 1, 256) ** 2, atol=1e-10)

    def test_akyuz_stack_denoise_preserves_shape(self) -> None:
        output = akyuz_ldr_stack_denoise(
            self.stack,
            self.exposures,
            linearization="gamma2.2",
        )
        self.assertEqual(output.shape, self.stack.shape)
        self.assertTrue(np.isfinite(output).all())


class ToolTests(unittest.TestCase):
    def test_false_color_crop_and_exposure(self) -> None:
        image = np.linspace(0.01, 10, 20 * 30 * 3).reshape(20, 30, 3)
        colored, maximum = false_color(image, "log", visualize=False)
        cropped, rectangle = hdr_image_crop(image, (3, 4, 10, 8))
        exposed, exposure = automatic_exposure(image)
        self.assertEqual(colored.shape, image.shape)
        self.assertEqual(maximum, float(np.max(luminance(image))))
        self.assertEqual(cropped.shape, (8, 10, 3))
        self.assertEqual(rectangle, (3, 4, 10, 8))
        self.assertGreater(exposure, 0)
        self.assertEqual(exposed.shape, image.shape)

    def test_white_balance_neutralizes_gray_world(self) -> None:
        image = np.ones((8, 8, 3)) * np.array([0.5, 1.0, 2.0])
        balanced, color, _ = image_white_balance(image, "gray_world")
        np.testing.assert_allclose(np.mean(balanced, axis=(0, 1)), np.mean(color))

    def test_color_calibration_recovers_matrix(self) -> None:
        rng = np.random.default_rng(12)
        source = rng.random((12, 12, 3))
        matrix = np.array([[1.1, 0.1, 0], [0, 0.9, 0.05], [0.02, 0, 1.2]])
        target = source @ matrix
        calibrated, estimated = image_color_calibration(source, target)
        np.testing.assert_allclose(calibrated, target, atol=1e-12)
        np.testing.assert_allclose(estimated, matrix.T, atol=1e-12)

    def test_rotate_ll_identical_points_is_identity(self) -> None:
        image = np.random.default_rng(2).random((16, 32, 3))
        rotated = rotate_ll_gui(image, (8, 8), (8, 8))
        np.testing.assert_allclose(rotated, image, atol=1e-6)


class LocalTMOUtilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.luminance = np.geomspace(0.001, 10, 16 * 16).reshape(16, 16)
        self.image = np.repeat(self.luminance[..., None], 3, axis=2)

    def test_visual_response_helpers_are_finite(self) -> None:
        sigma_cone, sigma_rod = saturation_parameters(self.luminance, self.luminance)
        response = sigmoid_response(self.luminance, 0.8, sigma_cone, np.ones_like(self.luminance))
        self.assertTrue(np.isfinite(sigma_cone).all())
        self.assertTrue(np.isfinite(sigma_rod).all())
        self.assertTrue(np.isfinite(response).all())
        self.assertTrue(np.all(response >= 0))

    def test_ciecam_adaptation_preserves_shape(self) -> None:
        adapted = ciecam02_chromatic_adaptation(self.image, np.array([95.0, 100.0, 109.0]))
        self.assertEqual(adapted.shape, self.image.shape)
        self.assertTrue(np.isfinite(adapted).all())

    def test_local_filters_and_segments(self) -> None:
        adaptation, detail = ashikhmin_filtering(self.luminance, 8)
        reinhard = reinhard_filtering(self.luminance)
        segments = create_segments(self.image)
        masks = generate_masks(segments - np.min(segments) + 1, int(np.ptp(segments)) + 1)
        for result in (adaptation, detail, reinhard, segments, masks):
            self.assertTrue(np.isfinite(result).all())
        self.assertEqual(masks.shape[:2], self.luminance.shape)

    def test_sparse_minimizers_return_finite_fields(self) -> None:
        poisson = poisson_solver(np.zeros((8, 8)))
        result, matrix = lischinski_minimization(
            np.log(self.luminance),
            np.ones_like(self.luminance),
        )
        np.testing.assert_allclose(poisson, 0, atol=1e-12)
        self.assertEqual(matrix.shape, (self.luminance.size, self.luminance.size))
        self.assertTrue(np.isfinite(result).all())

    def test_krawczyk_partition_and_histogram_ceiling(self) -> None:
        histogram = np.ones(64)
        centers, totals = krawczyk_kmeans(np.array([-3.0, 1.0]), histogram)
        framework, distance, final_centers = krawczyk_image_partition(
            centers,
            np.log10(self.luminance),
            np.array([-3.0, 1.0]),
            totals,
        )
        self.assertEqual(framework.shape, self.luminance.shape)
        self.assertEqual(distance.shape, self.luminance.shape)
        self.assertGreaterEqual(final_centers.size, 1)
        self.assertLessEqual(np.max(histogram_ceiling(np.array([100.0, 1.0, 1.0]), 0.5)), 51.0)

    def test_ashikhmin_visibility_is_piecewise_monotonic(self) -> None:
        values = np.array([0.001, 0.0034, 0.5, 1.0, 7.2444, 20.0])
        output = tvi_ashikhmin(values)
        self.assertTrue(np.all(np.diff(output) >= 0))


class LocalTMOTests(unittest.TestCase):
    def setUp(self) -> None:
        luminance_values = np.geomspace(0.01, 100, 16 * 16).reshape(16, 16)
        self.image = np.repeat(luminance_values[..., None], 3, axis=2)

    def test_local_tone_mappers_are_finite(self) -> None:
        for operator in (
            ashikhmin_tmo,
            durand_tmo,
            kim_kautz_consistent_tmo,
            lischinski_tmo,
            pattanaik_tmo,
            ward_hist_adj_tmo,
        ):
            output = operator(self.image)
            self.assertEqual(output.shape, self.image.shape, operator.__name__)
            self.assertTrue(np.isfinite(output).all(), operator.__name__)

    def test_exposure_fusion_operators_are_bounded(self) -> None:
        for operator in (mertens_tmo, raman_tmo, bruce_expo_blend_tmo):
            output = operator(self.image)
            self.assertEqual(output.shape, self.image.shape, operator.__name__)
            self.assertTrue(np.all((output >= 0) & (output <= 1)), operator.__name__)

    def test_mertens_uses_full_normalized_range(self) -> None:
        output = mertens_tmo(self.image)
        self.assertAlmostEqual(float(np.min(output)), 0.0)
        self.assertAlmostEqual(float(np.max(output)), 1.0)

    def test_yee_pattanaik_wrappers_return_adaptation_maps(self) -> None:
        for operator in (yp_ferwerda_tmo, yp_tumblin_tmo, yp_ward_global_tmo):
            output, adaptation = operator(self.image, max_layers=2)
            self.assertEqual(output.shape, self.image.shape, operator.__name__)
            self.assertEqual(adaptation.shape, self.image.shape[:2], operator.__name__)
            self.assertTrue(np.isfinite(output).all(), operator.__name__)

    def test_tone_mapping_does_not_modify_input(self) -> None:
        original = self.image.copy()
        durand_tmo(self.image)
        np.testing.assert_array_equal(self.image, original)

    def test_kuang_tmo_is_bounded(self) -> None:
        output = kuang_tmo(self.image)
        self.assertEqual(output.shape, self.image.shape)
        self.assertTrue(np.all((output >= 0) & (output <= 1)))

    def test_van_hateren_accepts_black_pixels(self) -> None:
        image = self.image.copy()
        image[:4, :4] = 0
        output = van_hateren_tmo(image)
        self.assertTrue(np.isfinite(output).all())


class VideoGenerationAndDisplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frames = np.linspace(0.01, 2, 12 * 14 * 3 * 3).reshape(12, 14, 3, 3)

    def test_video_tone_mappers_preserve_frame_stack(self) -> None:
        for output in (
            gamma_exposure_tmov(self.frames),
            static_tmov(self.frames),
            kiser_tmov(self.frames)[0],
        ):
            self.assertEqual(output.shape, self.frames.shape)
            self.assertTrue(np.isfinite(output).all())

    def test_video_generation_helpers(self) -> None:
        image = np.random.default_rng(4).random((32, 40, 3))
        panning = create_hdrv_from_image(image, rows=8, columns=10, frames=3)
        response = find_hdr_ldr_crf(image, image)
        scale = find_hdr_ldr_scale(image, image)
        enhanced = banterle_enhance_ldr_frame(image, image, image)
        self.assertEqual(panning.shape, (8, 10, 3, 3))
        self.assertEqual(response.shape, (256, 3))
        self.assertGreater(scale, 0)
        self.assertEqual(enhanced.shape, image.shape)

    def test_pu2_and_monitor_encoding_are_finite(self) -> None:
        values = pu2_encode(np.array([1e-5, 0.1, 1.0, 100.0, 1e10]))
        encoded = HDRMonitorDriver().encode(self.frames[..., 0])
        self.assertTrue(np.all(np.diff(values) > 0))
        self.assertTrue(np.isfinite(encoded).all())
        self.assertTrue(np.all((encoded >= 0) & (encoded <= 1)))


class TMORunnerTests(unittest.TestCase):
    def test_registry_contains_all_matlab_tmos(self) -> None:
        self.assertEqual(len(TMO_SPECS), 31)
        self.assertEqual(get_tmo("reinhard").name, "ReinhardTMO")
        self.assertEqual(get_tmo("ReinhardTMO").name, "ReinhardTMO")

    def test_auto_reader_resolves_extension_transfer(self) -> None:
        self.assertEqual(resolve_reader("image.png", "auto", "rec2020").resolved["transfer"], "pq")
        self.assertEqual(resolve_reader("image.exr", "auto", "rec2020").resolved["transfer"], "linear_times_100")

    def test_runner_converts_primaries_and_runs_selected_tmo(self) -> None:
        image = np.ones((4, 4, 3))
        working = convert_working_primaries(image, "rec2020", "rec709")
        restored = convert_working_primaries(working, "rec709", "rec2020")
        output, spec = run_tmo(working, "logarithmic")
        self.assertEqual(spec.name, "LogarithmicTMO")
        self.assertEqual(output.shape, image.shape)
        self.assertTrue(np.isfinite(output).all())
        np.testing.assert_allclose(restored, image, atol=1e-4)


class StackTests(unittest.TestCase):
    def test_sort_stack_reorders_last_axis(self) -> None:
        stack = np.zeros((1, 1, 1, 3))
        stack[0, 0, 0, :] = [2, 1, 3]
        sorted_stack, exposures = sort_stack(stack, np.array([2, 1, 3]))
        np.testing.assert_array_equal(exposures, [1, 2, 3])
        np.testing.assert_array_equal(sorted_stack[0, 0, 0, :], [1, 2, 3])

    def test_stack_histogram_counts_pixels(self) -> None:
        stack = np.zeros((2, 3, 3, 2), dtype=np.uint8)
        histogram = compute_stack_histogram(stack)
        self.assertEqual(histogram.shape, (256, 3, 2))
        np.testing.assert_array_equal(histogram[0], np.full((3, 2), 6))

    def test_stack_disk_round_trip(self) -> None:
        stack = np.zeros((2, 3, 3, 2), dtype=np.float64)
        stack[..., 0] = 0.25
        stack[..., 1] = 0.75
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "frame"
            write_ldr_stack(stack, prefix, "png")
            restored, norm = read_ldr_stack(directory, "png", normalize=True)
        self.assertEqual(norm, 255.0)
        np.testing.assert_allclose(restored, stack, atol=1 / 255)


if __name__ == "__main__":
    unittest.main()
