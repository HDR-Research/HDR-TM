from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve

from .colorspace import convert_linear_space, luminance
from .core import remove_specials
from .filters import bilateral_filter


def bleaching_parameters(a_cone: np.ndarray, a_rod: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return 2e6 / (2e6 + np.asarray(a_cone)), 0.04 / (0.04 + np.asarray(a_rod))


def saturation_parameters(a_cone: np.ndarray, a_rod: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cone, rod = np.asarray(a_cone), np.asarray(a_rod)
    j = 1.0 / (5e5 * rod + 1.0)
    k = 1.0 / (5.0 * cone + 1.0)
    sigma_rod = 2.5874 * rod / (19000 * j**2 * rod + 0.2615 * (1 - j**2) ** 4 * np.cbrt(np.sqrt(rod)))
    sigma_cone = 12.9223 * cone / (k**4 * cone + (1 - k**4) ** 2 * np.cbrt(cone))
    return remove_specials(sigma_cone), remove_specials(sigma_rod)


def sigmoid_response(image: np.ndarray, n: float, sigma: np.ndarray, b: np.ndarray) -> np.ndarray:
    power = np.power(np.maximum(image, 0), n)
    return remove_specials(power / (power + np.power(sigma, n)) * b)


def sigmoid_color_response(luminance_value: np.ndarray, n: float, sigma: np.ndarray, b: np.ndarray) -> np.ndarray:
    power = np.power(np.maximum(luminance_value, 0), n)
    sigma_power = np.power(sigma, n)
    return remove_specials(power * (n * b * sigma_power) / (power + sigma_power) ** 2)


def ciecam02_degree_adaptation(l_a: np.ndarray, f: float) -> np.ndarray:
    return f * (1.0 - np.exp(-(np.asarray(l_a) + 42.0) / 92.0) / 3.6)


def ciecam02_f_l(l_a: np.ndarray) -> np.ndarray:
    l_a = np.asarray(l_a)
    k = 1.0 / (5.0 * l_a + 1.0)
    return 0.2 * k**4 * (5 * l_a) + 0.1 * (1 - k**4) ** 2 * np.cbrt(5 * l_a)


def ciecam02_chromatic_adaptation(image_xyz: np.ndarray, white_xyz: np.ndarray) -> np.ndarray:
    cat02 = np.array([[0.7328, 0.4296, -0.1624], [-0.7036, 1.6975, 0.0061], [0.0030, 0.0136, 0.9834]])
    d65 = np.array([96.047, 100.0, 108.883])
    white = np.asarray(white_xyz, dtype=np.float64).reshape(3)
    source_rgb = cat02 @ white
    target_rgb = cat02 @ d65
    d = ciecam02_degree_adaptation(0.2 * white[1], 1.0)
    scale = d * target_rgb / np.maximum(source_rgb, 1e-12) + 1.0 - d
    adapted = convert_linear_space(np.asarray(image_xyz), cat02) * scale
    return convert_linear_space(adapted, np.linalg.inv(cat02))


def tvi_ashikhmin(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    output = np.empty_like(value)
    first = value < 0.0034
    second = (value >= 0.0034) & (value < 1.0)
    third = (value >= 1.0) & (value < 7.2444)
    output[first] = value[first] / 0.0014
    output[second] = 2.4483 + np.log(value[second] / 0.0034) / 0.4027
    output[third] = 16.563 + (value[third] - 1.0) / 0.4027
    output[~(first | second | third)] = 32.0693 + np.log(value[~(first | second | third)] / 7.2444) / 0.0556
    return output


def kuang_gamma(average_surround_param: str) -> float:
    values = {"dark": 1.5, "dim": 1.25, "average": 1.0}
    try:
        return values[str(average_surround_param).lower()]
    except KeyError as error:
        raise ValueError("Surround must be 'dark', 'dim', or 'average'") from error


def kuang_normalized_gamma(image: np.ndarray, gamma_value: float) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    maximum = float(np.max(image))
    return np.power(np.maximum(image / maximum, 0), gamma_value) * maximum if maximum > 0 else image.copy()


def fattal_phi(grad_x: np.ndarray, grad_y: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    magnitude = np.hypot(grad_x, grad_y)
    return remove_specials(np.power(magnitude, beta - 1.0) * alpha ** (1.0 - beta))


def chiu_glare(image: np.ndarray, glare_opt: tuple[float, float, int] = (0.8, 8.0, 121)) -> np.ndarray:
    center_weight, falloff, size = glare_opt
    size = int(size) | 1
    axis = np.arange(size) - size // 2
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    radius = np.hypot(xx, yy)
    kernel = np.exp(-radius / max(float(falloff), 1e-12))
    kernel[size // 2, size // 2] = 0
    kernel *= (1.0 - center_weight) / max(np.sum(kernel), 1e-12)
    kernel[size // 2, size // 2] = center_weight
    source = np.asarray(image, dtype=np.float64)
    if source.ndim == 2:
        return ndimage.convolve(source, kernel, mode="nearest")
    return np.stack([ndimage.convolve(source[..., c], kernel, mode="nearest") for c in range(source.shape[2])], axis=2)


def ashikhmin_filtering(luminance_image: np.ndarray, s_max: int = 64) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(luminance_image, dtype=np.float64)
    adaptation = ndimage.gaussian_filter(source, 0.2, mode="nearest")
    selected = np.zeros(source.shape, dtype=bool)
    scale = 1
    while scale <= max(int(s_max), 1):
        small = ndimage.gaussian_filter(source, max(scale / 5.0, 0.2), mode="nearest")
        large = ndimage.gaussian_filter(source, max(2 * scale / 5.0, 0.4), mode="nearest")
        contrast = np.abs(small - large) / np.maximum(small, 1e-12)
        choose = (~selected) & (contrast < 0.5)
        adaptation[choose] = small[choose]
        selected |= choose
        scale *= 2
    adaptation[~selected] = large[~selected]
    return adaptation, remove_specials(source / np.maximum(adaptation, 1e-12))


def reinhard_gaussian_filter(image: np.ndarray, scale: float, alpha_i: float) -> np.ndarray:
    return ndimage.gaussian_filter(np.asarray(image, dtype=np.float64), max(alpha_i * scale / np.sqrt(2.0), 1e-12), mode="nearest")


def reinhard_filtering(luminance_image: np.ndarray, p_alpha: float = 0.18, p_phi: float = 8.0, p_epsilon: float = 0.05) -> np.ndarray:
    source = np.asarray(luminance_image, dtype=np.float64)
    result = source.copy()
    unresolved = np.ones(source.shape, dtype=bool)
    for scale in np.power(1.6, np.arange(9)):
        v1 = reinhard_gaussian_filter(source, scale, 0.35)
        v2 = reinhard_gaussian_filter(source, scale, 0.56)
        denominator = (2**p_phi) * p_alpha / (scale**2) + v1
        contrast = np.abs(v1 - v2) / np.maximum(denominator, 1e-12)
        choose = unresolved & (contrast > p_epsilon)
        result[choose] = v1[choose]
        unresolved[choose] = False
    result[unresolved] = v1[unresolved]
    return result


def reinhard_bilateral_filtering(luminance_image: np.ndarray, p_alpha: float = 0.18, p_phi: float = 8.0, p_epsilon: float = 0.05) -> np.ndarray:
    source = np.asarray(luminance_image, dtype=np.float64)
    sigma_range = max(float(np.std(source)) * max(p_epsilon, 1e-3), 1e-6)
    spatial = max(min(source.shape) * p_alpha / max(p_phi, 1e-6), 1.0)
    return bilateral_filter(source, sigma_spatial=spatial, sigma_range=sigma_range)


def stevenson_detail_enhancement(detail_layer: np.ndarray, f_l: np.ndarray) -> np.ndarray:
    return np.power(np.maximum(detail_layer, 0), np.power(np.asarray(f_l) + 0.8, 0.25))


def ward_downsampling(image: np.ndarray) -> np.ndarray:
    source = np.asarray(image)
    height, width = source.shape[:2]
    scale = min(1.0, 64.0 / max(min(height, width), 1))
    return cv2.resize(source, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)


def yee_pattanaik_luminance_adaptation(image: np.ndarray, max_layers: int = 32) -> np.ndarray:
    log_l = np.log10(np.maximum(luminance(image), 1e-12))
    layers = max(1, int(max_layers))
    result = np.zeros_like(log_l)
    for width in np.linspace(0.5, 2.0, layers):
        categories = np.floor((log_l - np.min(log_l)) / width).astype(int)
        labels = np.zeros_like(categories, dtype=np.int32)
        offset = 0
        for category in np.unique(categories):
            component, count = ndimage.label(categories == category, np.ones((3, 3)))
            active = component > 0
            labels[active] = component[active] + offset
            offset += count
        for label in range(1, offset + 1):
            mask = labels == label
            if np.any(mask):
                result[mask] += np.mean(log_l[mask])
    return np.power(10.0, result / layers)


def generate_masks(image_bins: np.ndarray, n_levels: int) -> np.ndarray:
    bins = np.asarray(image_bins)
    output = np.zeros((*bins.shape, int(n_levels)), dtype=np.int32)
    for index in range(int(n_levels)):
        output[..., index], _ = ndimage.label(bins == index + 1, np.ones((3, 3)))
    return output


def compute_fusion_mask(neighbors: np.ndarray, component: np.ndarray, image_bins: np.ndarray, label: int) -> tuple[np.ndarray, int]:
    output = np.asarray(image_bins).copy()
    values = np.asarray(neighbors).reshape(-1)
    if values.size:
        output[np.asarray(component) == label] = values[0]
    return output, int(values.size)


def create_segments(image: np.ndarray) -> np.ndarray:
    lum = ndimage.gaussian_filter(luminance(image).astype(np.float64), 0.2, mode="nearest")
    positive = lum[lum > 0]
    if positive.size == 0:
        return np.zeros(lum.shape, dtype=int)
    low = int(np.floor(np.log10(np.min(positive))))
    high = int(np.ceil(np.log10(np.max(positive))))
    original = np.floor(np.log10(np.maximum(lum, 10.0**low))).astype(int) - low + 1
    bins = original.copy()
    threshold = round(0.005 * bins.size)
    structure = np.ones((3, 3))
    for _ in range(100):
        previous = bins.copy()
        for level in range(1, high - low + 2):
            components, count = ndimage.label(bins == level, structure)
            for label in range(1, count + 1):
                mask = components == label
                if 0 < np.count_nonzero(mask) < threshold:
                    border = ndimage.binary_dilation(mask, structure) & ~mask
                    neighbors = bins[border]
                    neighbors = neighbors[neighbors != level]
                    if neighbors.size:
                        bins[mask] = np.min(neighbors)
        if np.array_equal(previous, bins):
            break
    for level in np.unique(bins):
        mask = bins == level
        bins[mask] = round(float(np.mean(original[mask])))
    return bins + low - 1


def create_segments_approx(image: np.ndarray) -> np.ndarray:
    lum = ndimage.gaussian_filter(luminance(image).astype(np.float64), 0.2, mode="nearest")
    epsilon = 0.015 / 2.0
    log_l = np.log10(lum + epsilon)
    minimum = max(float(np.min(log_l)), float(np.min(np.round(log_l))))
    bins = np.round(log_l) - minimum + 1
    guide = np.log10(np.maximum(lum, 1e-12)) - minimum + 1
    iterations = int(np.ceil(np.sqrt(0.005 * lum.size)) / 4)
    for _ in range(iterations):
        bins = bilateral_filter(bins, guide, 0, float(np.max(guide)), 4, 0.25)
    return np.round(bins) + minimum - 1


def krawczyk_kmeans(bound: np.ndarray, histogram: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = map(float, np.asarray(bound).reshape(2))
    centers = np.arange(lower, upper + 1.0, 1.0)
    if centers[-1] < upper:
        centers = np.append(centers, upper)
    hist = np.asarray(histogram, dtype=np.float64).reshape(-1)
    values = np.linspace(lower, upper, hist.size)
    totals = np.zeros_like(centers)
    for _ in range(100):
        assignment = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        new_centers, new_totals = [], []
        for index in range(centers.size):
            selected = assignment == index
            total = np.sum(hist[selected])
            if total > 0:
                new_centers.append(np.sum(values[selected] * hist[selected]) / total)
                new_totals.append(total)
        new_centers = np.asarray(new_centers)
        totals = np.asarray(new_totals)
        if centers.shape == new_centers.shape and np.allclose(centers, new_centers):
            centers = new_centers
            break
        centers = new_centers
    index = 0
    while index < centers.size - 1:
        if abs(centers[index] - centers[index + 1]) < 1:
            total = totals[index] + totals[index + 1]
            centers[index] = (centers[index] * totals[index] + centers[index + 1] * totals[index + 1]) / total
            totals[index] = total
            centers, totals = np.delete(centers, index + 1), np.delete(totals, index + 1)
        else:
            index += 1
    return centers, totals


def krawczyk_max_distance(centers: np.ndarray, bound: np.ndarray) -> float:
    centers = np.asarray(centers)
    return float(np.max(np.abs(np.diff(centers)))) if centers.size > 1 else float(np.diff(np.asarray(bound).reshape(2))[0])


def krawczyk_p_norm(centers: np.ndarray, log_l: np.ndarray, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-12)
    return np.sum(np.exp(-((np.asarray(centers)[:, None, None] - np.asarray(log_l)[None]) ** 2) / (2 * sigma**2)), axis=0)


def krawczyk_image_partition(centers: np.ndarray, log_l: np.ndarray, bound: np.ndarray, total_pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers, totals = np.asarray(centers, dtype=float).copy(), np.asarray(total_pixels, dtype=float).copy()
    log_l = np.asarray(log_l)
    while True:
        distances = np.abs(log_l[..., None] - centers)
        framework = np.argmin(distances, axis=-1) + 1
        distance = np.min(distances, axis=-1)
        sigma = max(krawczyk_max_distance(centers, bound), 1e-12)
        norm = krawczyk_p_norm(centers, log_l, sigma)
        merged = False
        for index in range(centers.size - 1):
            probability = remove_specials(np.exp(-((centers[index] - log_l) ** 2) / (2 * sigma**2)) / norm)
            if not np.any(probability[framework == index + 1] > 0.6):
                total = totals[index] + totals[index + 1]
                centers[index] = (centers[index] * totals[index] + centers[index + 1] * totals[index + 1]) / total
                totals[index] = total
                centers, totals = np.delete(centers, index + 1), np.delete(totals, index + 1)
                merged = True
                break
        if not merged:
            return framework, distance, centers


def poisson_solver(function: np.ndarray, smoothing_cost: float = 0.0) -> np.ndarray:
    source = np.asarray(function, dtype=np.float64)
    rows, columns = source.shape
    row_laplacian = sparse.diags([-np.ones(rows - 1), 2 * np.ones(rows), -np.ones(rows - 1)], [-1, 0, 1])
    column_laplacian = sparse.diags([-np.ones(columns - 1), 2 * np.ones(columns), -np.ones(columns - 1)], [-1, 0, 1])
    matrix = sparse.kronsum(column_laplacian, row_laplacian, format="csr")
    if smoothing_cost > 0:
        matrix = matrix + smoothing_cost * sparse.eye(source.size)
    return spsolve(matrix, -source.ravel()).reshape(source.shape)


def lischinski_minimization(
    log_luminance: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None = None,
    alpha: float = 1.0,
    lambda_value: float = 0.4,
) -> tuple[np.ndarray, sparse.csr_matrix]:
    source = np.asarray(log_luminance, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    weights = np.ones_like(source) if weights is None else np.asarray(weights, dtype=np.float64)
    rows, columns = source.shape
    horizontal = lambda_value / (np.abs(np.diff(source, axis=1)) ** alpha + 1e-4)
    vertical = lambda_value / (np.abs(np.diff(source, axis=0)) ** alpha + 1e-4)
    index = np.arange(source.size).reshape(source.shape)
    rr = [index.ravel()]
    cc = [index.ravel()]
    data = [weights.ravel().copy()]
    for first, second, conductance in (
        (index[:, :-1], index[:, 1:], horizontal),
        (index[:-1, :], index[1:, :], vertical),
    ):
        a, b, w = first.ravel(), second.ravel(), conductance.ravel()
        rr.extend([a, b, a, b])
        cc.extend([a, b, b, a])
        data.extend([w, w, -w, -w])
    matrix = sparse.coo_matrix((np.concatenate(data), (np.concatenate(rr), np.concatenate(cc))), shape=(source.size, source.size)).tocsr()
    result = spsolve(matrix, (target * weights).ravel()).reshape(source.shape)
    return result, matrix


def histogram_ceiling(histogram: np.ndarray, k: float) -> np.ndarray:
    output = np.asarray(histogram, dtype=np.float64).copy()
    tolerance = np.sum(output) * 0.025
    inactive = 0
    while True:
        total = np.sum(output)
        if total < tolerance:
            break
        ceiling = total * k
        trimmed = output > ceiling
        trimmings = float(np.sum(output[trimmed] - ceiling))
        output[trimmed] = ceiling
        inactive = inactive + 1 if not np.any(trimmed) else 0
        if trimmings <= tolerance or inactive >= 2:
            break
    return output


BleachingParameters = bleaching_parameters
SaturationParameters = saturation_parameters
SigmoidResponse = sigmoid_response
SigmoidColorResponse = sigmoid_color_response
CIECAM02_DegreeAdaptation = ciecam02_degree_adaptation
CIECAM02_F_L = ciecam02_f_l
CIECAM02_ChromaticAdaptation = ciecam02_chromatic_adaptation
TVI_Ashikhmin = tvi_ashikhmin
KuangGamma = kuang_gamma
KuangNormalizedGamma = kuang_normalized_gamma
FattalPhi = fattal_phi
ChiuGlare = chiu_glare
AshikhminFiltering = ashikhmin_filtering
ReinhardGaussianFilter = reinhard_gaussian_filter
ReinhardFiltering = reinhard_filtering
ReinhardBilateralFiltering = reinhard_bilateral_filtering
StevensonDetailEnhancement = stevenson_detail_enhancement
WardDownsampling = ward_downsampling
YeePattanaikLuminanceAdaptation = yee_pattanaik_luminance_adaptation
GenerateMasks = generate_masks
computeFusionMask = compute_fusion_mask
CreateSegments = create_segments
CreateSegmentsApprox = create_segments_approx
KrawczykKMeans = krawczyk_kmeans
KrawczykMaxDistance = krawczyk_max_distance
KrawczykPNorm = krawczyk_p_norm
KrawczykImagePartition = krawczyk_image_partition
PoissonSolver = poisson_solver
LischinskiMinimization = lischinski_minimization
