from __future__ import annotations

import numpy as np

from .core import check_three_color, remove_specials


RGB_TO_XYZ = np.array(
    [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]
)
XYZ_TO_LMS = np.array(
    [[0.4002, 0.7075, -0.0807], [-0.2280, 1.1500, 0.0612], [0.0, 0.0, 0.9184]]
)
LMS_TO_IPT = np.array(
    [[0.4000, 0.4000, 0.2000], [4.4550, -4.8510, 0.3960], [0.8056, 0.3572, -1.1628]]
)
RGB_TO_YUV = np.array(
    [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.616, -0.51499, -0.10001]]
)


def convert_linear_space(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    check_three_color(image)
    matrix = np.asarray(matrix)
    if matrix.shape != (3, 3):
        raise ValueError("Color transformation matrix must be 3x3")
    return np.einsum("...j,ij->...i", image, matrix)


def convert_rgb_to_xyz(image: np.ndarray, inverse: bool = False) -> np.ndarray:
    return convert_linear_space(image, np.linalg.inv(RGB_TO_XYZ) if inverse else RGB_TO_XYZ)


def convert_rgb_to_yuv(image: np.ndarray, inverse: bool = False) -> np.ndarray:
    return convert_linear_space(image, np.linalg.inv(RGB_TO_YUV) if inverse else RGB_TO_YUV)


def convert_rgb_to_srgb(image: np.ndarray, inverse: bool = False) -> np.ndarray:
    image = np.asarray(image)
    if not inverse:
        return np.where(
            image <= 0.0031308,
            12.92 * image,
            1.055 * np.power(image, 1.0 / 2.4) - 0.055,
        )
    return np.where(
        image <= 0.04045,
        image / 12.92,
        np.power((image + 0.055) / 1.055, 2.4),
    )


def convert_xyz_to_lms(image: np.ndarray, inverse: bool = False) -> np.ndarray:
    return convert_linear_space(image, np.linalg.inv(XYZ_TO_LMS) if inverse else XYZ_TO_LMS)


def convert_lms_to_ipt(image: np.ndarray, inverse: bool = False) -> np.ndarray:
    if not inverse:
        nonlinear = np.sign(image) * np.power(np.abs(image), 0.43)
        return convert_linear_space(nonlinear, LMS_TO_IPT)
    linear = convert_linear_space(image, np.linalg.inv(LMS_TO_IPT))
    return np.sign(linear) * np.power(np.abs(linear), 1.0 / 0.43)


def convert_xyz_to_ipt(image: np.ndarray, inverse: bool = False) -> np.ndarray:
    if inverse:
        return convert_xyz_to_lms(convert_lms_to_ipt(image, True), True)
    return convert_lms_to_ipt(convert_xyz_to_lms(image), False)


def convert_ipt_to_ich(image: np.ndarray, inverse: bool = False) -> np.ndarray:
    check_three_color(image)
    output = np.asarray(image).copy()
    if not inverse:
        output[..., 1] = np.hypot(image[..., 1], image[..., 2])
        output[..., 2] = np.arctan2(image[..., 1], image[..., 2])
    else:
        output[..., 1] = np.sin(image[..., 2]) * image[..., 1]
        output[..., 2] = np.cos(image[..., 2]) * image[..., 1]
    return output


def convert_rgb_to_ich(image: np.ndarray, inverse: bool = False) -> np.ndarray:
    if inverse:
        return convert_rgb_to_xyz(convert_xyz_to_ipt(convert_ipt_to_ich(image, True), True), True)
    return convert_ipt_to_ich(convert_xyz_to_ipt(convert_rgb_to_xyz(image)))


def cielab_function(values: np.ndarray, inverse: bool = False) -> np.ndarray:
    values = np.asarray(values)
    if not inverse:
        threshold = (6.0 / 29.0) ** 3
        return np.where(
            values > threshold,
            np.cbrt(values),
            values * ((29.0 / 6.0) ** 2) / 3.0 + 4.0 / 29.0,
        )
    threshold = 6.0 / 29.0
    return np.where(
        values > threshold,
        values**3,
        (values - 4.0 / 29.0) * 3.0 * (6.0 / 29.0) ** 2,
    )


def convert_xyz_to_cielab(
    image: np.ndarray,
    inverse: bool = False,
    white_point: np.ndarray = np.ones(3),
) -> np.ndarray:
    check_three_color(image)
    wp = np.asarray(white_point)
    if not inverse:
        scaled = image / wp
        fy = cielab_function(scaled[..., 1])
        output = np.empty_like(scaled, dtype=np.float64)
        output[..., 0] = 116.0 * fy - 16.0
        output[..., 1] = 500.0 * (cielab_function(scaled[..., 0]) - fy)
        output[..., 2] = 200.0 * (fy - cielab_function(scaled[..., 2]))
        return remove_specials(output)
    common = (image[..., 0] + 16.0) / 116.0
    output = np.empty_like(image, dtype=np.float64)
    output[..., 1] = wp[1] * cielab_function(common, True)
    output[..., 0] = wp[0] * cielab_function(common + image[..., 1] / 500.0, True)
    output[..., 2] = wp[2] * cielab_function(common - image[..., 2] / 200.0, True)
    return remove_specials(output)


def convert_xyz_to_cielch(
    image: np.ndarray,
    inverse: bool = False,
    white_point: np.ndarray = np.ones(3),
) -> np.ndarray:
    if not inverse:
        lab = convert_xyz_to_cielab(image, False, white_point)
        output = lab.copy()
        output[..., 1] = np.hypot(lab[..., 1], lab[..., 2])
        output[..., 2] = np.mod(np.degrees(np.arctan2(lab[..., 2], lab[..., 1])), 360.0)
        return output
    lab = np.empty_like(image, dtype=np.float64)
    angle = np.radians(image[..., 2])
    lab[..., 0] = image[..., 0]
    lab[..., 1] = np.cos(angle) * image[..., 1]
    lab[..., 2] = np.sin(angle) * image[..., 1]
    return convert_xyz_to_cielab(lab, True, white_point)


def convert_xyz_to_yxy(image: np.ndarray, inverse: bool = False) -> np.ndarray:
    check_three_color(image)
    output = np.empty_like(image, dtype=np.float64)
    if not inverse:
        norm = np.sum(image, axis=2)
        output[..., 0] = image[..., 1]
        output[..., 1] = np.divide(image[..., 0], norm, out=np.zeros_like(norm), where=norm != 0)
        output[..., 2] = np.divide(image[..., 1], norm, out=np.zeros_like(norm), where=norm != 0)
    else:
        ratio = np.divide(image[..., 0], image[..., 2], out=np.zeros_like(image[..., 0]), where=image[..., 2] != 0)
        output[..., 0] = ratio * image[..., 1]
        output[..., 1] = image[..., 0]
        output[..., 2] = ratio * (1.0 - image[..., 1] - image[..., 2])
    return remove_specials(output)


def convert_xyz_to_luv(
    image: np.ndarray,
    inverse: bool = False,
    white_point: np.ndarray = np.ones(3),
) -> np.ndarray:
    check_three_color(image)
    wp = np.asarray(white_point)
    wp_norm = wp[0] + 15 * wp[1] + 3 * wp[2]
    un, vn = 4 * wp[0] / wp_norm, 9 * wp[1] / wp_norm
    output = np.zeros_like(image, dtype=np.float64)
    if not inverse:
        y = image[..., 1] / wp[1]
        lstar = np.where(y <= (6 / 29) ** 3, y * (29 / 3) ** 3, 116 * np.cbrt(y) - 16)
        norm = image[..., 0] + 15 * image[..., 1] + 3 * image[..., 2]
        up = np.divide(4 * image[..., 0], norm, out=np.zeros_like(norm), where=norm != 0)
        vp = np.divide(9 * image[..., 1], norm, out=np.zeros_like(norm), where=norm != 0)
        output[..., 0] = lstar
        output[..., 1] = 13 * lstar * (up - un)
        output[..., 2] = 13 * lstar * (vp - vn)
    else:
        lstar = image[..., 0]
        denom = 13 * lstar
        up = np.divide(image[..., 1], denom, out=np.zeros_like(lstar), where=denom != 0) + un
        vp = np.divide(image[..., 2], denom, out=np.zeros_like(lstar), where=denom != 0) + vn
        y = np.where(lstar <= 8, lstar * (3 / 29) ** 3, ((lstar + 16) / 116) ** 3)
        output[..., 1] = y * wp[1]
        output[..., 0] = output[..., 1] * np.divide(9 * up, 4 * vp, out=np.zeros_like(up), where=vp != 0)
        output[..., 2] = output[..., 1] * np.divide(12 - 3 * up - 20 * vp, 4 * vp, out=np.zeros_like(up), where=vp != 0)
    return remove_specials(output)


def convert_lms_to_lalpha_beta(image: np.ndarray, inverse: bool = False) -> np.ndarray:
    if inverse:
        matrix = np.diag([np.sqrt(3) / 3, np.sqrt(6) / 6, np.sqrt(2) / 2]) @ np.array(
            [[1, 1, 1], [1, 1, -2], [1, -1, 0]]
        )
        return 10 ** (convert_linear_space(image, matrix) - 0.001)
    matrix = np.array([[1, 1, 1], [1, 1, -1], [1, -2, 0]]) @ np.diag(
        [1 / np.sqrt(3), 1 / np.sqrt(6), 1 / np.sqrt(2)]
    )
    return convert_linear_space(np.log10(image + 0.001), matrix)


def convert_rgb2020_to_rgb709(image: np.ndarray, clip: bool = True) -> np.ndarray:
    matrix = np.array(
        [[1.6605, -0.5876, -0.0728], [-0.2146, 1.1329, -0.0083], [-0.0182, -0.1006, 1.1187]]
    )
    output = convert_linear_space(image, matrix)
    return np.clip(output, 0.0, 1.0) if clip else output


def convert_rgb709_to_rgb2020(image: np.ndarray, clip: bool = True) -> np.ndarray:
    matrix = np.array(
        [[1.6605, -0.5876, -0.0728], [-0.2146, 1.1329, -0.0083], [-0.0182, -0.1006, 1.1187]]
    )
    output = convert_linear_space(image, np.linalg.inv(matrix))
    return np.clip(output, 0.0, 1.0) if clip else output


def create_matrix_from_primaries(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    white_point: np.ndarray,
) -> np.ndarray:
    red, green, blue, white_point = map(
        lambda value: np.asarray(value, dtype=np.float64).reshape(3),
        (red, green, blue, white_point),
    )
    rows = []
    targets = []
    for primary, target in (
        (red, [1, 0, 0]),
        (green, [0, 1, 0]),
        (blue, [0, 0, 1]),
        (white_point, [1, 1, 1]),
    ):
        for channel in range(3):
            row = np.zeros(9)
            row[channel * 3 : (channel + 1) * 3] = primary
            rows.append(row)
            targets.append(target[channel])
    solution, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(targets), rcond=None)
    return solution.reshape(3, 3)


def create_matrix_from_primaries_xy(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    white_point: np.ndarray,
) -> np.ndarray:
    def xy_to_xyz(xy: np.ndarray) -> np.ndarray:
        x, y = np.asarray(xy, dtype=np.float64)
        return np.array([x / y, 1.0, (1.0 - x - y) / y])

    return create_matrix_from_primaries(
        xy_to_xyz(red),
        xy_to_xyz(green),
        xy_to_xyz(blue),
        xy_to_xyz(white_point),
    )


def luminance(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        return image
    if image.shape[2] == 3:
        return 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
    return np.mean(image, axis=2)


def luma(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        return image
    if image.shape[2] == 3:
        return 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    return np.mean(image, axis=2)


def scotopic_luminance(xyz: np.ndarray) -> np.ndarray:
    check_three_color(xyz)
    ratio = (xyz[..., 1] + xyz[..., 2]) / (xyz[..., 0] + 1e-6)
    return xyz[..., 1] * (1.33 * (1.0 + ratio) - 1.68)


def helmholtz_kohlrausch_luminance(image: np.ndarray) -> np.ndarray:
    lch = convert_xyz_to_cielch(convert_rgb_to_xyz(image))
    lightness, chroma, hue = lch[..., 0], lch[..., 1], lch[..., 2]
    angle = ((hue - 90.0) / 2.0) * np.pi / 360.0
    result = lightness + (2.5 - 0.025 * lightness) * (
        0.116 * np.abs(np.sin(angle)) + 0.085
    ) * chroma
    maximum = np.max(result)
    return result / maximum if maximum > 0 else result


def ipt_colorfulness(ipt: np.ndarray) -> np.ndarray:
    check_three_color(ipt)
    return np.hypot(ipt[..., 1], ipt[..., 2])


def saturation_pouli(chroma: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    return remove_specials(chroma / np.hypot(chroma, intensity))


ConvertLinearSpace = convert_linear_space
ConvertRGBtoXYZ = convert_rgb_to_xyz
ConvertRGBtoYUV = convert_rgb_to_yuv
ConvertRGBtosRGB = convert_rgb_to_srgb
ConvertXYZtoLMS = convert_xyz_to_lms
ConvertLMStoIPT = convert_lms_to_ipt
ConvertXYZtoIPT = convert_xyz_to_ipt
ConvertIPTtoICh = convert_ipt_to_ich
ConvertRGBtoICh = convert_rgb_to_ich
CIELabFunction = cielab_function
ConvertXYZtoCIELab = convert_xyz_to_cielab
ConvertXYZtoCIELCh = convert_xyz_to_cielch
ConvertXYZtoYxy = convert_xyz_to_yxy
ConvertXYZtoLUV = convert_xyz_to_luv
ConvertLMStoLAlphaBeta = convert_lms_to_lalpha_beta
ConvertRGB2020 = convert_rgb2020_to_rgb709
CreateMatrixFromPrimaries = create_matrix_from_primaries
CreateMatrixFromPrimaries_xy = create_matrix_from_primaries_xy
lum = luminance
lumHK = helmholtz_kohlrausch_luminance
lumScotopic = scotopic_luminance
IPTColorfullness = ipt_colorfulness
SaturationPouli = saturation_pouli
