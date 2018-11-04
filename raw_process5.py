""" Simple RAW image processing module"""

import sys
import os
import scipy
from scipy import signal
import numpy as np
from numpy.lib.stride_tricks import as_strided
import rawpy

""" Process RAW file into a image file.

Example usage:
raw = read("sample.ARW")
rgb = process(raw)
write(rgb, "output.ARW")
"""

def read(filename):
    """
    Read RAW data from specified file. Currently supported formats are
        ARW (Sony RAW format)
        JPEG with Raspberry Pi V2.1 camera RAW

    :param filename: path to the target RAW file
    """

    return rawpy.imread(filename)

def check_functions(filename):
    """ Check what functions to be enabled based on filename"""
    white_level = 1024
    shading_enable = True
    defect_correction_enable = True
    noise_parameters = (8, 2, 246)
    wbg_norm = 1
    extension = os.path.splitext(filename)[1]
    if extension in (".ARW", ".arw"):
        shading_enable = False
        defect_correction_enable = False
        white_level = 8192
        noise_parameters = (8, 0.2, 25)
        wbg_norm = 1024
    return (shading_enable, defect_correction_enable, white_level, wbg_norm, noise_parameters)

DEFALT_MATRIX = (1024, 0, 0, 0, 1024, 0, 0, 0, 1024)
DEFALT_TONE = ((0, 64, 128, 192, 256), (0, 64, 128, 192, 256))

def process(filename, output_filename, color_matrix=DEFALT_MATRIX, tone_curve=DEFALT_TONE):
    """
    This processes RAW data that was read by read() method.
    Must be called after read() operation. No error is checked.
    """
    shading_enable, defect_crrection_enable, white_level, wbg_norm, noise_param = check_functions(filename)
    raw = read(filename)
    raw_array = get_raw_array(raw)

    raw_array = black_level_correction(raw_array, raw.black_level_per_channel, raw.raw_pattern)
    if defect_crrection_enable:
        raw_array = defect_crrection(raw_array)
    if shading_enable:
        raw_array = lens_shading_correction(raw_array, LSC_DEFAULT)
    raw_array = white_balance_Bayer(raw_array, raw.camera_whitebalance, wbg_norm, raw.raw_pattern)
    rgb_array = advanced_demosaic(raw_array, raw.raw_pattern)
    del raw_array, raw
    rgb_array = noise_filter(rgb_array, noise_param[0], noise_param[1], noise_param[2])
    rgb_array = color_correction_matrix(rgb_array, color_matrix)
    rgb_array = gamma_correction(rgb_array/white_level, 2.2)
    rgb_array = edge_correction(rgb_array, 2, 0.25, 1, 0.25)
    rgb_array = tone_curve_correction(rgb_array, tone_curve[0], tone_curve[1])
    write(rgb_array, output_filename)

def get_raw_array(raw):
    """ convert raw_img into numpy array"""
    h, w = raw.sizes.raw_height, raw.sizes.raw_width
    raw_array = np.array(raw.raw_image).reshape((h, w)).astype('float')
    return raw_array

def black_level_correction(raw_array, black_level_per_channel, bayer_pattern):
    # rearrange black level
    black_level = [0] * 4
    black_level[bayer_pattern[0, 0]] = black_level_per_channel[bayer_pattern[0, 0]]
    black_level[bayer_pattern[0, 1]] = black_level_per_channel[bayer_pattern[0, 1]]
    black_level[bayer_pattern[1, 0]] = black_level_per_channel[bayer_pattern[1, 0]]
    black_level[bayer_pattern[1, 1]] = black_level_per_channel[bayer_pattern[1, 1]]

    blc_raw = raw_array.copy()
    blc_raw[0::2, 0::2] -= black_level[0]
    blc_raw[0::2, 1::2] -= black_level[1]
    blc_raw[1::2, 0::2] -= black_level[2]
    blc_raw[1::2, 1::2] -= black_level[3]
    return blc_raw

def defect_crrection(raw_array):
    dpc_raw = raw_array.copy()
    footprint = np.ones((5, 5))
    footprint[2, 2] = 0
    for (yo, xo) in ((0, 0), (1, 0), (0, 1), (1, 1)):
        single_channel = dpc_raw[yo::2, xo::2]
        flt = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4
        average = scipy.signal.convolve2d(single_channel, flt, mode='same')
        local_max = scipy.ndimage.filters.maximum_filter(single_channel, footprint=footprint, mode='mirror')
        local_min = scipy.ndimage.filters.minimum_filter(single_channel, footprint=footprint, mode='mirror')
        threshold = 16
        mask = (single_channel < local_min - threshold) + (single_channel > local_max + threshold)
        single_channel[mask] = average[mask]
    return dpc_raw


LSC_DEFAULT = [np.array([6.07106808e-07, 9.60556906e-01]),
               np.array([6.32044369e-07, 9.70694361e-01]),
               np.array([6.28455183e-07, 9.72493898e-01]),
               np.array([9.58743579e-07, 9.29427169e-01])]

def lens_shading_correction(raw, coef):
    """
    Apply lens shading correction to Bayer input (raw)
    Parameter (coef) needs to be array type of coef[4][2]
    coef[color][0] is coefficient on 2nd order term.
    coef[color][1] is offset
    """

    h, w = raw.shape
    gain_map = np.zeros((h, w))
    center_y, center_x = h // 2, w // 2
    x = np.arange(0, w) - center_x
    y = np.arange(0, h) - center_y
    xs, ys = np.meshgrid(x, y, sparse=True)
    r2 = ys * ys + xs * xs
    gain_map[::2, ::2] = r2[::2, ::2] * coef[0][0] + coef[0][1]
    gain_map[1::2, ::2] = r2[1::2, ::2] * coef[1][0] + coef[1][1]
    gain_map[::2, 1::2] = r2[::2, 1::2] * coef[2][0] + coef[2][1]
    gain_map[1::2, 1::2] = r2[1::2, 1::2] * coef[3][0] + coef[3][1]
    return raw * gain_map

def preview_demosaic(raw_array, bayer_pattern):
    """ Very simple demosaic with down sampling for preview purpose"""
    h, w = raw_array.shape[0], raw_array.shape[1]
    shuffle = np.zeros((h // 2, w // 2, 4))
    shuffle[:, :, bayer_pattern[0, 0]] += raw_array[0::2, 0::2]
    shuffle[:, :, bayer_pattern[0, 1]] += raw_array[0::2, 1::2]
    shuffle[:, :, bayer_pattern[1, 0]] += raw_array[1::2, 0::2]
    shuffle[:, :, bayer_pattern[1, 1]] += raw_array[1::2, 1::2]
    dms_img = np.zeros((h // 2, w // 2, 3))
    dms_img[:, :, 0] = shuffle[:, :, 0]
    dms_img[:, :, 1] = (shuffle[:, :, 1] + shuffle[:, :, 3]) / 2
    dms_img[:, :, 2] = shuffle[:, :, 2]
    return dms_img

def simple_demosaic(raw, raw_array):
    """ Simple demosaic algorithm with linear interpolation """
    h, w = raw_array.shape
    dms_img2 = np.zeros((h, w, 3))

    green = raw_array.copy()
    green[(raw.raw_colors == 0) | (raw.raw_colors == 2)] = 0
    g_flt = np.array([[0, 1 / 4, 0], [1 / 4, 1, 1 / 4], [0, 1 / 4, 0]])
    dms_img2[:, :, 1] = signal.convolve2d(green, g_flt, boundary='symm', mode='same')

    red = raw_array.copy()
    red[raw.raw_colors != 0] = 0
    rb_flt = np.array([[1 / 4, 1 / 2, 1 / 4], [1 / 2, 1, 1 / 2], [1 / 4, 1 / 2, 1 / 4]])
    dms_img2[:, :, 0] = signal.convolve2d(red, rb_flt, boundary='symm', mode='same')

    blue = raw_array.copy()
    blue[raw.raw_colors != 2] = 0
    rb_flt = np.array([[1 / 4, 1 / 2, 1 / 4], [1 / 2, 1, 1 / 2], [1 / 4, 1 / 2, 1 / 4]])
    dms_img2[:, :, 2] = signal.convolve2d(blue, rb_flt, boundary='symm', mode='same')
    return dms_img2

def advanced_demosaic(dms_input, bayer_pattern):
    """ Demosaic algorithm in frequency domain """
    hlpf = np.array([[1, 2, 3, 4, 3, 2, 1]]) / 16
    vlpf = np.transpose(hlpf)
    hhpf = np.array([[-1, 2, -3, 4, -3, 2, -1]]) / 16
    vhpf = np.transpose(hhpf)
    identity_filter = np.zeros((7, 7))
    identity_filter[3, 3] = 1

    # generate FIR filters to extract necessary components
    FC1 = np.matmul(vhpf, hhpf)
    FC2H = np.matmul(vlpf, hhpf)
    FC2V = np.matmul(vhpf, hlpf)
    FL = identity_filter - FC1 - FC2V - FC2H

    # f_C1 at 4 corners
    c1_mod = signal.convolve2d(dms_input, FC1, boundary='symm', mode='same')
    # f_C1^1 at wy = 0, wx = +Pi/-Pi
    c2h_mod = signal.convolve2d(dms_input, FC2H, boundary='symm', mode='same')
    # f_C1^1 at wy = +Pi/-Pi, wx = 0
    c2v_mod = signal.convolve2d(dms_input, FC2V, boundary='symm', mode='same')
    # f_L at center
    f_L = signal.convolve2d(dms_input, FL, boundary='symm', mode='same')

    # Move c1 to the center by shifting by Pi in both x and y direction
    # f_c1 = c1 * (-1)^x * (-1)^y
    f_c1 = c1_mod.copy()
    f_c1[:, 1::2] *= -1
    f_c1[1::2, :] *= -1
    if bayer_pattern[0, 0] == 1 or bayer_pattern[0, 0] == 3:
        f_c1 *= -1
    # Move c2a to the center by shifting by Pi in x direction, same for c2b in y direction
    c2h = c2h_mod.copy()
    c2h[:, 1::2] *= -1
    if bayer_pattern[0, 0] == 2 or bayer_pattern[1, 0] == 2:
        c2h *= -1
    c2v = c2v_mod.copy()
    c2v[1::2, :] *= -1
    if bayer_pattern[0, 0] == 2 or bayer_pattern[0, 1] == 2:
        c2v *= -1
    # f_c2 = (c2v_mod * x_mod + c2h_mod * y_mod) / 2
    f_c2 = (c2v + c2h) / 2

    # generate RGB channel using 
    # [R, G, B] = [[1, 1, 2], [1, -1, 0], [1, 1, - 2]] x [L, C1, C2]
    height, width = dms_input.shape
    dms_img = np.zeros((height, width, 3))
    dms_img[:, :, 0] = f_L + f_c1 + 2 * f_c2
    dms_img[:, :, 1] = f_L - f_c1
    dms_img[:, :, 2] = f_L + f_c1 - 2 * f_c2

    return dms_img

def white_balance_Bayer(raw_array, wbg, wbg_norm, bayer_pattern):
    """ Apply white balance to bayer input"""
    img_wb = raw_array.copy()
    img_wb[0::2, 0::2] *= wbg[bayer_pattern[0, 0]] / wbg_norm
    img_wb[0::2, 1::2] *= wbg[bayer_pattern[0, 1]] / wbg_norm
    img_wb[1::2, 0::2] *= wbg[bayer_pattern[1, 0]] / wbg_norm
    img_wb[1::2, 1::2] *= wbg[bayer_pattern[1, 1]] / wbg_norm
    return img_wb


def noise_filter(rgb_array, coef=8, read_noise=2, shot_noise=246):
    """ Apply bilateral noise filter to RGB image"""
    h, w, _ = rgb_array.shape
    luma_img = rgb_array[:, :, 0] + rgb_array[:, :, 1] + rgb_array[:, :, 2]
    average = scipy.ndimage.filters.uniform_filter(luma_img, 5, mode='mirror')
    sigma_map = average * shot_noise + read_noise
    del average
    sigma_map[sigma_map < 1] = 1
    sy, sx = sigma_map.strides
    sigma_tile = as_strided(sigma_map, strides=(sy, sx, 0, 0), shape=(h, w, 5, 5))
    sigma_tile = sigma_tile[2:h-2, 2:w-2, :, :]
    del sigma_map

    sy, sx = luma_img.strides
    luma_tile = as_strided(luma_img, strides=(sy, sx, 0, 0), shape=(h, w, 5, 5))
    luma_tile = luma_tile[2:h-2, 2:w-2, :, :]
    luma_box = as_strided(luma_img, strides=(sy, sx, sy, sx), shape=(h-4, w-4, 5, 5))    
    del luma_img
    
    diff = luma_box - luma_tile
    del luma_tile, luma_box
    diff = diff * diff
    weight = np.exp(-coef * diff / sigma_tile)
    del diff, sigma_tile
    weight_sum = weight.sum(axis=(2, 3))

    sy, sx, sz, sw = weight.strides
    weight_extend = as_strided(weight, strides=(sy, sx, 0, sz, sw), shape=(h-4, w-4, 3, 5, 5))
    del weight
    sy, sx = weight_sum.strides
    weight_sum_extend = as_strided(weight_sum, strides=(sy, sx, 0), shape=(h-4, w-4, 3))
    del weight_sum
    
    sy, sx, sz = rgb_array.strides
    img_boxes = as_strided(rgb_array, strides=(sy, sx, sz, sy, sx), shape=(h-4, w-4, 3, 5, 5))
    img_flt = (weight_extend * img_boxes).sum(axis=(3, 4)) / weight_sum_extend
    return img_flt

def color_correction_matrix(rgb_array, color_matrix):
    """ Apply color correction matrix to RGB array"""
    img_ccm = np.zeros_like(rgb_array)
    ccm = np.array(color_matrix).reshape((3, 3))
    norm = ccm.sum(axis=1).mean()
    for c in (0, 1, 2):
        img_ccm[:, :, c] = ccm[c, 0] * rgb_array[:, :, 0] + \
                           ccm[c, 1] * rgb_array[:, :, 1] + \
                           ccm[c, 2] * rgb_array[:, :, 2]
    return img_ccm / norm


def gamma_correction(rgb_array, gamma_coef):
    """ Apply gamma correction to RGB image"""
    img_gamma = rgb_array.copy()
    img_gamma[img_gamma < 0] = 0
    img_gamma = np.power(img_gamma, 1/gamma_coef)
    return img_gamma

def apply_matrix(input_array, matrix):
    img_out = np.zeros_like(input_array)
    for c in (0, 1, 2):
        img_out[:, :, c] = matrix[c, 0] * input_array[:, :, 0] + \
                           matrix[c, 1] * input_array[:, :, 1] + \
                           matrix[c, 2] * input_array[:, :, 2]
    return img_out

RGB_TO_YCBCR = np.array([[0.299, 0.587, 0.144],
                         [-0.168736, -0.331264, 0.5],
                         [0.5, -0.418688, -0.081312]])

def edge_correction(rgb_array, sigma1=2, coef1=0.25, sigma2=1, coef2=0.25):
    """ Edge correction for RGB input"""
    img_rgb = rgb_array.copy() * 256
    img_rgb[img_rgb < 0] = 0
    img_rgb[img_rgb > 255] = 255
    
    img_ycbcr = apply_matrix(img_rgb, RGB_TO_YCBCR)
    luma = img_ycbcr[:, :, 0]
    unsharpen1 = scipy.ndimage.gaussian_filter(luma, sigma=sigma1)
    unsharpen2 = scipy.ndimage.gaussian_filter(luma, sigma=sigma2)
    sharpen = luma + coef1 * (luma - unsharpen1) + coef2 * (luma - unsharpen2)
    img_ycbcr[:, :, 0] = sharpen

    ycbcr2rgb = np.linalg.inv(RGB_TO_YCBCR)
    img_shp_rgb = apply_matrix(img_ycbcr, ycbcr2rgb) / 256
    img_shp_rgb[img_shp_rgb < 0] = 0
    img_shp_rgb[img_shp_rgb > 1] = 1
    return img_shp_rgb

def tone_curve_correction(img_rgb, xs=(0, 64, 128, 192, 256), ys=(0, 64, 128, 192, 256)):
    func = scipy.interpolate.splrep(xs, ys)
    img_ycbcr = apply_matrix(img_rgb * 256, RGB_TO_YCBCR)
    img_ycbcr[:, :, 0] = scipy.interpolate.splev(img_ycbcr[:, :, 0], func)
    ycbcr2rgb = np.linalg.inv(RGB_TO_YCBCR)
    img_rgb_out = apply_matrix(img_ycbcr, ycbcr2rgb)
    return img_rgb_out / 256

def write(rgb_image, output_filename):
    """
    Write the processed RGB image to a specified file as PNG format.
    Thsi must be called after process(). No error is checked.
    :param output_filename: path to the output file. Extension must be png.
    """
    import imageio
    outimg = rgb_image.copy() * 256
    outimg[outimg < 0] = 0
    outimg[outimg > 255] = 255
    imageio.imwrite(output_filename, outimg.astype('uint8'))

def main(argv):
    """ main function """
    if (len(argv) < 2):
        print("Usage: {} input_filename [output_filename] \"color_matrix\" \"tone_x\" \"tone_y\"".format(argv[0]))
        print("\tDefault output_filename is output.png")
        print("\tDefault matrix is identity matrix \"1024, 0, 0, 0, 1024, 0, 0, 0, 1024\"")
        print("\tDefault tone curve is identity function \"0, 128, 256] [0, 128, 256\"")
        print("\tExample: python3 {} sample.ARW sample.png \"1141, -205, 88, -52, 1229, -154, 70, -225, 1179\" \"0, 72, 128, 200, 256\" \"0, 56, 128, 220, 256\"".format(argv[0]))
        print("\tSupported RAW format is ARW (Sony RAW) and Raspberry Pi (embedded in JPEG)")
        print("\tSupported output format is PNG only")
        return

    filename = argv[1]
    output_filename = "output.png"
    color_matrix = [1024, 0, 0, 0, 1024, 0, 0, 0, 1024]
    tone_curve = [(0, 64, 128, 192, 256), (0, 64, 128, 192, 256)]
    if len(argv) > 2:
        output_filename = argv[2]
    if len(argv) > 3:
        color_matrix = [int(value) for value in (argv[3]).split(',')]
    if len(argv) > 4:
        tone_curve[0] = [int(value) for value in (argv[4]).split(',')]
    if len(argv) > 5:
        tone_curve[1] = [int(value) for value in (argv[5]).split(',')]

    process(filename, output_filename, color_matrix, tone_curve)


if __name__ == "__main__":
    main(sys.argv)
