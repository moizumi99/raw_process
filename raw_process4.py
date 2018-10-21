import rawpy
import numpy as np
import sys
from scipy import signal

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

    :param filename: path to the target RAW file
    """
    return rawpy.imread(filename)


def process(raw, color_matrix=(1024, 0, 0, 0, 1024, 0, 0, 0, 1024)):
    """
    This processes RAW data that was read by read() method.
    Must be called after read() operation. No error is checked.
    """
    raw_array = get_raw_array(raw)
    bayer_pattern = raw.raw_pattern

    # rearrange black level
    black_level = [0] * 4
    black_level[bayer_pattern[0, 0]] = raw.black_level_per_channel[bayer_pattern[0, 0]]
    black_level[bayer_pattern[0, 1]] = raw.black_level_per_channel[bayer_pattern[0, 1]]
    black_level[bayer_pattern[1, 0]] = raw.black_level_per_channel[bayer_pattern[1, 0]]
    black_level[bayer_pattern[1, 1]] = raw.black_level_per_channel[bayer_pattern[1, 1]]

    blc_raw = black_level_correction(raw_array, black_level)

    wbg = np.array(raw.camera_whitebalance) / 1024
    wb_raw = white_balance_Bayer(blc_raw, wbg, bayer_pattern)
    
    dms_img = advanced_demosaic(wb_raw, bayer_pattern)
    img_ccm = color_correction_matrix(dms_img, color_matrix)
    img_gamma = gamma_correction(img_ccm, 2.2)
    return img_gamma


def write(rgb_image, output_filename):
    """
    Write the processed RGB image to a specified file as PNG format.
    Thsi must be called after process(). No error is checked.
    :param output_filename: path to the output file. Extension must be png.
    """
    import imageio
    outimg = rgb_image.copy()
    outimg[outimg < 0] = 0
    outimg = outimg / outimg.max() * 255
    imageio.imwrite(output_filename, outimg.astype('uint8'))


def get_raw_array(raw):
    h, w = raw.sizes.raw_height, raw.sizes.raw_width
    raw_array = np.array(raw.raw_image).reshape((h, w)).astype('float')
    return raw_array


def black_level_correction(raw_array, black_level):
    blc_raw = raw_array.copy()
    blc_raw[0::2, 0::2] -= black_level[0]
    blc_raw[0::2, 1::2] -= black_level[1]
    blc_raw[1::2, 0::2] -= black_level[2]
    blc_raw[1::2, 1::2] -= black_level[3]
    return blc_raw


def preview_demosaic(raw_array, bayer_pattern):
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

def white_balance_Bayer(raw_array, wbg, bayer_pattern):
    img_wb = raw_array.copy()
    img_wb[0::2, 0::2] *= wbg[bayer_pattern[0, 0]] 
    img_wb[0::2, 1::2] *= wbg[bayer_pattern[0, 1]]
    img_wb[1::2, 0::2] *= wbg[bayer_pattern[1, 0]]
    img_wb[1::2, 1::2] *= wbg[bayer_pattern[1, 1]]
    return img_wb


def color_correction_matrix(rgb_array, color_matrix):
    img_ccm = np.zeros_like(rgb_array)
    ccm = np.array(color_matrix).reshape((3, 3))
    norm = ccm.sum(axis=1).mean()
    for c in (0, 1, 2):
        img_ccm[:, :, c] = ccm[c, 0] * rgb_array[:, :, 0] + \
                           ccm[c, 1] * rgb_array[:, :, 1] + \
                           ccm[c, 2] * rgb_array[:, :, 2]
    return img_ccm / norm


def gamma_correction(rgb_array, gamma):
    img_gamma = rgb_array.copy()
    img_gamma[img_gamma < 0] = 0
    img_gamma = img_gamma / img_gamma.max()
    img_gamma = np.power(img_gamma, 1/gamma)
    return img_gamma


def main(argv):
    if (len(argv) < 2):
        print("Usage: {} input_filename [output_filename] [color_matrix]".format(argv[0]))
        print("\tDefault output_filename is output.png")
        print("\tDefault matrix is identity matrix ([1024, 0, 0, 0, 1024, 0, 0, 0, 1024]")
        print("\tExample: python3 {} sample.ARW sample.png \"1141, -205, 88, -52, 1229, -154, 70, -225, 1179\"".format(argv[0]))
        print("\tSupported RAW format is ARW (Sony RAW)")
        print("\tSupported output format is PNG only")
        return

    filename = argv[1]
    output_filename = "output.png"
    color_matrix = [1024, 0, 0, 0, 1024, 0, 0, 0, 1024]
    if len(argv) > 2:
        output_filename = argv[2]
    if len(argv) > 3:
        color_matrix = [int(value) for value in (argv[3]).split(',')]

    color_matrix = [1024, 0, 0, 0, 1024, 0, 0, 0, 1024]
    raw = read(filename)
    rgb_image = process(raw, color_matrix)
    write(rgb_image, output_filename)


if __name__ == "__main__":
    main(sys.argv)
