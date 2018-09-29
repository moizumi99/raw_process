import rawpy
import numpy as np
import math
import sys

class raw_process():
    """ Process RAW file into a image file.

    Example usage:
    rp = raw_process()
    rp.set_matrix([1024, 0, 0, 0, 1024, 0, 0, 0, 1024])
    rp.read("sample.ARW")
    rp.process()
    rp.write("output.ARW")

    Attributes:
        color_matrix: color matrix for color correction. See the setter method.

    """

    def __init__(self):
        self.color_matrix = [1024, 0, 0, 0, 1024, 0, 0, 0, 1024]
        self._rgb_image = None
        self._raw = None

    def set_matrix(self, color_matrix):
        """
        Setter for color_matrix.
        Matrix size is 3x3 represented in size 9 array of int.
        The values are normalized to 1024.

        :param color_matrix: int array of size 9, which represents 3x3 color matrix
        """
        self.color_matrix = color_matrix

    def read(self, filename):
        """
        Read RAW data from specified file. Currently supported formats are
            ARW (Sony RAW format)

        :param filename: path to the target RAW file
        """
        self._raw = rawpy.imread(filename)

    def process(self):
        """
        This processes RAW data that was read by read() method.
        Must be called after read() operation. No error is checked.
        """
        raw_array = self._get_raw()
        blc_raw = self._black_level_correction(raw_array)
        dms_img = self._preview_demosaic(blc_raw)
        img_wb = self._white_balance(dms_img)
        img_ccm = self._color_correction_matrix(img_wb)
        img_gamma = self._gamma_correction(img_ccm)

        self._rgb_image = img_gamma
        self._rgb_image_set = True

    def write(self, output_filename):
        """
        Write the processed RGB image to a specified file as PNG format.
        Thsi must be called after process(). No error is checked.
        :param output_filename: path to the output file. Extension must be png.
        """
        import imageio
        outimg = self._rgb_image.copy()
        outimg[outimg < 0] = 0
        outimg = outimg / outimg.max() * 255
        imageio.imwrite(output_filename, outimg.astype('uint8'))

    def _get_raw(self):
        self._h, self._w = self._raw.sizes.raw_height, self._raw.sizes.raw_width
        h, w = self._h, self._w
        raw_array = np.array(self._raw.raw_image).reshape((h, w)).astype('float')
        return raw_array

    def _black_level_correction(self, raw_array):
        blc = self._raw.black_level_per_channel
        bayer_pattern = self._raw.raw_pattern
        blc_raw = raw_array.copy()
        h, w = self._h, self._w
        for y in range(0, h, 2):
            for x in range(0, w, 2):
                colors = [0, 0, 0]
                blc_raw[y + 0, x + 0] -= blc[bayer_pattern[0, 0]]
                blc_raw[y + 0, x + 1] -= blc[bayer_pattern[0, 1]]
                blc_raw[y + 1, x + 0] -= blc[bayer_pattern[1, 0]]
                blc_raw[y + 1, x + 1] -= blc[bayer_pattern[1, 1]]
        return blc_raw

    def _preview_demosaic(self, raw_array):
        bayer_pattern = self._raw.raw_pattern
        h, w = self._h, self._w
        dms_img = np.zeros((h // 2, w // 2, 3))
        for y in range(0, h, 2):
            for x in range(0, w, 2):
                colors = [0, 0, 0, 0]
                colors[bayer_pattern[0, 0]] += raw_array[y + 0, x + 0]
                colors[bayer_pattern[0, 1]] += raw_array[y + 0, x + 1]
                colors[bayer_pattern[1, 0]] += raw_array[y + 1, x + 0]
                colors[bayer_pattern[1, 1]] += raw_array[y + 1, x + 1]
                dms_img[y // 2, x // 2, 0] = colors[0]
                dms_img[y // 2, x // 2, 1] = (colors[1] + colors[3]) / 2
                dms_img[y // 2, x // 2, 2] = colors[2]
        return dms_img

    def _white_balance(self, raw_array):
        wb = np.array(self._raw.camera_whitebalance)
        img_wb = np.zeros_like(raw_array).reshape((-1, 3))
        for index, pixel in enumerate(raw_array.reshape(-1, 3)):
            pixel = pixel * wb[:3] / 1024
            img_wb[index] = pixel
        return img_wb.reshape(raw_array.shape)

    def _color_correction_matrix(self, raw_array):
        img_ccm = np.zeros_like(raw_array).reshape((-1, 3))
        ccm = np.array(self.color_matrix).reshape((3, 3))
        for index, pixel in enumerate(raw_array.reshape((-1, 3))):
            pixel = np.dot(ccm, pixel)
            img_ccm[index] = pixel
        return img_ccm.reshape(raw_array.shape)

    def _gamma_correction(self, raw_array):
        img_gamma = raw_array.copy().flatten()
        img_gamma[img_gamma < 0] = 0
        img_gamma = img_gamma / img_gamma.max()
        for index, val in enumerate(img_gamma):
            img_gamma[index] = math.pow(val, 1 / 2.2)
        return img_gamma.reshape(raw_array.shape)



def main(argv):
    if (len(argv) < 2):
        print("Usage: {} input_filename [output_filename] [color_matrix]".format(argv[0]))
        print("\tDefault output_filename is output.png")
        print("\tDefault matrix is identity matrix ([1024, 0, 0, 0, 1024, 0, 0, 0, 1024]")
        print("\tExample: python3 {} sample.ARW sample.png \"1141, -205, 88, -52, 1229, -154, 70, -225, 1179\"".format(argv[0]))
        print("\tSupported RAW format is ARW (Sony RAW)")
        return

    filename = argv[1]
    output_filename = "output.png"
    color_matrix = [1024, 0, 0, 0, 1024, 0, 0, 0, 1024]
    if len(argv) > 2:
        output_filename = argv[2]
    if len(argv) > 3:
        color_matrix = [int(value) for value in (argv[3]).split(',')]

    rp = raw_process()
    rp.set_matrix(color_matrix)
    rp.read(filename)
    rp.process()
    rp.write(output_filename)


if __name__ == "__main__":
    main(sys.argv)