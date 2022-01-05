# For this part of the assignment, You can use inbuilt functions to compute the fourier transform
# You are welcome to use fft that are available in numpy and opencv
import cv2
import numpy as np

class Filtering:

    def __init__(self, image):
        """initializes the variables frequency filtering on an input image
        takes as input:
        image: the input image
        """
        self.image = image
        self.mask = self.get_mask

    def get_mask(self, shape):
        """Computes a user-defined mask
        takes as input:
        shape: the shape of the mask to be generated
        rtype: a 2d numpy array with size of shape
        """

        mask = np.zeros(shape)
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                mask[j][i] = 1

        image = cv2.circle(mask, (283, 229), 8, 0, -1)
        image = cv2.circle(image, (229, 283), 8, 0, -1)

        return image

    def post_process_image(self, image):
        """Post processing to display DFTs and IDFTs
        takes as input:
        image: the image obtained from the inverse fourier transform
        return an image with full contrast stretch
        -----------------------------------------------------
        You can perform post processing as needed. For example,
        1. You can perfrom log compression
        2. You can perfrom a full contrast stretch (fsimage)
        3. You can take negative (255 - fsimage)
        4. etc.
        """

        log = np.log(image)
        shape = np.shape(log)
        fsimage = np.zeros(shape, dtype=np.uint8)

        min = log[0][0]
        max = log[0][0]

        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                if log[i][j] < min:
                    min = log[i][j]
                if log[i][j] > max:
                    max = log[i][j]

        p = 255 / (max - min)
        l = (0-min) * p

        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                fsimage[i][j] = p * log[i][j] + l

        return fsimage

    def filter(self):
        """Performs frequency filtering on an input image
        returns a filtered image, magnitude of DFT, magnitude of filtered DFT
        ----------------------------------------------------------
        You are allowed to used inbuilt functions to compute fft
        There are packages available in numpy as well as in opencv
        Steps:
        1. Compute the fft of the image
        2. shift the fft to center the low frequencies
        3. get the mask (write your code in functions provided above) the functions can be called by self.filter(shape)
        4. filter the image frequency based on the mask (Convolution theorem)
        5. compute the inverse shift
        6. compute the inverse fourier transform
        7. compute the magnitude
        8. You will need to do post processing on the magnitude and depending on the algorithm (use post_process_image to write this code)
        Note: You do not have to do zero padding as discussed in class, the inbuilt functions takes care of that
        filtered image, magnitude of DFT, magnitude of filtered DFT: Make sure all images being returned have grey scale full contrast stretch and dtype=uint8
        """

        shape = np.shape(self.image)
        fft = np.fft.fft2(self.image)
        fft_shift = np.fft.fftshift(fft)

        mag_dft = self.post_process_image(np.abs(fft_shift))
        mag_fdft = np.zeros(np.shape(mag_dft))

        mask = self.get_mask(shape)
        filtered_fft = np.zeros(shape, dtype=complex)

        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                filtered_fft[i][j] = fft_shift[i][j] * mask[i][j]
                mag_fdft[i][j] = mag_dft[i][j] * mask[i][j]

        inverse_shift = np.fft.ifftshift(filtered_fft)
        inverse_fft = np.fft.ifft2(inverse_shift)

        img = np.abs(inverse_fft)

        return [img, mag_dft, mag_fdft]
