import numpy as np
import math


class Filtering:

    def __init__(self, image):
        self.image = image

    def get_gaussian_filter(self):
        """Initialzes and returns a 5X5 Gaussian filter
            Use the formula for a 2D gaussian to get the values for a 5X5 gaussian filter
        """

        ftr = np.zeros((5, 5))
        sigma = 1

        for i in range(0, 5):
            for j in range(0, 5):
                y = i - 2
                x = j - 2
                ftr[i][j] = (1/(2*math.pi*sigma*sigma)) * math.exp(-((x*x+y*y)/2*sigma*sigma))

        return ftr

    def get_laplacian_filter(self):
        """Initialzes and returns a 3X3 Laplacian filter"""

        ftr = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]

        return ftr

    def filter(self, filter_name):
        """Perform filtering on the image using the specified filter, and returns a filtered image
            takes as input:
            filter_name: a string, specifying the type of filter to use ["gaussian", laplacian"]
            return type: a 2d numpy array
                """
        shape = np.shape(self.image)
        ftr_img = np.zeros((shape[0], shape[1]))

        if filter_name == "gaussian":
            padding = 4
            ftr = self.get_gaussian_filter()
            padding_img = np.zeros((shape[0]+padding*2, shape[1]+padding*2))

            for i in range(0, shape[0]):
                for j in range(0, shape[1]):
                    padding_img[i+padding][j+padding] = self.image[i][j]
            for i in range(0, shape[0]):
                for j in range(0, shape[1]):
                    y = i + padding
                    x = j + padding
                    intensity = 0
                    for m in range(0, padding+1):
                        for n in range(0, padding+1):
                            intensity += ftr[m][n] * padding_img[y-(padding-m)][x-(padding-n)]
                    ftr_img[i][j] = intensity

        if filter_name == "laplacian":
            padding = 2
            ftr = self.get_laplacian_filter()
            padding_img = np.zeros((shape[0] + padding * 2, shape[1] + padding * 2))

            for i in range(0, shape[0]):
                for j in range(0, shape[1]):
                    padding_img[i + padding][j + padding] = self.image[i][j]
            for i in range(0, shape[0]):
                for j in range(0, shape[1]):
                    y = i + padding
                    x = j + padding
                    intensity = 0
                    for m in range(0, padding + 1):
                        for n in range(0, padding + 1):
                            intensity += ftr[m][n] * padding_img[y - (padding - m)][x - (padding - n)]
                    ftr_img[i][j] = intensity

        return ftr_img

