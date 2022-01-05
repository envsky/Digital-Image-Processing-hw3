# For this part of the assignment, please implement your own code for all computations,
# Do not use inbuilt functions like fft from either numpy, opencv or other libraries
import numpy as np
import math


class Dft:
    def __init__(self):
        pass

    def forward_transform(self, matrix):
        """Computes the forward Fourier transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a complex matrix representing fourier transform"""

        shape = np.shape(matrix)
        n = shape[0]
        fft = np.zeros((n, n), dtype=complex)

        for u in range(0, n):
            for v in range(0, n):
                value = 0
                for i in range(0, n):
                    for j in range(0, n):
                        value += matrix[i][j] * (math.cos((2*math.pi/n)*(u*i+v*j)) - 1j*math.sin((2*math.pi/n)*(u*i+v*j)))
                fft[u][v] = value

        return fft

    def inverse_transform(self, matrix):
        """Computes the inverse Fourier transform of the input matrix
        You can implement the inverse transform formula with or without the normalizing factor.
        Both formulas are accepted.
        takes as input:
        matrix: a 2d matrix (DFT) usually complex
        returns a complex matrix representing the inverse fourier transform"""

        shape = np.shape(matrix)
        n = shape[0]
        inverse = np.zeros((n, n), dtype=complex)

        for i in range(0, n):
            for j in range(0, n):
                value = 0
                for u in range(0, n):
                    for v in range(0, n):
                        value += matrix[u][v] * (math.cos((2 * math.pi / n) * (u * i + v * j)) + 1j * math.sin(
                            (2 * math.pi / n) * (u * i + v * j)))
                inverse[i][j] = value

        return inverse

    def magnitude(self, matrix):
        """Computes the magnitude of the input matrix (iDFT)
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing magnitude of the complex matrix"""

        shape = np.shape(matrix)
        n = shape[0]

        mag = np.zeros((n, n))

        for u in range(0, n):
            for v in range(0, n):
                mag[u][v] = math.sqrt(math.pow(np.real(matrix[u][v]), 2) + math.pow(np.imag(matrix[u][v]), 2))

        return mag
