import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load the input image

img = cv2.imread('input_image2.jpg')

# Convert the input image to grayscale

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Fast Fourier Transform (FFT) to the grayscale image

f = np.fft.fft2(gray)
fshift = np.fft.fftshift(f)


# Apply a high-pass filter to the Fourier Transfor

rows, cols = gray.shape
crow, ccol = int(rows/2), int(cols/2)
fshift[crow-50:crow+50, ccol-50:ccol+50] = 0

# Apply Inverse Fast Fourier Transform (IFFT) to the filtered Fourier Transform image to get the final output image

f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# Plot the input image, Fourier Transform image, and output image using Matplotlib

plt.subplot(131), plt.imshow(gray, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])


plt.subplot(133), plt.imshow(img_back, cmap='gray')
plt.title('Output Image'), plt.xticks([]), plt.yticks([])

plt.show()