var_type = type(variable)
print(f'String to be shown {variable_value_to_be_shown}')

# %% IMAGEIO
import imageio.v3 as iio

image = iio.imread('path_to_image')

# %% MATPLOTLIB
import matplotlib.pyplot as plt

image = plt.imread('path_to_image')

plt.figure()
plt.subplot(1,2,1)
plt.imshow(image1)
plt.title('Title of the first image')
plt.xlabel('X axis label [units]')
plt.ylabel('Y axis label [units]')

plt.subplot(1,2,2)
plt.hist(image1.ravel())
plt.title('Title of the second image')
plt.xlabel('X axis label [units]')
plt.ylabel('Y axis label [units]')
plt.show()

plt.close('all')

# %% NUMPY

import numpy as np

var_d_type = variable.dtype
image = np.zeros_like(image)
image = np.ones(x_size, y_size)
img_shape = np.shape(image)
vector = np.linspace(start = start, stop = stop, num = number_of_elements)

spectrum = amplitude_spectrum * np.exp(1j*phase_spectrum)
spectrum_1 = np.log(np.fft.fftshift(np.abs(np.fft.fft2(image))))
spectrum_2 = np.fft.fftshift(np.angle(np.fft.fft2(image)))
image = np.real(np.fft.ifft2(spectrum))

# %% SCIPY
import scipy
mat = scipy.io.loadmat('path_to_mat_file').get('label')
filtered_img = scipy.ndimage.gaussian_filter(input = img_noisy, sigma = sigma)
filtered_img = scipy.signal.convolve2d(img_noisy, psf, mode = 'valid')
filtered_img = scipy.signal.medfilt2d(input = img_noisy, kernel_size = kernel_size)

# %% SIMPLE ITK
import SimpleITK as sitk

image = sitk.ReadImage('path_to_image')
img_as_array = sitk.GetArrayFromImage(image)

# %% SKIMAGE
import skimage
image = skimage.io.imread('path_to_image')
gs_image = skimage.color.rgb2gray(image)
img_in_float_from_0_to_1 = skimage.img_as_float(gs_image)
image = skimage.transform.rotate(image,angle)


