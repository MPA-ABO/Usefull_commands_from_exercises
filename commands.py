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

# %% SCIPY
import scipy
mat = scipy.io.loadmat('path_to_mat_file').get('label')

# %% SIMPLE ITK
import SimpleITK as sitk

image = sitk.ReadImage('path_to_image')
img_as_array = sitk.GetArrayFromImage(image)

# %% SKIMAGE
import skimage
image = skimage.io.imread('path_to_image')
gs_image = skimage.color.rgb2gray(image)
