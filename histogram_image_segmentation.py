from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_float, img_as_ubyte,io
import numpy as np
from matplotlib import pyplot as plt

img = img_as_float(io.imread("images/noisy_image1.jpg"))

sigma_est = np.mean(estimate_sigma(img, multichannel=True))

denoise = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=True,
                           patch_size=5, patch_distance=3,multichannel=True)

denoise_ubyte = img_as_ubyte(denoise)
plt.imshow(denoise_ubyte, cmap='gray')

plt.hist(denoise_ubyte.flat, bins=100, range=(0,80))

segment1= (denoise_ubyte <= 55)
segment2= (denoise_ubyte > 55) & (denoise_ubyte <= 110)
segment3= (denoise_ubyte > 110) & (denoise_ubyte <= 210)
segment4= (denoise_ubyte >= 210)

all_segments= np.zeros((denoise_ubyte.shape[0], denoise_ubyte.shape[1], 3))

all_segments[segment1] = (1,0,0)
all_segments[segment2] = (0,1,0)
all_segments[segment3] = (0,0,1)
all_segments[segment4] = (1,1,0)

#plt.imshow(all_segments)

from scipy import ndimage as nd

segment1_opened = nd.binary_opening(segment1, np.ones((3,3)))
segment1_closed = nd.binary_closing(segment1_opened, np.ones((3,3)))

segment2_opened = nd.binary_opening(segment2, np.ones((3,3)))
segment2_closed = nd.binary_closing(segment2_opened, np.ones((3,3)))

segment3_opened = nd.binary_opening(segment3, np.ones((3,3)))
segment3_closed = nd.binary_closing(segment3_opened, np.ones((3,3)))

segment4_opened = nd.binary_opening(segment4, np.ones((3,3)))
segment4_closed = nd.binary_closing(segment4_opened, np.ones((3,3)))

all_segments_cleaned= np.zeros((denoise_ubyte.shape[0], denoise_ubyte.shape[1], 3))

all_segments_cleaned[segment1_closed] = (1,0,0)
all_segments_cleaned[segment2_closed] = (0,1,0)
all_segments_cleaned[segment3_closed] = (0,0,1)
all_segments_cleaned[segment4_closed] = (1,1,0)


plt.imshow(all_segments_cleaned)

plt.imsave("images/segmented_image1.jpg",all_segments_cleaned)
