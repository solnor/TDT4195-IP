import skimage
import skimage.io
import skimage.transform
import os
import numpy as np
import utils
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # DO NOT CHANGE
    impath = os.path.join("images", "noisy_moon.png")
    im = utils.read_im(impath)

    # START YOUR CODE HERE ### (You can change anything inside this block)
    
    kernel = np.ones_like(im)
    # points = [y-coord,x-coord,radius]
    r = 6
    points = [
            [269, 0,   r],
            [269, 29,  r],
            [269, 58,  r],
            [269, 88,  r],
            [269, 116, r],
            [269, 145, r],
            [269, 174, r],
            [269, 202, 3],
            [269, 261, 3],
            [269, 290, r],
            [269, 320, r],
            [269, 348, r],
            [269, 377, r],
            [269, 406, r],
            [269, 436, r],
            ]
    
    for point in points:
        rr, cc = skimage.draw.disk((point[0], point[1]), point[2])
        kernel[rr,cc] = 0.0
    kernel_fft = np.fft.fftshift(kernel)

    # Fourier transform image, shift the frequencies to centre
    im_fft = np.fft.fft2(im)
    im_fft = np.fft.fftshift(im_fft)
    
    # Shift the frequencies of the kernel
    kernel_fft = np.fft.fftshift(kernel_fft)


    # Convolve the image with the kernel in the frequency domain
    conv_result_fft = np.multiply(im_fft, kernel_fft)
    # Shift the frequencies back before inverse Fourier transforming
    conv_result_fft = np.fft.fftshift(conv_result_fft)
    im_filtered = np.fft.ifft2(conv_result_fft).real
    # Shift frequencies s.t. zero-frequency is in centre again
    conv_result_fft = np.fft.fftshift(conv_result_fft)

    # Get magnitude of images in frequency domain for visualisation. 
    im_fft          = np.log(np.abs(im_fft) + 1)
    fft_kernel      = np.log(np.abs(kernel_fft) + 1)
    conv_result_fft = np.log(np.abs(conv_result_fft) + 1)

    plt.figure(figsize=(20, 4))
    plt.subplot(1, 5, 1)
    plt.imshow(im, cmap="gray")

    plt.subplot(1, 5, 2)
    # Visualize FFT
    plt.imshow(im_fft, cmap="gray")
    plt.subplot(1, 5, 3)
    # Visualize FFT kernel
    plt.imshow(fft_kernel, cmap="gray")
    
    plt.subplot(1, 5, 4)
    # Visualize filtered FFT image
    plt.imshow(conv_result_fft, cmap="gray")
    
    plt.subplot(1, 5, 5)
    # Visualize filtered spatial image
    plt.imshow(im_filtered, cmap="gray")
    plt.show()
    ### END YOUR CODE HERE ###
    utils.save_im("moon_filtered.png", utils.normalize(im_filtered))
